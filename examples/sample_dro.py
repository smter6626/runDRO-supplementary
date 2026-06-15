"""Distributionally-Robust Optimization of Probabilistic Circuits.

Solves the saddle-point problem

    sup_theta  inf_phi  max_{lambda>=0}
        log(E_{Q_phi}[P_theta(X)]) + lambda * (CW(P_hat, Q_phi) - eps)

with the updates

    theta  <- theta  + eta_theta  * grad_theta  E_Q[log(P)]
    phi    <- phi    - eta_phi   * [grad_phi log(E_Q[P]) + lambda * grad_phi CW(P_hat, Q)]
    lambda <- max(0, lambda + eta_lambda * (CW(P_hat, Q) - eps))

The theta ascent direction grad_theta E_Q[log(P)] is estimated by drawing samples
from Q_phi (ancestral sampling via Circuit.sample) and differentiating the mean
log-likelihood of those samples under P_theta (Circuit.mean_log_likelihood_and_grad).
The phi and lambda updates are unchanged and still use the log(E_Q[P]) objective.

P_hat is the fixed empirical PC. P_theta and Q_phi are deep copies of P_hat that
share its structure but carry independently learnable parameters. For every outer
iteration of theta we run K inner iterations of (phi, lambda).

Tools used (all differentiable):
    log_exp_query_and_grad(P, Q) -> (log E, grad_wrt_P, grad_wrt_Q)
    cw_distance_and_grad(P_hat, Q) -> (CW, grad_wrt_Q)
    Circuit.sample(n, seed) -> list of {var: value} samples from Q
    Circuit.mean_log_likelihood_and_grad(data) -> (mean log P, grad_wrt_P)
"""

import time

import gurobipy as gp
import numpy as np

from fastcircuits import (
    CategoricalInputNode,
    Circuit,
    ProductNode,
    SumNode,
    cw_distance,
    cw_distance_and_grad,
    log_exp_query,
    log_exp_query_and_grad,
)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
P_HAT_PATH = "test_circuit_1.json"  # fixed empirical PC P_hat

NUM_P_ITERS = 200  # outer iterations updating theta (P)
K = 5             # inner iterations updating phi (Q) and lambda per theta step

ETA_THETA = 1e-2   # learning rate for theta (P, ascent on E_Q[log P])
ETA_PHI = 5e-2     # learning rate for phi   (Q, descent)
ETA_LAMBDA = 20.0  # learning rate for lambda (large is safe: blend bounds the step)

# theta update is a sample-based estimate of grad_theta E_Q[log P]: draw
# THETA_NUM_SAMPLES samples from Q_phi and ascend the mean log-likelihood under
# P_theta. THETA_SEED=None reseeds every outer iter (fresh MC noise); set an int
# for reproducibility.
THETA_NUM_SAMPLES = 1000
THETA_SEED = None

# Scale-invariant gradient balancing for the phi update. When True the two
# gradient directions (log E and CW) are each normalized to unit L2 norm and
# combined as a CONVEX BLEND with weight w = lam / (1 + lam) on the CW direction:
#
#     g_phi = (1 - w) * grad_logE_hat + w * grad_CW_hat
#
# This keeps ||g_phi|| <= ~1 regardless of lam, so the effective phi step stays
# ~ETA_PHI no matter how large lam grows. Without it the combined gradient norm
# scales like lam, the effective learning rate becomes ETA_PHI * lam, and large
# lam (e.g. 184) overshoots -> Q bounces around simplex vertices, CW stays above
# eps on average, and the integral lam update winds up without bound.
NORMALIZE_PHI_GRADS = True

EPSILON = 0.01      # Wasserstein-ball radius (constraint: CW(P_hat, Q) < eps)
LAMBDA_INIT = 0.0  # initial Lagrange multiplier
LAMBDA_MAX = 100.0  # safety cap on lambda (anti-windup); w=lam/(1+lam) -> 0.99
LAMBDA_LEAK = 1.0  # multiplicative leak on lambda each update (e.g. 0.999); 1.0 disables
ALPHA = 1       # penalty factor for the constraint

METRIC_P = 1.0     # CW order
SCALE_FACTOR = 256 * 784   # CW scale factor
PROB_FLOOR = 1e-20

cw_kw = dict(metric_p=METRIC_P, scale_factor=SCALE_FACTOR)


# ----------------------------------------------------------------------------
# Circuit traversal + simplex projection helpers
# ----------------------------------------------------------------------------
def iter_circuit_nodes(root):
    """Post-order DFS over the circuit DAG (each node yielded once)."""
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        yield node
        if isinstance(node, (SumNode, ProductNode)):
            stack.extend(node.children())


def project_simplex(x):
    """Euclidean projection of a vector onto the probability simplex."""
    x = np.asarray(x, dtype=np.float64)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(x) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(x - theta, 0.0)


def simplex_step(params, grad, lr, *, ascent):
    """One projected-gradient step on the simplex (ascent or descent)."""
    p = np.asarray(params, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    sign = 1.0 if ascent else -1.0
    return project_simplex(p + sign * lr * g).tolist()


def apply_step(circuit, sum_grads, cat_grads, lr, *, ascent):
    """Apply one projected-gradient step to every learnable node in `circuit`."""
    for node in iter_circuit_nodes(circuit.root):
        nid = int(node.id)
        if isinstance(node, SumNode) and nid in sum_grads:
            node.set_parameters_list(
                simplex_step(node.parameters_list(), sum_grads[nid], lr, ascent=ascent)
            )
        elif isinstance(node, CategoricalInputNode) and nid in cat_grads:
            node.set_probabilities_list(
                simplex_step(node.probabilities_list(), cat_grads[nid], lr, ascent=ascent)
            )


def global_grad_norm(grads):
    """L2 norm of a GCWGradients object over all sum/categorical entries."""
    sq = 0.0
    for d in (grads.sum_grads, grads.cat_grads):
        for v in d.values():
            a = np.asarray(v, dtype=np.float64)
            sq += float(a @ a)
    return sq ** 0.5


def combine_phi_grads(logexp_grads, cw_grads, lam, *, normalize):
    """Combined phi descent direction, keyed by node id.

    With ``normalize=True`` each direction is rescaled to unit norm (s_E, s_C)
    and blended convexly with weight ``w = lam / (1 + lam)`` on the CW direction:

        g = (1 - w) * s_E * grad_logE + w * s_C * grad_CW

    so ``||g|| <= ~1`` and the effective phi step is ~ETA_PHI independent of lam.
    With ``normalize=False`` it falls back to the raw Lagrangian gradient
    ``grad_logE + lam * grad_CW``.
    """
    if normalize:
        n_e = global_grad_norm(logexp_grads)
        n_c = global_grad_norm(cw_grads)
        s_e = 1.0 / n_e if n_e > 0.0 else 0.0
        s_c = 1.0 / n_c if n_c > 0.0 else 0.0  # 0 at CW minimum (Q == P_hat)
        w = lam / (1.0 + lam)  # weight on CW direction, saturates at 1
        c_e = (1.0 - w) * s_e
        c_c = w * s_c
    else:
        c_e = 1.0
        c_c = lam

    sum_g, cat_g = {}, {}
    for out, le_dict, cw_dict in (
        (sum_g, logexp_grads.sum_grads, cw_grads.sum_grads),
        (cat_g, logexp_grads.cat_grads, cw_grads.cat_grads),
    ):
        for nid in set(le_dict) | set(cw_dict):
            le = le_dict.get(nid)
            cw = cw_dict.get(nid)
            le = 0.0 if le is None else np.asarray(le, dtype=np.float64)
            cw = 0.0 if cw is None else np.asarray(cw, dtype=np.float64)
            out[nid] = c_e * le + c_c * cw
    return sum_g, cat_g


# ----------------------------------------------------------------------------
# Setup: P_hat fixed, P_theta and Q_phi as independent deep copies
# ----------------------------------------------------------------------------
env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()
cw_kw["gurobi_env"] = env

P_hat = Circuit.load(P_HAT_PATH)   # fixed empirical distribution
P_theta = Circuit.load(P_HAT_PATH)  # learnable copy (theta)
Q_phi = Circuit.load(P_HAT_PATH)    # learnable copy (phi)

lam = float(LAMBDA_INIT)

init_log_exp = log_exp_query(P_theta, Q_phi)
init_cw = cw_distance(P_hat, Q_phi, **cw_kw)
print(f"P_hat: {P_HAT_PATH}")
print(f"K={K}  NUM_P_ITERS={NUM_P_ITERS}  eps={EPSILON}")
print(f"eta_theta={ETA_THETA}  eta_phi={ETA_PHI}  eta_lambda={ETA_LAMBDA}")
print(
    f"Initial: log(E)={init_log_exp:.8f}  CW={init_cw:.8f}  "
    f"violation={init_cw - EPSILON:+.8f}  lambda={lam:.4f}\n"
)


def objective(log_exp_val, cw_val, lam):
    """log(E_Q[P]) + lambda * (CW(P_hat, Q) - eps)."""
    return log_exp_val + lam * ALPHA * (cw_val - EPSILON)


# ----------------------------------------------------------------------------
# Optimization loop
# ----------------------------------------------------------------------------
t_start = time.perf_counter()
for p_iter in range(1, NUM_P_ITERS + 1):
    # --- K inner iterations: update phi (Q) and lambda ---
    for k in range(1, K + 1):
        log_exp_val, _, grad_phi = log_exp_query_and_grad(P_theta, Q_phi)
        cw_val, cw_grads = cw_distance_and_grad(P_hat, Q_phi, **cw_kw)

        # lambda <- max(0, lambda + eta_lambda * (CW - eps))   [ascent first so
        # the penalty is active on this same phi step instead of lagging by one]
        violation = ALPHA * (cw_val - EPSILON)
        lam = min(LAMBDA_MAX, max(0.0, LAMBDA_LEAK * lam + ETA_LAMBDA * violation))

        # phi <- phi - eta_phi * [grad_phi log E + lambda * grad_phi CW]
        sum_g, cat_g = combine_phi_grads(
            grad_phi, cw_grads, lam, normalize=NORMALIZE_PHI_GRADS
        )
        apply_step(Q_phi, sum_g, cat_g, ETA_PHI, ascent=False)

        obj = objective(log_exp_val, cw_val, lam)
        print(
            f"  [P {p_iter:3d}/{NUM_P_ITERS} | Q {k:2d}/{K}] "
            f"violation={violation:+.8f}  log(E)={log_exp_val:.8f}  "
            f"objective={obj:.8f}  lambda={lam:.6f}"
        )

    # --- one outer iteration: update theta (P), ascent on E_Q[log P] ---
    # Sample from Q_phi, then differentiate the mean log-likelihood of those
    # samples under P_theta (Monte-Carlo estimate of grad_theta E_Q[log P]).
    q_samples = Q_phi.sample(THETA_NUM_SAMPLES, seed=THETA_SEED)
    mean_log_p, grad_theta = P_theta.mean_log_likelihood_and_grad(q_samples)
    apply_step(P_theta, grad_theta.sum_grads, grad_theta.cat_grads, ETA_THETA, ascent=True)

elapsed = time.perf_counter() - t_start

# ----------------------------------------------------------------------------
# Final report
# ----------------------------------------------------------------------------
final_log_exp = log_exp_query(P_theta, Q_phi)
final_cw = cw_distance(P_hat, Q_phi, **cw_kw)
final_obj = objective(final_log_exp, final_cw, lam)
print(
    f"\nFinal: log(E)={final_log_exp:.8f}  CW={final_cw:.8f}  "
    f"violation={final_cw - EPSILON:+.8f}  objective={final_obj:.8f}  lambda={lam:.6f}"
)
print(f"log(E) change: {final_log_exp - init_log_exp:+.8f}")
print(f"Total time: {elapsed:.3f}s")

env.dispose()

P_theta.save("dro_P_theta.json")
Q_phi.save("dro_Q_phi.json")
