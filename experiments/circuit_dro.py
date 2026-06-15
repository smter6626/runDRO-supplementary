"""Apply DRO to learned PCs under likelihood_learned_pcs/."""

import argparse
import csv
import time
from pathlib import Path

import gurobipy as gp
import matplotlib.pyplot as plt
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
from fastcircuits.circuits.circuit_serializer import CircuitSerializer

DEBD_DIR = Path("DEBD/datasets")
PC_DIR = Path("likelihood_learned_pcs")
ADV_DIR = Path("adversarially_perturbed_datasets")
OUT_DIR = Path("circuit_dro_learned_pcs")
BLOCK_SIZE = 4
SEED = 0
PROGRESS_EVERY = 1
CHECKPOINT_ITERS = 5
SAVE_CHECKPOINTS = False
WARM_START_ITERS = 30

NUM_Q_ITERS = 5
ETA_THETA = 1e-2
ETA_PHI = 1e-2
ETA_LAMBDA = 20.0
NORMALIZE_PHI_GRADS = True
LAMBDA_INIT = 0.0
LAMBDA_MAX = 1000.0
LAMBDA_LEAK = 1.0
ALPHA = 1
METRIC_P = 1.0
SCALE_FACTOR = 1
PROB_FLOOR = 1e-20


def debd_datasets() -> list[str]:
    return sorted(
        d.name
        for d in DEBD_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}.test.data").is_file()
    )


def pc_path(dataset: str) -> Path:
    return (
        PC_DIR
        / "hclt"
        / dataset
        / str(BLOCK_SIZE)
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )


def test_path(dataset: str) -> Path:
    return DEBD_DIR / dataset / f"{dataset}.test.data"


def adv_path(dataset: str, k: int) -> Path:
    return (
        ADV_DIR
        / "hclt"
        / dataset
        / str(BLOCK_SIZE)
        / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}_dataset_K{k}.data"
    )


def load_dataset(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        return np.array([list(map(int, row)) for row in csv.reader(f)], dtype=np.int8)


def datapoint_to_assignment(x: np.ndarray) -> dict[int, int]:
    return {i: int(x[i]) for i in range(len(x))}


def mean_log_likelihood(pc: Circuit, data: np.ndarray) -> float:
    total = 0.0
    for row in data:
        total += pc.log_likelihood(datapoint_to_assignment(row))
    return total / len(data)


def out_path(dataset: str, k: int) -> Path:
    return (
        OUT_DIR
        / "hclt"
        / dataset
        / str(BLOCK_SIZE)
        / f"K{k}"
        / f"hclt_{dataset}_blocksize{BLOCK_SIZE}_seed{SEED}.json"
    )


def checkpoint_path(dst: Path, iteration: int) -> Path:
    return dst.with_name(f"{dst.stem}_iter{iteration}{dst.suffix}")


def plot_path(dst: Path) -> Path:
    return dst.with_name(f"{dst.stem}_ll_plot.png")


def save_ll_plot(
    iters: list[int],
    ll_test: list[float],
    ll_adv: list[float],
    path: Path,
    *,
    k: int,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(iters, ll_test, label="LL_test")
    ax.plot(iters, ll_adv, label=f"LL_adv (K={k})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean log-likelihood")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


def project_to_simplex_tangent(grad):
    """Project an unconstrained gradient onto the simplex tangent space."""
    g = np.asarray(grad, dtype=np.float64)
    return g - g.mean()


def simplex_step(params, grad, lr, *, ascent):
    """One tangent-projected gradient step, then clip and renormalize."""
    p = np.asarray(params, dtype=np.float64)
    g = project_to_simplex_tangent(grad)
    sign = 1.0 if ascent else -1.0
    x = p + sign * lr * g
    x = np.clip(x, PROB_FLOOR, None)
    return (x / x.sum()).tolist()


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
    """Combined phi descent direction, keyed by node id."""
    if normalize:
        n_e = global_grad_norm(logexp_grads)
        n_c = global_grad_norm(cw_grads)
        s_e = 1.0 / n_e if n_e > 0.0 else 0.0
        s_c = 1.0 / n_c if n_c > 0.0 else 0.0
        w = lam / (1.0 + lam)
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


def clone_circuit(pc: Circuit) -> Circuit:
    return Circuit(CircuitSerializer.loads(CircuitSerializer.dumps(pc.root)))


def update_q_phi(
    p_theta: Circuit,
    q_phi: Circuit,
    p_hat: Circuit,
    lam: float,
    *,
    k: int,
    cw_kw: dict,
) -> float:
    """Run NUM_Q_ITERS inner phi/lambda updates; leave p_theta unchanged."""
    for _ in range(1, NUM_Q_ITERS + 1):
        _, _, grad_phi = log_exp_query_and_grad(p_theta, q_phi)
        cw_val, cw_grads = cw_distance_and_grad(p_hat, q_phi, **cw_kw)

        violation = ALPHA * (cw_val - k)
        lam = min(LAMBDA_MAX, max(0.0, LAMBDA_LEAK * lam + ETA_LAMBDA * violation))

        sum_g, cat_g = combine_phi_grads(
            grad_phi, cw_grads, lam, normalize=NORMALIZE_PHI_GRADS
        )
        apply_step(q_phi, sum_g, cat_g, ETA_PHI, ascent=False)
    return lam


def dro_progress_metrics(
    p_theta: Circuit,
    q_phi: Circuit,
    p_hat: Circuit,
    test_data: np.ndarray,
    adv_data: np.ndarray,
    *,
    k: int,
    cw_kw: dict,
) -> tuple[float, float, float, float]:
    log_exp_val = log_exp_query(p_theta, q_phi)
    cw_val = cw_distance(p_hat, q_phi, **cw_kw)
    ll_test = mean_log_likelihood(p_theta, test_data)
    ll_adv = mean_log_likelihood(p_theta, adv_data)
    return log_exp_val, cw_val, ll_test, ll_adv


def print_dro_progress(
    label: str,
    log_exp_val: float,
    cw_val: float,
    lam: float,
    ll_test: float,
    ll_adv: float,
    *,
    k: int,
) -> None:
    print(
        f"  [{label}] log(E)={log_exp_val:.6f}  "
        f"CW={cw_val:.6f}  violation={cw_val - k:+.6f}  lambda={lam:.4f}  "
        f"LL_test={ll_test:.6f}  LL_adv(K={k})={ll_adv:.6f}"
    )


def apply_dro(
    p_hat: Circuit,
    *,
    iters: int,
    k: int,
    test_data: np.ndarray,
    adv_data: np.ndarray,
    dst: Path,
) -> Circuit:
    """Run DRO on a fixed empirical PC and return the optimized P_theta."""
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    cw_kw = dict(metric_p=METRIC_P, scale_factor=SCALE_FACTOR, gurobi_env=env)

    p_theta = clone_circuit(p_hat)
    q_phi = clone_circuit(p_hat)
    lam = float(LAMBDA_INIT)

    init_log_exp = log_exp_query(p_theta, q_phi)
    init_cw = cw_distance(p_hat, q_phi, **cw_kw)
    init_ll_test = mean_log_likelihood(p_theta, test_data)
    init_ll_adv = mean_log_likelihood(p_theta, adv_data)
    iter_hist = [0]
    ll_test_hist = [init_ll_test]
    ll_adv_hist = [init_ll_adv]
    print(
        f"  Initial: log(E)={init_log_exp:.6f}  CW={init_cw:.6f}  "
        f"violation={init_cw - k:+.6f}  lambda={lam:.4f}  "
        f"LL_test={init_ll_test:.6f}  LL_adv(K={k})={init_ll_adv:.6f}"
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    ll_plot = plot_path(dst)
    t_start = time.perf_counter()

    if WARM_START_ITERS > 0:
        print(f"  Warm start: {WARM_START_ITERS} Q-only iteration(s)")
        for w_iter in range(1, WARM_START_ITERS + 1):
            lam = update_q_phi(p_theta, q_phi, p_hat, lam, k=k, cw_kw=cw_kw)
            if w_iter % PROGRESS_EVERY == 0 or w_iter == WARM_START_ITERS:
                log_exp_val = log_exp_query(p_theta, q_phi)
                cw_val = cw_distance(p_hat, q_phi, **cw_kw)
                print(
                    f"  [warm-start {w_iter}/{WARM_START_ITERS}] "
                    f"log(E)={log_exp_val:.6f}  CW={cw_val:.6f}  "
                    f"violation={cw_val - k:+.6f}  lambda={lam:.4f}"
                )

    for p_iter in range(1, iters + 1):
        lam = update_q_phi(p_theta, q_phi, p_hat, lam, k=k, cw_kw=cw_kw)

        _, grad_theta, _ = log_exp_query_and_grad(p_theta, q_phi)
        apply_step(
            p_theta, grad_theta.sum_grads, grad_theta.cat_grads, ETA_THETA, ascent=True
        )

        log_exp_val, cw_val, ll_test, ll_adv = dro_progress_metrics(
            p_theta, q_phi, p_hat, test_data, adv_data, k=k, cw_kw=cw_kw
        )
        iter_hist.append(p_iter)
        ll_test_hist.append(ll_test)
        ll_adv_hist.append(ll_adv)

        if p_iter % CHECKPOINT_ITERS == 0:
            save_ll_plot(iter_hist, ll_test_hist, ll_adv_hist, ll_plot, k=k)
            if SAVE_CHECKPOINTS:
                ckpt = checkpoint_path(dst, p_iter)
                p_theta.save(ckpt)
                print(f"  Checkpoint saved: {ckpt}  plot: {ll_plot}")
            else:
                print(f"  Plot saved: {ll_plot}")

        if p_iter % PROGRESS_EVERY == 0 or p_iter == iters:
            print_dro_progress(
                f"{p_iter}/{iters}",
                log_exp_val,
                cw_val,
                lam,
                ll_test,
                ll_adv,
                k=k,
            )

    elapsed = time.perf_counter() - t_start
    final_log_exp = log_exp_query(p_theta, q_phi)
    final_cw = cw_distance(p_hat, q_phi, **cw_kw)
    final_ll_test = mean_log_likelihood(p_theta, test_data)
    final_ll_adv = mean_log_likelihood(p_theta, adv_data)
    print(
        f"  Final: log(E)={final_log_exp:.6f}  CW={final_cw:.6f}  "
        f"violation={final_cw - k:+.6f}  lambda={lam:.4f}  "
        f"LL_test={final_ll_test:.6f}  LL_adv(K={k})={final_ll_adv:.6f}  "
        f"log(E) change={final_log_exp - init_log_exp:+.6f}  "
        f"LL_test change={final_ll_test - init_ll_test:+.6f}  "
        f"LL_adv change={final_ll_adv - init_ll_adv:+.6f}  time={elapsed:.1f}s"
    )

    if iters % CHECKPOINT_ITERS != 0:
        save_ll_plot(iter_hist, ll_test_hist, ll_adv_hist, ll_plot, k=k)
        print(f"  Plot saved: {ll_plot}")

    env.dispose()
    return p_theta


def generate_circuit(dataset: str, iters: int, k: int) -> None:
    src = pc_path(dataset)
    dst = out_path(dataset, k)
    if dst.exists():
        print(f"Skipping {dataset} ({dst} exists)")
        return
    if not src.exists():
        print(f"Skipping {dataset} (missing {src})")
        return

    test_file = test_path(dataset)
    adv_file = adv_path(dataset, k)
    if not test_file.exists():
        print(f"Skipping {dataset} (missing {test_file})")
        return
    if not adv_file.exists():
        print(f"Skipping {dataset} (missing {adv_file})")
        return

    test_data = load_dataset(test_file)
    adv_data = load_dataset(adv_file)
    pc = Circuit.load(src)
    dro_pc = apply_dro(
        pc, iters=iters, k=k, test_data=test_data, adv_data=adv_data, dst=dst
    )
    dro_pc.save(dst)
    print(f"Saved {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply DRO to learned PCs under likelihood_learned_pcs/."
    )
    parser.add_argument(
        "--iters",
        type=int,
        required=True,
        help="Outer theta iterations (dro NUM_P_ITERS).",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Wasserstein-ball radius / epsilon (integer).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="DEBD dataset names (default: all with test splits).",
    )
    args = parser.parse_args()

    if args.iters < 1:
        parser.error("--iters must be at least 1")
    if args.k < 0:
        parser.error("--k must be non-negative")

    datasets = args.datasets if args.datasets else debd_datasets()
    print(
        f"Running DRO on {len(datasets)} dataset(s), "
        f"iters={args.iters}, k={args.k}"
    )
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        generate_circuit(dataset, args.iters, args.k)


if __name__ == "__main__":
    main()
