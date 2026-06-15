"""Minimize E_Q[P(X)] between a fixed circuit1 and learnable circuit2.

With circuit1 fixed, each step evaluates ``log_exp_query_and_grad`` (log-space
for numerical stability) and applies projected simplex descent to every sum /
categorical parameter in circuit2. Minimizing log(E) is equivalent to
minimizing E for E > 0.
"""

import time

import numpy as np

from fastcircuits import (
    CategoricalInputNode,
    Circuit,
    ProductNode,
    SumNode,
    log_exp_query,
    log_exp_query_and_grad,
)

CIRCUIT_PATH_1 = "test_circuit_1.json"
CIRCUIT_PATH_2 = "test_circuit_2.json"

num_steps = 50
learning_rate = 1e-2
prob_floor = 1e-20


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


def project_simplex_tangent_step(params, grad, lr):
    """Strict Euclidean projection of a gradient step onto the probability simplex."""
    x = np.asarray(params, dtype=np.float64) + lr * np.asarray(grad, dtype=np.float64)

    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(x) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(x - theta, 0).tolist()


def apply_gradients_to_circuit2(circuit2, sum_grads, cat_grads, lr):
    """Update every learnable parameter node in circuit2 in place."""
    for node in iter_circuit_nodes(circuit2.root):
        nid = int(node.id)
        if isinstance(node, SumNode) and nid in sum_grads:
            params = node.parameters_list()
            node.set_parameters_list(
                project_simplex_tangent_step(params, sum_grads[nid], lr)
            )
        elif isinstance(node, CategoricalInputNode) and nid in cat_grads:
            probs = node.probabilities_list()
            node.set_probabilities_list(
                project_simplex_tangent_step(probs, cat_grads[nid], lr)
            )


circuit1 = Circuit.load(CIRCUIT_PATH_1)
circuit2 = Circuit.load(CIRCUIT_PATH_2)

t0 = time.perf_counter()
initial_log_exp = log_exp_query(circuit1, circuit2)
print(f"Initial log(E_Q[P(X)]): {initial_log_exp:.8f}")
print(f"Initial E_Q[P(X)]:      {np.exp(initial_log_exp):.8f}")
print(f"  (computed in {time.perf_counter() - t0:.3f}s)")

for step in range(1, num_steps + 1):
    t0 = time.perf_counter()
    log_exp_val, _, grads2 = log_exp_query_and_grad(circuit1, circuit2)
    grad_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    apply_gradients_to_circuit2(
        circuit2, grads2.sum_grads, grads2.cat_grads, learning_rate
    )
    apply_time = time.perf_counter() - t0

    print(
        f"  step {step:4d}: log(E) = {log_exp_val:.8f}  "
        f"E = {np.exp(log_exp_val):.8f}  "
        f"(grad {grad_time:.3f}s, apply {apply_time:.3f}s)"
    )

t0 = time.perf_counter()
final_log_exp = log_exp_query(circuit1, circuit2)
final_time = time.perf_counter() - t0

print(f"\nFinal log(E_Q[P(X)]):  {final_log_exp:.8f}")
print(f"Final E_Q[P(X)]:       {np.exp(final_log_exp):.8f}")
print(f"Log improvement:       {final_log_exp - initial_log_exp:+.8f}")
print(f"Final eval time:       {final_time:.3f}s")

circuit2.save("test_circuit_2_exp_minimized.json")
