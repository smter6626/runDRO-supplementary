"""Minimize CW distance between a fixed circuit1 and learnable circuit2.

With circuit1 fixed, each step evaluates ``cw_distance_and_grad`` and applies
projected simplex descent to every sum / categorical parameter in circuit2.
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
)

CIRCUIT_PATH_1 = "test_circuit_1.json"
CIRCUIT_PATH_2 = "test_circuit_2.json"
METRIC_P = 1.0
SCALE_FACTOR = 1

num_steps = 50
learning_rate = 1e-5
prob_floor = 1e-20

cw_kw = dict(metric_p=METRIC_P, scale_factor=SCALE_FACTOR)


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
    # 1. Take the unconstrained step
    x = np.asarray(params, dtype=np.float64) - lr * np.asarray(grad, dtype=np.float64)
    
    # 2. Sort x in descending order to find the threshold
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    
    # 3. Find the coordinates that will remain positive after shifting
    rho = np.nonzero(u * np.arange(1, len(x) + 1) > (cssv - 1))[0][-1]
    
    # 4. Compute the Lagrange multiplier theta
    theta = (cssv[rho] - 1) / (rho + 1)
    
    # 5. Project and return
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


env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()
cw_kw["gurobi_env"] = env

circuit1 = Circuit.load(CIRCUIT_PATH_1)
circuit2 = Circuit.load(CIRCUIT_PATH_2)

t0 = time.perf_counter()
initial_distance = cw_distance(circuit1, circuit2, **cw_kw)
print(f"Initial CW W_{METRIC_P}^{METRIC_P}: {initial_distance:.8f}")
print(f"  (computed in {time.perf_counter() - t0:.3f}s)")

for step in range(1, num_steps + 1):
    t0 = time.perf_counter()
    distance, grads = cw_distance_and_grad(circuit1, circuit2, **cw_kw)
    grad_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    apply_gradients_to_circuit2(
        circuit2, grads.sum_grads, grads.cat_grads, learning_rate
    )
    apply_time = time.perf_counter() - t0

    wp = distance ** (1.0 / METRIC_P)
    print(
        f"  step {step:4d}: W_{METRIC_P}^{METRIC_P} = {distance:.8f}  "
        f"W_{METRIC_P} = {wp:.8f}  "
        f"(grad {grad_time:.3f}s, apply {apply_time:.3f}s)"
    )

t0 = time.perf_counter()
final_distance = cw_distance(circuit1, circuit2, **cw_kw)
final_time = time.perf_counter() - t0

final_wp = final_distance ** (1.0 / METRIC_P)
print(f"\nFinal CW W_{METRIC_P}^{METRIC_P}:   {final_distance:.8f}")
print(f"Final CW W_{METRIC_P}:            {final_wp:.8f}")
print(f"Improvement (W_p^p):              {final_distance - initial_distance:+.8f}")
print(f"Final eval time:                  {final_time:.3f}s")

env.dispose()

circuit2.save("test_circuit_2_minimized.json")
