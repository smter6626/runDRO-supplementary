"""Minimal MLE demo: gradient ascent on mean log-likelihood."""

import numpy as np

from fastcircuits import CategoricalInputNode, Circuit, ProductNode, SumNode

SEED = 0
N_SAMPLES = 200
N_STEPS = 100
LR = 0.05
PROB_FLOOR = 1e-12


def build_circuit():
    cat_a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.6, 0.4])
    cat_b = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.2, 0.8])
    sum_x0 = SumNode(id=2, children=[cat_a, cat_b], parameters=[0.7, 0.3])

    cat_c = CategoricalInputNode(id=3, scope_var=1, probabilities=[0.3, 0.7])
    cat_d = CategoricalInputNode(id=4, scope_var=1, probabilities=[0.9, 0.1])
    sum_x1 = SumNode(id=5, children=[cat_c, cat_d], parameters=[0.5, 0.5])

    root = ProductNode(id=6, children=[sum_x0, sum_x1])
    root.propagate_scope()
    return Circuit(root)


def random_dataset(n, seed):
    rng = np.random.default_rng(seed)
    return [
        {0: int(rng.integers(0, 2)), 1: int(rng.integers(0, 2))}
        for _ in range(n)
    ]


def iter_nodes(root):
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        if isinstance(node, (SumNode, ProductNode)):
            stack.extend(node.children())


def simplex_ascent_step(params, grad, lr):
    p = np.asarray(params, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64) - np.mean(grad)
    x = np.clip(p + lr * g, PROB_FLOOR, None)
    return (x / x.sum()).tolist()


def apply_gradients(pc, grads, lr):
    for node in iter_nodes(pc.root):
        nid = int(node.id)
        if isinstance(node, SumNode) and nid in grads.sum_grads:
            node.set_parameters_list(
                simplex_ascent_step(node.parameters_list(), grads.sum_grads[nid], lr)
            )
        elif isinstance(node, CategoricalInputNode) and nid in grads.cat_grads:
            node.set_probabilities_list(
                simplex_ascent_step(node.probabilities_list(), grads.cat_grads[nid], lr)
            )


def main():
    pc = build_circuit()
    dataset = random_dataset(N_SAMPLES, SEED)

    initial_ll, _ = pc.mean_log_likelihood_and_grad(dataset)
    print(f"Initial mean log-likelihood: {initial_ll:.6f}")

    for step in range(1, N_STEPS + 1):
        mean_ll, grads = pc.mean_log_likelihood_and_grad(dataset)
        apply_gradients(pc, grads, LR)
        if step in (1, N_STEPS // 2, N_STEPS):
            print(f"  step {step:3d}: mean log-likelihood = {mean_ll:.6f}")

    final_ll, _ = pc.mean_log_likelihood_and_grad(dataset)
    print(f"Final mean log-likelihood:   {final_ll:.6f}")
    print(f"Improvement:               {final_ll - initial_ll:+.6f}")

    if final_ll <= initial_ll:
        raise SystemExit("MLE learning did not improve mean log-likelihood")

    print("OK: gradient ascent increased mean log-likelihood")


if __name__ == "__main__":
    main()
