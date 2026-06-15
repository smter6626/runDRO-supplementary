"""Maximize the GCW cross-term between two random PCs via differentiable subgradients.

Mirrors ``test_gcw.py``: builds two region-graph embedding circuits, keeps
``circuit1`` fixed, and runs projected gradient ascent on all parameters of
``circuit2`` using ``gcw_crossterm_and_grad``.

Also prints a timing breakdown: Cython GCW solver vs Python-side gradient
application (traversal, NumPy projection, reading/writing parameter lists).
"""

import time
from dataclasses import dataclass

import gurobipy as gp
import numpy as np

from fastcircuits import (
    CategoricalInputNode,
    ProductNode,
    RandomRegionGraph,
    RegionEmbeddingBuilder,
    SumNode,
)
from fastcircuits.gcw import gcw_crossterm, gcw_crossterm_and_grad

# --- Hyperparameters (same style as test_gcw.py) ---
num_vars_1 = 20
num_categories_1 = 10
block_size_1 = 2
num_vars_2 = 5
num_categories_2 = 10
block_size_2 = 2

num_steps = 50
learning_rate = 0.1
prob_floor = 1e-8
warmup_steps = 2


@dataclass
class TimingStats:
    """Accumulated wall times in seconds."""

    cython_gcw_and_grad: float = 0.0
    cython_gcw_value: float = 0.0
    python_apply_total: float = 0.0
    python_apply_traverse: float = 0.0
    python_apply_project: float = 0.0
    python_apply_read_lists: float = 0.0
    cython_apply_write: float = 0.0

    def add(self, other: "TimingStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def report(self, *, label: str, num_steps: int) -> None:
        total = (
            self.cython_gcw_and_grad
            + self.cython_gcw_value
            + self.python_apply_total
        )
        if total <= 0:
            print(f"{label}: no timed work")
            return

        cython_total = self.cython_gcw_and_grad + self.cython_gcw_value
        python_total = self.python_apply_total

        print(f"\n=== Timing: {label} ({num_steps} optimization steps) ===")
        print(f"  Total wall time:           {total:8.3f}s  (100.0%)")
        print()
        print(f"  Cython GCW (solver):       {cython_total:8.3f}s  ({100 * cython_total / total:5.1f}%)")
        print(f"    gcw_crossterm_and_grad:  {self.cython_gcw_and_grad:8.3f}s  ({100 * self.cython_gcw_and_grad / total:5.1f}%)")
        print(f"    gcw_crossterm (value):   {self.cython_gcw_value:8.3f}s  ({100 * self.cython_gcw_value / total:5.1f}%)")
        print()
        print(f"  Python apply (optimizer):  {python_total:8.3f}s  ({100 * python_total / total:5.1f}%)")
        print(f"    DAG traversal/dispatch:  {self.python_apply_traverse:8.3f}s  ({100 * self.python_apply_traverse / total:5.1f}%)")
        print(f"    NumPy project/renorm:    {self.python_apply_project:8.3f}s  ({100 * self.python_apply_project / total:5.1f}%)")
        print(f"    read *_list() (Py objs): {self.python_apply_read_lists:8.3f}s  ({100 * self.python_apply_read_lists / total:5.1f}%)")
        print(f"    set_*_list() (Cython):   {self.cython_apply_write:8.3f}s  ({100 * self.cython_apply_write / total:5.1f}%)")

        if num_steps > 0:
            print()
            print(f"  Per-step averages:")
            print(f"    gcw_crossterm_and_grad:  {self.cython_gcw_and_grad / num_steps * 1000:8.2f} ms")
            print(f"    Python apply (total):    {self.python_apply_total / num_steps * 1000:8.2f} ms")


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
    """Gradient-ascent step on the probability simplex."""
    p = np.asarray(params, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    g = g - g.mean()
    p = p + lr * g
    p = np.clip(p, prob_floor, None)
    return (p / p.sum()).tolist()


def apply_gradients_to_circuit2(circuit2, grads, lr, stats: TimingStats | None = None):
    """Update every learnable parameter node in circuit2 in place."""
    t0 = time.perf_counter()
    for node in iter_circuit_nodes(circuit2.root):
        if stats is not None:
            stats.python_apply_traverse += time.perf_counter() - t0
            t0 = time.perf_counter()

        nid = int(node.id)
        if isinstance(node, SumNode) and nid in grads.sum_grads:
            if stats is not None:
                params = node.parameters_list()
                stats.python_apply_read_lists += time.perf_counter() - t0
                t0 = time.perf_counter()
            else:
                params = node.parameters_list()

            if stats is not None:
                new_params = project_simplex_tangent_step(
                    params, grads.sum_grads[nid], lr
                )
                stats.python_apply_project += time.perf_counter() - t0
                t0 = time.perf_counter()
                node.set_parameters_list(new_params)
                stats.cython_apply_write += time.perf_counter() - t0
                t0 = time.perf_counter()
            else:
                new_params = project_simplex_tangent_step(
                    params, grads.sum_grads[nid], lr
                )
                node.set_parameters_list(new_params)

        elif isinstance(node, CategoricalInputNode) and nid in grads.cat_grads:
            if stats is not None:
                probs = node.probabilities_list()
                stats.python_apply_read_lists += time.perf_counter() - t0
                t0 = time.perf_counter()
            else:
                probs = node.probabilities_list()

            if stats is not None:
                new_probs = project_simplex_tangent_step(
                    probs, grads.cat_grads[nid], lr
                )
                stats.python_apply_project += time.perf_counter() - t0
                t0 = time.perf_counter()
                node.set_probabilities_list(new_probs)
                stats.cython_apply_write += time.perf_counter() - t0
                t0 = time.perf_counter()
            else:
                new_probs = project_simplex_tangent_step(
                    probs, grads.cat_grads[nid], lr
                )
                node.set_probabilities_list(new_probs)

    if stats is not None:
        stats.python_apply_traverse += time.perf_counter() - t0
        stats.python_apply_total = (
            stats.python_apply_traverse
            + stats.python_apply_project
            + stats.python_apply_read_lists
            + stats.cython_apply_write
        )


env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

region_graph1 = RandomRegionGraph(
    frozenset(range(num_vars_1)),
    partitions_per_region=2,
    sub_regions_per_partition=2,
).generate(frozenset(range(num_vars_1)))

circuit1 = RegionEmbeddingBuilder(
    region_graph1,
    num_categories=num_categories_1,
    block_size=block_size_1,
    sum_concentration=1.0,
    input_distribution="categorical",
    alpha=1.0,
    scope_offset=0,
).build()

region_graph2 = RandomRegionGraph(
    frozenset(range(num_vars_2)),
    partitions_per_region=2,
    sub_regions_per_partition=2,
).generate(frozenset(range(num_vars_2)))

circuit2 = RegionEmbeddingBuilder(
    region_graph2,
    num_categories=num_categories_2,
    block_size=block_size_2,
    sum_concentration=1.0,
    input_distribution="categorical",
    alpha=1.0,
    scope_offset=num_vars_1,
).build()

scale_1 = num_vars_1 * num_categories_1
scale_2 = num_vars_2 * num_categories_2
gcw_kw = dict(
    metric_p=1.0,
    scale_factor_1=scale_1,
    scale_factor_2=scale_2,
    gurobi_env=env,
)

t0 = time.perf_counter()
initial = gcw_crossterm(circuit1, circuit2, **gcw_kw)
setup_gcw = time.perf_counter() - t0
print(f"Initial GCW cross-term: {initial}")

# Warmup (not counted in optimization timing)
for _ in range(warmup_steps):
    _, grads = gcw_crossterm_and_grad(circuit1, circuit2, **gcw_kw)
    apply_gradients_to_circuit2(circuit2, grads, learning_rate)

loop_stats = TimingStats()
for step in range(1, num_steps + 1):
    t0 = time.perf_counter()
    cross, grads = gcw_crossterm_and_grad(circuit1, circuit2, **gcw_kw)
    loop_stats.cython_gcw_and_grad += time.perf_counter() - t0

    apply_gradients_to_circuit2(circuit2, grads, learning_rate, stats=loop_stats)

    if step == 1 or step % 50 == 0 or step == num_steps:
        print(f"  step {step:4d}: GCW cross-term = {cross:.8f}")

t0 = time.perf_counter()
final = gcw_crossterm(circuit1, circuit2, **gcw_kw)
final_gcw = time.perf_counter() - t0

print(f"\nFinal GCW cross-term:   {final}")
print(f"Improvement:            {final - initial:+.8f}")

loop_stats.report(label="optimization loop", num_steps=num_steps)

print(f"\n=== One-off (outside loop) ===")
print(f"  Initial gcw_crossterm:     {setup_gcw:8.3f}s")
print(f"  Final gcw_crossterm:       {final_gcw:8.3f}s")
print(
    f"\n  Cython vs Python (loop only): "
    f"{100 * loop_stats.cython_gcw_and_grad / (loop_stats.cython_gcw_and_grad + loop_stats.python_apply_total):.1f}% "
    f"Cython solver, "
    f"{100 * loop_stats.python_apply_total / (loop_stats.cython_gcw_and_grad + loop_stats.python_apply_total):.1f}% "
    f"Python apply"
)

env.dispose()
