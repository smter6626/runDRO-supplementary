"""Minimize the full GCW distance between a fixed circuit1 and learnable circuit2.

GCW_distance = ExpectedSquaredDistance(c1) + ExpectedSquaredDistance(c2)
               - 2 * GCWCrossterm(c1, c2)

With c1 fixed, we minimize ESD(c2) - 2*crossterm over circuit2's parameters by
accumulating gradients from separate Cython tapes (ESD and crossterm) and
applying projected simplex descent.
"""

import time
from dataclasses import dataclass

import gurobipy as gp
import numpy as np

from fastcircuits import (
    CategoricalInputNode,
    Circuit,
    ProductNode,
    RandomRegionGraph,
    RegionEmbeddingBuilder,
    SumNode,
    expected_squared_distance,
    expected_squared_distance_and_grad,
    gcw_crossterm,
    gcw_crossterm_and_grad,
)

# --- Hyperparameters (same style as gcw_maximization.py) ---

embedding_vars = 20
embedding_categories = 10
embedding_block_size = 4
embedding_partitions_per_region = 4
embedding_sub_regions_per_partition = 4
embedding_sum_concentration = 1.0
embedding_input_distribution = "categorical"
embedding_alpha = 1.0
embedding_scope_offset = 784

num_steps = 50
learning_rate = 100
prob_floor = 1e-8


@dataclass
class TimingStats:
    """Accumulated wall times in seconds."""

    cython_cross_and_grad: float = 0.0
    cython_esd_and_grad: float = 0.0
    cython_cross_value: float = 0.0
    cython_esd_value: float = 0.0
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
            self.cython_cross_and_grad
            + self.cython_esd_and_grad
            + self.cython_cross_value
            + self.cython_esd_value
            + self.python_apply_total
        )
        if total <= 0:
            print(f"{label}: no timed work")
            return

        cython_total = (
            self.cython_cross_and_grad
            + self.cython_esd_and_grad
            + self.cython_cross_value
            + self.cython_esd_value
        )
        python_total = self.python_apply_total

        print(f"\n=== Timing: {label} ({num_steps} optimization steps) ===")
        print(f"  Total wall time:           {total:8.3f}s  (100.0%)")
        print()
        print(
            f"  Cython (solver):           {cython_total:8.3f}s  "
            f"({100 * cython_total / total:5.1f}%)"
        )
        print(
            f"    gcw_crossterm_and_grad:  {self.cython_cross_and_grad:8.3f}s  "
            f"({100 * self.cython_cross_and_grad / total:5.1f}%)"
        )
        print(
            f"    esd_and_grad:            {self.cython_esd_and_grad:8.3f}s  "
            f"({100 * self.cython_esd_and_grad / total:5.1f}%)"
        )
        print(
            f"    gcw_crossterm (value):   {self.cython_cross_value:8.3f}s  "
            f"({100 * self.cython_cross_value / total:5.1f}%)"
        )
        print(
            f"    expected_squared_dist:   {self.cython_esd_value:8.3f}s  "
            f"({100 * self.cython_esd_value / total:5.1f}%)"
        )
        print()
        print(
            f"  Python apply (optimizer):  {python_total:8.3f}s  "
            f"({100 * python_total / total:5.1f}%)"
        )
        print(
            f"    DAG traversal/dispatch:  {self.python_apply_traverse:8.3f}s  "
            f"({100 * self.python_apply_traverse / total:5.1f}%)"
        )
        print(
            f"    NumPy project/renorm:    {self.python_apply_project:8.3f}s  "
            f"({100 * self.python_apply_project / total:5.1f}%)"
        )
        print(
            f"    read *_list() (Py objs): {self.python_apply_read_lists:8.3f}s  "
            f"({100 * self.python_apply_read_lists / total:5.1f}%)"
        )
        print(
            f"    set_*_list() (Cython):   {self.cython_apply_write:8.3f}s  "
            f"({100 * self.cython_apply_write / total:5.1f}%)"
        )

        if num_steps > 0:
            print()
            print("  Per-step averages:")
            print(
                f"    gcw_crossterm_and_grad:  "
                f"{self.cython_cross_and_grad / num_steps * 1000:8.2f} ms"
            )
            print(
                f"    esd_and_grad:            "
                f"{self.cython_esd_and_grad / num_steps * 1000:8.2f} ms"
            )
            print(
                f"    Python apply (total):    "
                f"{self.python_apply_total / num_steps * 1000:8.2f} ms"
            )


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


def project_simplex_tangent_step(params, grad, lr, *, descent=False):
    """Gradient step on the probability simplex."""
    p = np.asarray(params, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    g = g - g.mean()
    sign = -1.0 if descent else 1.0
    p = p + sign * lr * g
    p = np.clip(p, prob_floor, None)
    return (p / p.sum()).tolist()


def combine_grads(esd_grads, cross_grads):
    """Accumulate d/d(circuit2) of ESD(c2) - 2*crossterm."""
    sum_g: dict[int, np.ndarray] = {}
    cat_g: dict[int, np.ndarray] = {}

    for d_out, e_dict, c_dict in (
        (sum_g, esd_grads.sum_grads, cross_grads.sum_grads),
        (cat_g, esd_grads.cat_grads, cross_grads.cat_grads),
    ):
        for nid in set(e_dict) | set(c_dict):
            e_arr = e_dict.get(nid)
            c_arr = c_dict.get(nid)
            if e_arr is not None and c_arr is not None:
                d_out[nid] = np.asarray(e_arr, dtype=np.float64) - 2.0 * np.asarray(
                    c_arr, dtype=np.float64
                )
            elif e_arr is not None:
                d_out[nid] = np.asarray(e_arr, dtype=np.float64).copy()
            else:
                d_out[nid] = -2.0 * np.asarray(c_arr, dtype=np.float64)

    return sum_g, cat_g


def apply_gradients_to_circuit2(
    circuit2,
    sum_grads,
    cat_grads,
    lr,
    stats: TimingStats | None = None,
):
    """Update every learnable parameter node in circuit2 in place (descent)."""
    t0 = time.perf_counter()
    for node in iter_circuit_nodes(circuit2.root):
        if stats is not None:
            stats.python_apply_traverse += time.perf_counter() - t0
            t0 = time.perf_counter()

        nid = int(node.id)
        if isinstance(node, SumNode) and nid in sum_grads:
            if stats is not None:
                params = node.parameters_list()
                stats.python_apply_read_lists += time.perf_counter() - t0
                t0 = time.perf_counter()
            else:
                params = node.parameters_list()

            if stats is not None:
                new_params = project_simplex_tangent_step(
                    params, sum_grads[nid], lr, descent=True
                )
                stats.python_apply_project += time.perf_counter() - t0
                t0 = time.perf_counter()
                node.set_parameters_list(new_params)
                stats.cython_apply_write += time.perf_counter() - t0
                t0 = time.perf_counter()
            else:
                new_params = project_simplex_tangent_step(
                    params, sum_grads[nid], lr, descent=True
                )
                node.set_parameters_list(new_params)

        elif isinstance(node, CategoricalInputNode) and nid in cat_grads:
            if stats is not None:
                probs = node.probabilities_list()
                stats.python_apply_read_lists += time.perf_counter() - t0
                t0 = time.perf_counter()
            else:
                probs = node.probabilities_list()

            if stats is not None:
                new_probs = project_simplex_tangent_step(
                    probs, cat_grads[nid], lr, descent=True
                )
                stats.python_apply_project += time.perf_counter() - t0
                t0 = time.perf_counter()
                node.set_probabilities_list(new_probs)
                stats.cython_apply_write += time.perf_counter() - t0
                t0 = time.perf_counter()
            else:
                new_probs = project_simplex_tangent_step(
                    probs, cat_grads[nid], lr, descent=True
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


def gcw_distance(esd_c1, esd_c2, cross):
    return esd_c1 + esd_c2 - 2.0 * cross


env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

circuit1 = Circuit.load(
    "test_circuit_1.json"
)

embedding_region_graph = RandomRegionGraph(
    frozenset(range(embedding_vars)),
    partitions_per_region=embedding_partitions_per_region,
    sub_regions_per_partition=embedding_sub_regions_per_partition,
).generate(frozenset(range(embedding_vars)))

circuit2 = RegionEmbeddingBuilder(
    embedding_region_graph,
    num_categories=embedding_categories,
    block_size=embedding_block_size,
    sum_concentration=1.0,
    input_distribution=embedding_input_distribution,
    alpha=embedding_alpha,
    scope_offset=embedding_scope_offset,
).build()

scale_1 = 784 * 256
scale_2 = embedding_vars * embedding_categories
metric_p = 1.0
gcw_kw = dict(
    metric_p=metric_p,
    scale_factor_1=scale_1,
    scale_factor_2=scale_2,
    gurobi_env=env,
)
esd_kw = dict(metric_p=metric_p, scale_factor=scale_2)

t0 = time.perf_counter()
esd_c1 = expected_squared_distance(circuit1, metric_p=metric_p, scale_factor=scale_1)
setup_esd_c1 = time.perf_counter() - t0

t0 = time.perf_counter()
initial_cross = gcw_crossterm(circuit1, circuit2, **gcw_kw)
setup_cross = time.perf_counter() - t0

t0 = time.perf_counter()
initial_esd2 = expected_squared_distance(circuit2, **esd_kw)
setup_esd2 = time.perf_counter() - t0

initial_distance = gcw_distance(esd_c1, initial_esd2, initial_cross)
print(f"Initial GCW distance:     {initial_distance:.8f}")
print(f"  ESD(circuit1):          {esd_c1:.8f}")
print(f"  ESD(circuit2):          {initial_esd2:.8f}")
print(f"  GCW cross-term:         {initial_cross:.8f}")

loop_stats = TimingStats()
for step in range(1, num_steps + 1):
    t0 = time.perf_counter()
    cross, cross_grads = gcw_crossterm_and_grad(circuit1, circuit2, **gcw_kw)
    loop_stats.cython_cross_and_grad += time.perf_counter() - t0

    t0 = time.perf_counter()
    esd2, esd_grads = expected_squared_distance_and_grad(circuit2, **esd_kw)
    loop_stats.cython_esd_and_grad += time.perf_counter() - t0

    sum_grads, cat_grads = combine_grads(esd_grads, cross_grads)
    apply_gradients_to_circuit2(
        circuit2, sum_grads, cat_grads, learning_rate, stats=loop_stats
    )

    total_distance = gcw_distance(esd_c1, esd2, cross)
    print(
        f"  step {step:4d}: GCW distance = {total_distance:.8f}  "
        f"(ESD2={esd2:.6f}, cross={cross:.6f})"
    )

t0 = time.perf_counter()
final_cross = gcw_crossterm(circuit1, circuit2, **gcw_kw)
final_cross_time = time.perf_counter() - t0

t0 = time.perf_counter()
final_esd2 = expected_squared_distance(circuit2, **esd_kw)
final_esd2_time = time.perf_counter() - t0

final_distance = gcw_distance(esd_c1, final_esd2, final_cross)

loop_stats.cython_cross_value = final_cross_time
loop_stats.cython_esd_value = final_esd2_time
loop_stats.report(label="optimization loop", num_steps=num_steps)

print(f"\n=== One-off (outside loop) ===")
print(f"  Initial ESD(circuit1):     {setup_esd_c1:8.3f}s")
print(f"  Initial gcw_crossterm:     {setup_cross:8.3f}s")
print(f"  Initial ESD(circuit2):     {setup_esd2:8.3f}s")
print(f"  Final gcw_crossterm:       {final_cross_time:8.3f}s")
print(f"  Final ESD(circuit2):       {final_esd2_time:8.3f}s")

cython_loop = loop_stats.cython_cross_and_grad + loop_stats.cython_esd_and_grad
print(
    f"\n  Cython vs Python (loop only): "
    f"{100 * cython_loop / (cython_loop + loop_stats.python_apply_total):.1f}% "
    f"Cython solver, "
    f"{100 * loop_stats.python_apply_total / (cython_loop + loop_stats.python_apply_total):.1f}% "
    f"Python apply"
)

print(f"\nFinal GCW distance:       {final_distance:.8f}")
print(f"Improvement:              {final_distance - initial_distance:+.8f}")

env.dispose()

circuit2.save("learned_embedding_min_seed0.json")
