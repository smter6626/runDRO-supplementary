import gurobipy as gp

from fastcircuits import RandomRegionGraph, RegionEmbeddingBuilder
from fastcircuits.gcw import gcw_crossterm
from fastcircuits import Circuit, CircuitSerializer, load_learned_pc

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

# Load HCLT PC
circuit1 = Circuit.load("optimal-transport/learned_pcs/hclt/mnist/4/hclt_mnist_blocksize4_seed0.json")

num_vars_2 = 20
num_categories_2 = 10
block_size_2 = 2

region_graph2 = RandomRegionGraph(
    frozenset(range(num_vars_2)),
    partitions_per_region=2,
    sub_regions_per_partition=2,
).generate(frozenset(range(num_vars_2)))

circuit2 = RegionEmbeddingBuilder(
    region_graph2, num_categories=num_categories_2, block_size=block_size_2,
    sum_concentration=1.0, input_distribution="categorical",
    alpha=1.0, scope_offset=784,
).build()

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

import time
start = time.time()
# Compute GCW cross-term
gcw_cross_term = gcw_crossterm(circuit1, circuit2, metric_p=1.0, scale_factor_1=784 * 256, scale_factor_2=num_vars_2 * num_categories_2, gurobi_env=env)
print(f"GCW cross-term: {gcw_cross_term}")
print(f"Time: {time.time() - start:.2f}s")
env.dispose()