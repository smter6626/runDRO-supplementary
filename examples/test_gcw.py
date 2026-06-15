import gurobipy as gp

from fastcircuits import RandomRegionGraph, RegionEmbeddingBuilder
from fastcircuits.gcw import gcw_crossterm

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

num_vars_1 = 5
num_categories_1 = 10
block_size_1 = 2
num_vars_2 = 3
num_categories_2 = 10
block_size_2 = 2

region_graph1 = RandomRegionGraph(
    frozenset(range(num_vars_1)),
    partitions_per_region=2,
    sub_regions_per_partition=2,
).generate(frozenset(range(num_vars_1)))

circuit1 = RegionEmbeddingBuilder(
    region_graph1, num_categories=num_categories_1, block_size=block_size_1,
    sum_concentration=1.0, input_distribution="categorical",
    alpha=1.0, scope_offset=0,
).build()

region_graph2 = RandomRegionGraph(
    frozenset(range(num_vars_2)),
    partitions_per_region=2,
    sub_regions_per_partition=2,
).generate(frozenset(range(num_vars_2)))

circuit2 = RegionEmbeddingBuilder(
    region_graph2, num_categories=num_categories_2, block_size=block_size_2,
    sum_concentration=1.0, input_distribution="categorical",
    alpha=1.0, scope_offset=num_vars_1,
).build()

# Compute GCW cross-term
gcw_cross_term = gcw_crossterm(circuit1, circuit2, metric_p=1.0, scale_factor_1=num_vars_1 * num_categories_1, scale_factor_2=num_vars_2 * num_categories_2, gurobi_env=env)
print(f"GCW cross-term: {gcw_cross_term}")
env.dispose()