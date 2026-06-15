from fastcircuits import RandomRegionGraph, RegionEmbeddingBuilder

region_graph = RandomRegionGraph(
    frozenset(range(5)),
    partitions_per_region=2,
    sub_regions_per_partition=4,
).generate(frozenset(range(5)))

circuit = RegionEmbeddingBuilder(
    region_graph, num_categories=10, block_size=4,
    sum_concentration=1.0, input_distribution="categorical",
    alpha=1.0, scope_offset=0,
).build()

print(circuit.likelihood({0: 0, 1: 1, 2: 2, 3: 3, 4: 4}))
print(circuit.log_likelihood({0: 0, 1: 1, 2: 2, 3: 3, 4: 4}))