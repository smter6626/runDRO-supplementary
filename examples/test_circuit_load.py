import gurobipy as gp

from fastcircuits import RandomRegionGraph, RegionEmbeddingBuilder
from fastcircuits.gcw import gcw_crossterm
from fastcircuits import Circuit, CircuitSerializer, load_learned_pc

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

# Load HCLT PC
circuit = Circuit.load("optimal-transport/learned_pcs/hclt/mnist/4/hclt_mnist_blocksize4_seed0.json")

# compute a likelihood for a random assignment
assignment = {v: 123 for v in circuit.root.scope_as_list()}
print(f"Assignment: {assignment}")
ll = circuit.log_likelihood(assignment)
print(f"Likelihood: {ll}")