# fastcircuits

Fast probabilistic circuits in Cython: typed `cdef` nodes with C++ scope sets and pointer-based child traversal.

## Install

```bash
pip install -e .
pip install -e ".[dev]"   # optional: pytest, mypy
pip install -e ".[gcw,dev]"  # GCW cross-term query: gurobipy
```

Requires a C++17 compiler (Visual Studio Build Tools on Windows, GCC/Clang elsewhere) and Cython 3+.

## Quick example

```python
from fastcircuits import CategoricalInputNode, Circuit, ProductNode, SumNode

x0 = CategoricalInputNode(id=0, scope_var=3, probabilities=[0.7, 0.3])
x1 = CategoricalInputNode(id=1, scope_var=17, probabilities=[0.5, 0.5])
prod = ProductNode(id=2, children=[x0, x1])
root = SumNode(id=3, children=[prod], parameters=[1.0])
circuit = Circuit(root)
assert set(root.scope_as_list()) == {3, 17}
assert circuit.likelihood({3: 0, 17: 1}) == 0.35
```

## Nodes

| Class | Role |
|-------|------|
| `CategoricalInputNode` | Leaf with a 1D categorical PMF; scope is exactly one variable index |
| `ProductNode` | Product of children; scope = union of child scopes |
| `SumNode` | Weighted sum of children; parameters must be non-negative and sum to 1 |

DAGs are supported: the same child may appear under multiple parents. Each parent keeps Python references so C pointers stay valid.

## Queries

Likelihood evaluation traverses the graph bottom-up with **memoization** (by node id) for DAG-safe reuse.

| Node type | Likelihood | Log-likelihood |
|-----------|------------|----------------|
| `CategoricalInputNode` | `probabilities[assignment[scope_var]]` | `log p` |
| `ProductNode` | product of child likelihoods | sum of child log-likelihoods |
| `SumNode` | weighted sum of child likelihoods | `logsumexp(log w_i + child_i)` |

**Evidence (v1):** full assignments only — every variable in the root scope must be observed.

```python
from fastcircuits import likelihood, log_likelihood

likelihood(root, {3: 0, 17: 1})
log_likelihood(root, {3: 0, 17: 1})  # log-space; stable logsumexp at sums
```

## GCW cross-term query

The optional `fastcircuits.gcw` extension computes the **Gromov–Circuit–Wasserstein cross-term** between two circuits (scalar only; no coupling circuit). Install `[gcw]` and provide a Gurobi environment:

```python
import gurobipy as gp
from fastcircuits import CategoricalInputNode, gcw_crossterm

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.5, 0.5])
leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.6, 0.4])
cross = gcw_crossterm(leaf1, leaf2, gurobi_env=env)
```

| Dependency | Role |
|------------|------|
| **gurobipy** + Gurobi license | Optimal transport at sum×sum nodes |
| **scipy** | `linear_sum_assignment` (Hungarian) at product×product nodes |

Implementation is in Cython (`fastcircuits/gcw.pyx`); Gurobi and SciPy are called only at those internal solver steps.

### Differentiable GCW cross-term

`gcw_crossterm_and_grad` returns the same scalar as `gcw_crossterm` together with subgradients with respect to every `SumNode` parameter vector and `CategoricalInputNode` probability vector in `circuit2`. `circuit1` is treated as a fixed reference.

```python
from fastcircuits import gcw_crossterm_and_grad

value, grads = gcw_crossterm_and_grad(circuit_ref, circuit_learn, gurobi_env=env)
# grads.sum_grads: dict[node_id -> ndarray] for SumNode parameters
# grads.cat_grads: dict[node_id -> ndarray] for CategoricalInputNode probabilities
```

The gradient is an exact subgradient: the forward pass is identical to `gcw_crossterm`, and backward reuses the discrete forward choices (NW plan with row/column-binding modes, sum×sum LP basis with Gurobi duals, product×product Hungarian assignment, argmax index in product×other, and the max-of-two branch at sum×product). At kinks any valid subgradient is returned, so an optimizer should project gradients onto the simplex tangent (e.g. `g -= g.mean()`) before each step.

## Circuit builders

Build random or imported probabilistic circuits without hand-wiring nodes. Builders return a `Circuit` ready for likelihood queries.

### Region-graph embedding PC

Typical workflow for a small embedding circuit over variables offset from a base model (e.g. MNIST 784 + embedding vars):

```python
import numpy as np
from fastcircuits import RandomRegionGraph, RegionEmbeddingBuilder

np.random.seed(0)
num_vars = 5
block_size = 4
region_graph = RandomRegionGraph(
    frozenset(range(num_vars)),
    partitions_per_region=2,
    sub_regions_per_partition=block_size,
).generate(frozenset(range(num_vars)))

embedding = RegionEmbeddingBuilder(
    region_graph,
    num_categories=10,
    block_size=block_size,
    sum_concentration=1.0,
    input_distribution="categorical",
    alpha=1.0,
    scope_offset=784,
).build()

assignment = {v: 0 for v in embedding.root.scope_as_list()}
embedding.likelihood(assignment)
```

### Recursive embedding PC

`EmbeddingBuilder` builds a random PC by recursive partitioning with optional structural reuse:

```python
from fastcircuits import EmbeddingBuilder

circuit = EmbeddingBuilder(
    num_vars=8,
    num_categories=4,
    sum_arity=2,
    prod_arity=2,
    sum_concentration=1.0,
    sum_reuse_probability=0.0,
    prod_reuse_probability=0.0,
    input_distribution="categorical",
    alpha=1.0,
).build()
```

### PyJuice import

Convert a compiled PyJuice circuit (requires `pyjuice` and `torch` installed separately):

```python
from fastcircuits import PyjuiceBuilder

builder = PyjuiceBuilder(block_size=pc.root_ns.num_chs)
circuit = builder.build(pc, scope_offset=0)
```

## Serialization

Circuits are saved as UTF-8 JSON using the **`gcw-circuit-v1`** format (post-order node list, DAG sharing by index). This matches JSON files produced by the older optimal-transport library, so existing learned PCs can be loaded without retraining.

```python
from pathlib import Path
from fastcircuits import Circuit, CircuitSerializer, load_learned_pc

# Save / load
circuit.save("model.json")
restored = Circuit.load("model.json")

# Or use the serializer directly on the root node
CircuitSerializer.save(circuit.root, "model.json")
root = CircuitSerializer.load("model.json")

# Standard learned-PC layout:
#   {base}/{structure}/{dataset}/{block_size}/{structure}_{dataset}_blocksize{N}_seed{S}.json
# Example: learned_pcs/hclt/mnist/4/hclt_mnist_blocksize4_seed0.json
circuit = load_learned_pc("learned_pcs", "hclt", "mnist", block_size=4, seed=0)
```

Typical training pipeline: PyJuice HCLT + EM → `PyjuiceBuilder(block_size).build(pc)` → `circuit.save(path)`.

Gaussian input nodes are not supported in fastcircuits; they are rejected on load.

## Tests

```bash
pytest
pytest -m gurobi   # GCW tests (skip if gurobipy/license unavailable)
pytest -m pyjuice  # PyJuice import tests (skip if pyjuice/torch unavailable)
```

## Future work

Partial evidence / marginals and optional decomposability checks on product nodes.
