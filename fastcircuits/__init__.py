from fastcircuits.circuit import Circuit
from fastcircuits.circuits import (
    CircuitSerializer,
    EmbeddingBuilder,
    Partition,
    PyjuiceBuilder,
    RandomRegionGraph,
    Region,
    RegionEmbeddingBuilder,
    load_learned_pc,
)
from fastcircuits.nodes import (
    CategoricalInputNode,
    CircuitNode,
    Evidence,
    LogLikelihoodGradients,
    ProductNode,
    SumNode,
    likelihood,
    log_likelihood,
    mean_log_likelihood_and_grad,
    sample,
)

try:
    from fastcircuits.gcw import (
        GCWGradients,
        cw_distance,
        cw_distance_and_grad,
        expected_squared_distance,
        expected_squared_distance_and_grad,
        exp_query,
        exp_query_and_grad,
        log_exp_query,
        log_exp_query_and_grad,
        gcw_crossterm,
        gcw_crossterm_and_grad,
    )
except ImportError:
    gcw_crossterm = None  # type: ignore[misc, assignment]
    gcw_crossterm_and_grad = None  # type: ignore[misc, assignment]
    cw_distance = None  # type: ignore[misc, assignment]
    cw_distance_and_grad = None  # type: ignore[misc, assignment]
    expected_squared_distance = None  # type: ignore[misc, assignment]
    expected_squared_distance_and_grad = None  # type: ignore[misc, assignment]
    exp_query = None  # type: ignore[misc, assignment]
    exp_query_and_grad = None  # type: ignore[misc, assignment]
    log_exp_query = None  # type: ignore[misc, assignment]
    log_exp_query_and_grad = None  # type: ignore[misc, assignment]
    GCWGradients = None  # type: ignore[misc, assignment]

__version__ = "0.1.0"
__all__ = [
    "CategoricalInputNode",
    "Circuit",
    "CircuitNode",
    "CircuitSerializer",
    "EmbeddingBuilder",
    "Evidence",
    "LogLikelihoodGradients",
    "Partition",
    "ProductNode",
    "PyjuiceBuilder",
    "RandomRegionGraph",
    "Region",
    "RegionEmbeddingBuilder",
    "SumNode",
    "load_learned_pc",
    "GCWGradients",
    "cw_distance",
    "cw_distance_and_grad",
    "expected_squared_distance",
    "expected_squared_distance_and_grad",
    "exp_query",
    "exp_query_and_grad",
    "log_exp_query",
    "log_exp_query_and_grad",
    "gcw_crossterm",
    "gcw_crossterm_and_grad",
    "likelihood",
    "log_likelihood",
    "mean_log_likelihood_and_grad",
    "sample",
    "__version__",
]
