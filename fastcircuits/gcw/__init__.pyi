from fastcircuits.nodes import CircuitNode

class GCWGradients:
    value: float
    sum_grads: dict[int, object]
    cat_grads: dict[int, object]

def gcw_crossterm(
    circuit1: CircuitNode | object,
    circuit2: CircuitNode | object,
    metric_p: float = 1.0,
    scale_factor_1: float = 1.0,
    scale_factor_2: float = 1.0,
    gurobi_env: object = ...,
) -> float: ...


def gcw_crossterm_and_grad(
    circuit1: CircuitNode | object,
    circuit2: CircuitNode | object,
    metric_p: float = 1.0,
    scale_factor_1: float = 1.0,
    scale_factor_2: float = 1.0,
    gurobi_env: object = ...,
) -> tuple[float, GCWGradients]: ...


def expected_squared_distance(
    circuit: CircuitNode | object,
    metric_p: float = 1.0,
    scale_factor: float = 1.0,
) -> float: ...


def expected_squared_distance_and_grad(
    circuit: CircuitNode | object,
    metric_p: float = 1.0,
    scale_factor: float = 1.0,
) -> tuple[float, GCWGradients]: ...


def cw_distance(
    circuit1: CircuitNode | object,
    circuit2: CircuitNode | object,
    metric_p: float = 1.0,
    scale_factor: float = 1.0,
    gurobi_env: object = ...,
) -> float: ...


def cw_distance_and_grad(
    circuit1: CircuitNode | object,
    circuit2: CircuitNode | object,
    metric_p: float = 1.0,
    scale_factor: float = 1.0,
    gurobi_env: object = ...,
) -> tuple[float, GCWGradients]: ...


def exp_query(
    circuit1: CircuitNode | object,
    circuit2: CircuitNode | object,
) -> float: ...


def exp_query_and_grad(
    circuit1: CircuitNode | object,
    circuit2: CircuitNode | object,
) -> tuple[float, GCWGradients, GCWGradients]: ...


def log_exp_query(
    circuit1: CircuitNode | object,
    circuit2: CircuitNode | object,
) -> float: ...


def log_exp_query_and_grad(
    circuit1: CircuitNode | object,
    circuit2: CircuitNode | object,
) -> tuple[float, GCWGradients, GCWGradients]: ...
