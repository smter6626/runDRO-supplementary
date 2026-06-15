from typing import Mapping, Sequence

from fastcircuits.nodes import CircuitNode, LogLikelihoodGradients

class Circuit:
    root: CircuitNode

    def __init__(self, root: CircuitNode) -> None: ...
    def likelihood(self, assignment: dict[int, int]) -> float: ...
    def log_likelihood(self, assignment: dict[int, int]) -> float: ...
    def mean_log_likelihood_and_grad(
        self,
        dataset: Sequence[Mapping[int, int]],
    ) -> tuple[float, LogLikelihoodGradients]: ...
    def sample(
        self,
        n_samples: int,
        seed: int | None = None,
    ) -> list[dict[int, int]]: ...
