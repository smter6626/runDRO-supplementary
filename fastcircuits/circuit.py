from __future__ import annotations

from pathlib import Path
from typing import Union

from fastcircuits.nodes import CircuitNode, likelihood, log_likelihood, sample


class Circuit:
    """Wrapper around a probabilistic circuit root node."""

    def __init__(self, root: CircuitNode):
        self.root = root
        if not root.scope_as_list():
            root.propagate_scope()

    def likelihood(self, assignment: dict[int, int]) -> float:
        """Compute the likelihood of a full evidence assignment at the root."""
        return likelihood(self.root, assignment)

    def log_likelihood(self, assignment: dict[int, int]) -> float:
        """Compute the log-likelihood (log-space, logsumexp at sums)."""
        return log_likelihood(self.root, assignment)

    def mean_log_likelihood_and_grad(self, dataset):
        """Mean LL over a dataset (list of {var: value} dicts) and its gradient.

        Returns ``(mean_ll, grads)`` where ``grads`` is a
        :class:`LogLikelihoodGradients` bundle (``sum_grads`` / ``cat_grads``
        keyed by ``node.id``) of the mean log-likelihood w.r.t. the circuit's
        linear parameters. Use with simplex-tangent ascent for MLE learning.
        """
        from fastcircuits.nodes import mean_log_likelihood_and_grad

        return mean_log_likelihood_and_grad(self.root, dataset)

    def sample(
        self,
        n_samples: int,
        seed: int | None = None,
    ) -> list[dict[int, int]]:
        """Draw n_samples ancestral samples; each is a full-scope {var: value} dict."""
        return sample(self.root, n_samples, seed)

    def save(
        self,
        path: Union[str, Path],
        *,
        indent: int = 2,
        encoding: str = "utf-8",
    ) -> None:
        """Serialize this circuit to a UTF-8 JSON file (gcw-circuit-v1)."""
        from fastcircuits.circuits.circuit_serializer import CircuitSerializer

        CircuitSerializer.save(self.root, path, indent=indent, encoding=encoding)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        encoding: str = "utf-8",
    ) -> Circuit:
        """Load a circuit from a gcw-circuit-v1 JSON file."""
        from fastcircuits.circuits.circuit_serializer import CircuitSerializer

        root = CircuitSerializer.load(path, encoding=encoding)
        return cls(root)
