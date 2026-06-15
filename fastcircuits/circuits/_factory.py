"""Internal node factory with monotonic id allocation."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import numpy as np

from fastcircuits.nodes import CategoricalInputNode, ProductNode, SumNode

_ArrayLike = Union[Sequence[float], np.ndarray]


def _as_float_list(values: _ArrayLike) -> List[float]:
    if isinstance(values, np.ndarray):
        return values.astype(float).tolist()
    return [float(v) for v in values]


class _NodeFactory:
    """Monotonic id allocator; one instance per build()."""

    def __init__(self) -> None:
        self._next_id = 0

    def _alloc_id(self) -> int:
        node_id = self._next_id
        self._next_id += 1
        return node_id

    def categorical(self, scope_var: int, probabilities: _ArrayLike) -> CategoricalInputNode:
        return CategoricalInputNode(
            id=self._alloc_id(),
            scope_var=scope_var,
            probabilities=_as_float_list(probabilities),
        )

    def product(self, children: Iterable) -> ProductNode:
        return ProductNode(id=self._alloc_id(), children=list(children))

    def sum(self, children: Iterable, parameters: _ArrayLike) -> SumNode:
        return SumNode(
            id=self._alloc_id(),
            children=list(children),
            parameters=_as_float_list(parameters),
        )
