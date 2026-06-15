from typing import FrozenSet, Iterable, Tuple
import random


def _region_scope_key(region: "Region") -> Tuple[int, ...]:
    return tuple(sorted(region.scope))


def _scope_subset_key(subset: FrozenSet[int]) -> Tuple[int, ...]:
    return tuple(sorted(subset))


class Region:
    def __init__(self, scope):
        self.scope = frozenset(scope)
        self.partitions = []  # List of partitions: (sub_region1, sub_region2, ...)

    def __repr__(self):
        return f"Region(scope={sorted(list(self.scope))})"


class Partition:
    def __init__(self, sub_regions: Iterable["Region"]):
        # Deterministic order across runs (frozenset of Regions uses id-based hashing).
        self.sub_regions: Tuple[Region, ...] = tuple(
            sorted(sub_regions, key=_region_scope_key)
        )

    def __repr__(self):
        return f"Partition(sub_regions={list(self.sub_regions)})"


class RandomRegionGraph:
    def __init__(
        self,
        starting_scope: FrozenSet[int],
        partitions_per_region: int,
        sub_regions_per_partition: int,
    ):
        assert len(starting_scope) > 0, "Starting scope must be non-empty"
        assert partitions_per_region > 0, "Partitions per region must be positive"
        assert sub_regions_per_partition > 0, "Sub regions per partition must be positive"
        self.starting_scope = starting_scope
        self.partitions_per_region = partitions_per_region
        self.sub_regions_per_partition = sub_regions_per_partition

        self.region_cache = {}

    def generate(self, scope: FrozenSet[int]):
        if scope in self.region_cache:
            return self.region_cache[scope]
        # Create the root region
        root = Region(scope)

        # If scope has length 1, we're done
        if len(scope) == 1:
            return root

        # Otherwise, we need to create the partitions
        partitions = []
        for _ in range(self.partitions_per_region):
            sub_region_partitions = self._balanced_random_partition(
                scope, self.sub_regions_per_partition
            )
            sub_regions = [self.generate(partition) for partition in sub_region_partitions]
            partitions.append(Partition(sub_regions))
        root.partitions = partitions
        self.region_cache[scope] = root
        return root

    def _balanced_random_partition(
        self, input_set: FrozenSet[int], k: int
    ) -> Tuple[FrozenSet[int], ...]:
        """
        Randomly partitions a set into k evenly sized subsets,
        omitting any subsets that end up empty. Subsets are returned in sorted
        order by scope so downstream traversal is reproducible for a fixed RNG seed.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer greater than 0.")

        # Convert the set to a list so it can be shuffled
        items = list(input_set)
        random.shuffle(items)

        # Distribute items into k subsets using slicing
        # items[i::k] takes every k-th element starting from index i
        subsets = [frozenset(items[i::k]) for i in range(k)]

        # Filter out empty sets (which happens if k is greater than the number of items)
        non_empty = [s for s in subsets if s]
        return tuple(sorted(non_empty, key=_scope_subset_key))
