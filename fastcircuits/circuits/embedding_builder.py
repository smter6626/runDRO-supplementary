import random
from typing import Dict, Optional

import numpy as np
from scipy.special import gammaln

from fastcircuits.circuit import Circuit

from ._factory import _NodeFactory
from .region_graph import Partition, Region


class RegionEmbeddingBuilder:
    def __init__(
        self,
        region_graph: Region,
        num_categories: int,
        block_size: int,
        sum_concentration: float,
        input_distribution: str,
        alpha: Optional[float] = None,
        num_slices: Optional[int] = None,
        scope_offset: int = 0,
    ):
        assert input_distribution in [
            "binomial",
            "categorical",
            "sliced_categorical",
        ], "Invalid input distribution"
        assert block_size > 0, "Block size must be positive"
        assert 0 < sum_concentration, "Sum concentration must be positive"
        if input_distribution == "categorical":
            assert alpha is not None, "Alpha must be provided for categorical input distribution"
            assert alpha > 0, "Alpha must be positive"

        self.region_graph = region_graph
        self.num_categories = num_categories
        self.block_size = block_size
        self.sum_concentration = sum_concentration
        self.input_distribution = input_distribution
        self.alpha = alpha
        self.num_slices = num_slices
        self.scope_offset = scope_offset

    def build(self) -> Circuit:
        factory = _NodeFactory()
        children = []
        region_cache: Dict = {}
        for partition in self.region_graph.partitions:
            children.extend(
                self._partition_to_product_nodes(partition, region_cache, factory)
            )
        sum_params = np.random.dirichlet(
            np.ones(len(children)) * self.sum_concentration
        )
        root = factory.sum(children, sum_params)
        return Circuit(root)

    def _partition_to_product_nodes(
        self, partition: Partition, region_cache: Dict, factory: _NodeFactory
    ):
        sub_regions = []
        for sub_region in partition.sub_regions:
            sub_regions.append(
                self._region_to_sum_nodes(sub_region, region_cache, factory)
            )

        product_nodes = []
        for i in range(self.block_size):
            product_nodes.append(
                factory.product([sub_regions[j][i] for j in range(len(sub_regions))])
            )
        return product_nodes

    def _region_to_sum_nodes(
        self, region: Region, region_cache: Dict, factory: _NodeFactory
    ):
        if region.scope in region_cache:
            return region_cache[region.scope]
        scope_var = list(region.scope)[0] + self.scope_offset
        if len(region.scope) == 1:
            children = []
            if self.input_distribution == "binomial":
                for _ in range(self.block_size):
                    pmf, _ = self.generate_binomial_pmf(self.num_categories)
                    children.append(factory.categorical(scope_var, pmf))
            elif self.input_distribution == "categorical":
                for _ in range(self.block_size):
                    pmf = np.random.dirichlet(
                        np.ones(self.num_categories) * self.alpha
                    )
                    children.append(factory.categorical(scope_var, pmf))
        else:
            children = []
            for partition in region.partitions:
                children.extend(
                    self._partition_to_product_nodes(partition, region_cache, factory)
                )

        sum_nodes = []
        for _ in range(self.block_size):
            sum_params = np.random.dirichlet(
                np.ones(len(children)) * self.sum_concentration
            )
            sum_nodes.append(factory.sum(children, sum_params))
        region_cache[region.scope] = sum_nodes
        return sum_nodes

    def generate_binomial_pmf(self, n):
        """
        Generates a PMF for a binomial distribution with shape (n).
        Uses log-space calculations for numerical stability.
        """
        p = np.random.uniform(0, 1)
        n_minus_1 = n - 1
        k = np.arange(n)
        log_comb = gammaln(n_minus_1 + 1) - (gammaln(k + 1) + gammaln(n_minus_1 - k + 1))
        log_prob = k * np.log(p) + (n_minus_1 - k) * np.log1p(-p)
        pmf = np.exp(log_comb + log_prob)
        return pmf, p


class EmbeddingBuilder:
    def __init__(
        self,
        num_vars: int,
        num_categories: int,
        sum_arity: int,
        prod_arity: int,
        sum_concentration: float,
        sum_reuse_probability: float,
        prod_reuse_probability: float,
        input_distribution: str,
        alpha: Optional[float] = None,
        scope_offset: int = 0,
    ):
        assert input_distribution in ["binomial", "categorical"], "Invalid input distribution"
        assert 0 <= sum_reuse_probability <= 1, "Sum reuse probability must be between 0 and 1"
        assert 0 <= prod_reuse_probability <= 1, "Prod reuse probability must be between 0 and 1"
        assert 0 < sum_concentration, "Sum concentration must be positive"
        if input_distribution == "categorical":
            assert alpha is not None, "Alpha must be provided for categorical input distribution"
            assert alpha > 0, "Alpha must be positive"

        self.num_vars = num_vars
        self.num_categories = num_categories
        self.sum_arity = sum_arity
        self.prod_arity = prod_arity
        self.sum_concentration = sum_concentration
        self.sum_reuse_probability = sum_reuse_probability
        self.prod_reuse_probability = prod_reuse_probability
        self.input_distribution = input_distribution
        self.alpha = alpha
        self.scope_offset = scope_offset

    def build(self) -> Circuit:
        factory = _NodeFactory()
        scope = frozenset(
            range(self.scope_offset, self.scope_offset + self.num_vars)
        )
        sum_cache: Dict = {}
        prod_cache: Dict = {}
        input_cache: Dict = {}
        root = self._build(scope, sum_cache, prod_cache, input_cache, factory)
        return Circuit(root)

    def _build(self, scope, sum_cache, prod_cache, input_cache, factory: _NodeFactory):
        children = []
        for _ in range(self.sum_arity):
            if len(scope) == 1:
                scope_var = list(scope)[0]
                if self.input_distribution == "binomial":
                    pmf, _ = self.generate_binomial_pmf(self.num_categories)
                    children.append(factory.categorical(scope_var, pmf))
                elif self.input_distribution == "categorical":
                    pmf = np.random.dirichlet(
                        np.ones(self.num_categories) * self.alpha
                    )
                    children.append(factory.categorical(scope_var, pmf))
            else:
                if len(prod_cache.get(scope, [])) == 0:
                    prod_cache[scope] = []
                if (
                    random.random() < self.prod_reuse_probability
                    and len(prod_cache[scope]) > 0
                ):
                    product_node = random.choice(prod_cache[scope])
                    children.append(product_node)
                    continue
                partition = self.partition_set(scope, self.prod_arity)

                product_children = []
                for child_scope in partition:
                    if len(child_scope) == 0:
                        continue
                    if len(sum_cache.get(child_scope, [])) == 0:
                        sum_cache[child_scope] = []
                        product_child = self._build(
                            child_scope, sum_cache, prod_cache, input_cache, factory
                        )
                        sum_cache[child_scope].append(product_child)
                        product_children.append(product_child)
                    else:
                        if (
                            random.random() < self.sum_reuse_probability
                            and len(sum_cache[child_scope]) > 0
                        ):
                            product_children.append(
                                random.choice(sum_cache[child_scope])
                            )
                        else:
                            product_child = self._build(
                                child_scope,
                                sum_cache,
                                prod_cache,
                                input_cache,
                                factory,
                            )
                            sum_cache[child_scope].append(product_child)
                            product_children.append(product_child)
                product_node = factory.product(product_children)
                children.append(product_node)

        if len(scope) == 1:
            if input_cache.get(scope, None) is None:
                input_cache[scope] = []
            for child in children:
                input_cache[scope].append(child)
        else:
            if len(prod_cache.get(scope, [])) == 0:
                prod_cache[scope] = []
            for child in children:
                prod_cache[scope].append(child)

        sum_params = np.random.dirichlet(
            np.ones(len(children)) * self.sum_concentration
        )
        sum_node = factory.sum(children, sum_params)
        if sum_cache.get(scope, None) is None:
            sum_cache[scope] = []
        sum_cache[scope].append(sum_node)
        return sum_node

    def partition_set(self, input_set, k):
        elements = list(input_set)
        random.shuffle(elements)
        return [frozenset(elements[i::k]) for i in range(k)]

    def generate_binomial_pmf(self, n):
        """
        Generates a PMF for a binomial distribution with shape (n).
        Uses log-space calculations for numerical stability.
        """
        p = np.random.uniform(0, 1)
        n_minus_1 = n - 1
        k = np.arange(n)
        log_comb = gammaln(n_minus_1 + 1) - (gammaln(k + 1) + gammaln(n_minus_1 - k + 1))
        log_prob = k * np.log(p) + (n_minus_1 - k) * np.log1p(-p)
        pmf = np.exp(log_comb + log_prob)
        return pmf, p
