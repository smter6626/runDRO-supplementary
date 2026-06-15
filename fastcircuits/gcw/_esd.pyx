# distutils: language = c++

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np

from fastcircuits.nodes cimport (
    CategoricalInputNode,
    CircuitNode,
    ProductNode,
    SumNode,
)

from fastcircuits.gcw._common cimport (
    GCWGradients,
    fill_pairwise_distance,
    node_py_id,
    _unwrap_root,
)

cdef class ESDContext:
    """Expected squared distance E[d(x,x')^2] for a single circuit.

    Forward computes (mu, nu) at each node where mu = E[d] and nu = E[d^2]
    under independent draws. Backward is reverse-mode AD on the smooth
    (mu, nu) recursion; reuses :class:`GCWGradients` for the return bundle.
    """

    cdef double metric_p
    cdef double scale_factor
    cdef unordered_map[uint64_t, vector[double]] dist_cache
    cdef unordered_map[size_t, double] mu_cache
    cdef unordered_map[size_t, double] nu_cache
    cdef list order
    cdef unordered_map[size_t, double] bar_mu
    cdef unordered_map[size_t, double] bar_nu
    cdef dict sum_grads
    cdef dict cat_grads

    def __cinit__(self):
        self.order = []
        self.sum_grads = {}
        self.cat_grads = {}

    cdef uint64_t dist_cache_key(self, int scope_var, size_t n_bins) noexcept nogil:
        cdef uint64_t key = (<uint64_t>scope_var) << 32
        key |= <uint64_t>n_bins
        return key

    cdef vector[double]* get_distance_matrix(
        self,
        int scope_var,
        size_t n_bins,
    ) except *:
        cdef uint64_t key = self.dist_cache_key(scope_var, n_bins)
        if self.dist_cache.find(key) == self.dist_cache.end():
            fill_pairwise_distance(
                self.dist_cache[key], n_bins, self.metric_p, self.scale_factor
            )
        return &self.dist_cache[key]

    cdef void _reset(self):
        self.dist_cache.clear()
        self.mu_cache.clear()
        self.nu_cache.clear()
        self.order = []
        self.bar_mu.clear()
        self.bar_nu.clear()
        self.sum_grads = {}
        self.cat_grads = {}

    cdef object _sum_grad_arr(self, CircuitNode node, size_t nc):
        cdef object arr
        cdef int key = int(node.id)
        if key in self.sum_grads:
            return self.sum_grads[key]
        arr = np.zeros(nc, dtype=np.float64)
        self.sum_grads[key] = arr
        return arr

    cdef object _cat_grad_arr(self, CircuitNode node, size_t n):
        cdef object arr
        cdef int key = int(node.id)
        if key in self.cat_grads:
            return self.cat_grads[key]
        arr = np.zeros(n, dtype=np.float64)
        self.cat_grads[key] = arr
        return arr

    cdef tuple _forward_node(self, CircuitNode node) except *:
        cdef size_t nid = node_py_id(node)
        cdef unordered_map[size_t, double].iterator mu_it = self.mu_cache.find(nid)
        if mu_it != self.mu_cache.end():
            return (deref(mu_it).second, self.nu_cache[nid])

        cdef double mu
        cdef double nu
        cdef double total_mu
        cdef double total_nu
        cdef double sum_child_mu_sq
        cdef size_t i
        cdef size_t nc
        cdef size_t n_outcomes
        cdef size_t a
        cdef size_t b
        cdef double pa
        cdef double pb
        cdef double d_ab
        cdef CircuitNode child
        cdef vector[double]* d_x_ptr
        cdef CategoricalInputNode cat_node
        cdef SumNode sum_node
        cdef ProductNode prod_node
        cdef double child_mu
        cdef double child_nu

        if isinstance(node, CategoricalInputNode):
            cat_node = <CategoricalInputNode>node
            n_outcomes = cat_node.probabilities.size()
            d_x_ptr = self.get_distance_matrix(cat_node.scope_var_c(), n_outcomes)
            total_mu = 0.0
            total_nu = 0.0
            for a in range(n_outcomes):
                pa = cat_node.probabilities[a]
                for b in range(n_outcomes):
                    pb = cat_node.probabilities[b]
                    d_ab = deref(d_x_ptr)[a * n_outcomes + b]
                    total_mu += pa * d_ab * pb
                    total_nu += pa * d_ab * d_ab * pb
            mu = total_mu
            nu = total_nu
        elif isinstance(node, SumNode):
            sum_node = <SumNode>node
            total_mu = 0.0
            total_nu = 0.0
            nc = sum_node.num_children()
            for i in range(nc):
                child = sum_node.child_at(i)
                child_mu, child_nu = self._forward_node(child)
                total_mu += sum_node.parameter_at(i) * child_mu
                total_nu += sum_node.parameter_at(i) * child_nu
            mu = total_mu
            nu = total_nu
        elif isinstance(node, ProductNode):
            prod_node = <ProductNode>node
            total_mu = 0.0
            total_nu = 0.0
            sum_child_mu_sq = 0.0
            nc = prod_node.num_children()
            for i in range(nc):
                child = prod_node.child_at(i)
                child_mu, child_nu = self._forward_node(child)
                total_mu += child_mu
                total_nu += child_nu
                sum_child_mu_sq += child_mu * child_mu
            mu = total_mu
            nu = total_nu + mu * mu - sum_child_mu_sq
        else:
            raise TypeError(f"unsupported node type: {type(node)}")

        self.mu_cache[nid] = mu
        self.nu_cache[nid] = nu
        self.order.append(node)
        return (mu, nu)

    cdef void _backward(self) except *:
        cdef ssize_t k
        cdef CircuitNode node
        cdef size_t nid
        cdef double adj_mu
        cdef double adj_nu
        cdef unordered_map[size_t, double].iterator mu_it
        cdef unordered_map[size_t, double].iterator nu_it
        cdef size_t i
        cdef size_t nc
        cdef size_t n_outcomes
        cdef size_t kk
        cdef size_t j
        cdef CircuitNode child
        cdef object grads_arr
        cdef vector[double]* d_x_ptr
        cdef double accum
        cdef double d_ab
        cdef double child_mu
        cdef double child_nu
        cdef double theta_i
        cdef double mu_node
        cdef CategoricalInputNode cat_node
        cdef SumNode sum_node
        cdef ProductNode prod_node

        for k in range(<ssize_t>len(self.order) - 1, -1, -1):
            node = <CircuitNode>self.order[k]
            nid = node_py_id(node)
            mu_it = self.bar_mu.find(nid)
            nu_it = self.bar_nu.find(nid)
            adj_mu = 0.0 if mu_it == self.bar_mu.end() else deref(mu_it).second
            adj_nu = 0.0 if nu_it == self.bar_nu.end() else deref(nu_it).second
            if adj_mu == 0.0 and adj_nu == 0.0:
                continue

            if isinstance(node, CategoricalInputNode):
                cat_node = <CategoricalInputNode>node
                n_outcomes = cat_node.probabilities.size()
                d_x_ptr = self.get_distance_matrix(cat_node.scope_var_c(), n_outcomes)
                grads_arr = self._cat_grad_arr(cat_node, n_outcomes)
                for kk in range(n_outcomes):
                    accum = 0.0
                    for j in range(n_outcomes):
                        d_ab = deref(d_x_ptr)[kk * n_outcomes + j]
                        accum += d_ab * cat_node.probabilities[j]
                    grads_arr[kk] += adj_mu * 2.0 * accum
                    accum = 0.0
                    for j in range(n_outcomes):
                        d_ab = deref(d_x_ptr)[kk * n_outcomes + j]
                        accum += d_ab * d_ab * cat_node.probabilities[j]
                    grads_arr[kk] += adj_nu * 2.0 * accum
            elif isinstance(node, SumNode):
                sum_node = <SumNode>node
                nc = sum_node.num_children()
                grads_arr = self._sum_grad_arr(sum_node, nc)
                for i in range(nc):
                    child = sum_node.child_at(i)
                    child_mu = self.mu_cache[node_py_id(child)]
                    child_nu = self.nu_cache[node_py_id(child)]
                    theta_i = sum_node.parameter_at(i)
                    grads_arr[i] += adj_nu * child_nu + adj_mu * child_mu
                    self.bar_mu[node_py_id(child)] = (
                        self.bar_mu[node_py_id(child)] + adj_mu * theta_i
                    )
                    self.bar_nu[node_py_id(child)] = (
                        self.bar_nu[node_py_id(child)] + adj_nu * theta_i
                    )
            elif isinstance(node, ProductNode):
                prod_node = <ProductNode>node
                nc = prod_node.num_children()
                mu_node = self.mu_cache[nid]
                for i in range(nc):
                    child = prod_node.child_at(i)
                    child_mu = self.mu_cache[node_py_id(child)]
                    self.bar_mu[node_py_id(child)] = (
                        self.bar_mu[node_py_id(child)]
                        + adj_mu
                        + adj_nu * (2.0 * mu_node - 2.0 * child_mu)
                    )
                    self.bar_nu[node_py_id(child)] = (
                        self.bar_nu[node_py_id(child)] + adj_nu
                    )

    cdef double solve(self, CircuitNode root) except *:
        self._reset()
        _, nu = self._forward_node(root)
        return nu

    cdef tuple solve_with_grad(self, CircuitNode root):
        cdef double value
        cdef GCWGradients grads
        self._reset()
        _, value = self._forward_node(root)
        self.bar_nu[node_py_id(root)] = 1.0
        self._backward()
        grads = GCWGradients()
        grads.value = value
        grads.sum_grads = self.sum_grads
        grads.cat_grads = self.cat_grads
        return (value, grads)

cpdef double expected_squared_distance(
    object circuit,
    double metric_p=1.0,
    double scale_factor=1.0,
) except *:
    """Compute E[d(x, x')^2] for independent draws from ``circuit``.

    Uses the same separable L_p metric on categorical indices as the GCW
    cross-term query (``|i-j|^p / scale_factor``).
    """
    cdef CircuitNode root = _unwrap_root(circuit)
    cdef ESDContext ctx = ESDContext()
    ctx.metric_p = metric_p
    ctx.scale_factor = scale_factor
    return ctx.solve(root)


cpdef tuple expected_squared_distance_and_grad(
    object circuit,
    double metric_p=1.0,
    double scale_factor=1.0,
):
    """Compute expected squared distance and gradients w.r.t. all circuit params.

    Returns ``(value, grads)`` where ``grads`` is a :class:`GCWGradients`
    bundle with per-node gradient vectors for every ``SumNode`` and
    ``CategoricalInputNode`` in the circuit.
    """
    cdef CircuitNode root = _unwrap_root(circuit)
    cdef ESDContext ctx = ESDContext()
    ctx.metric_p = metric_p
    ctx.scale_factor = scale_factor
    return ctx.solve_with_grad(root)
