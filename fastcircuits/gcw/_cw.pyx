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
    CATEGORICAL,
    GCWGradients,
    GCWTapeEntry,
    NO_TAPE_IDX,
    NodeKind,
    PRODUCT,
    SUM,
    TAPE_CW_LEAF,
    TAPE_CW_PROD_PROD,
    TAPE_CW_SUM_SUM,
    dist_at_cross,
    fill_cross_distance,
    node_kind,
    node_py_id,
    nw_backward_marginals,
    nw_plan,
    solve_transport_lp,
    solve_transport_lp_with_duals,
    _scope_frozen,
    _unwrap_root,
)

cdef class CWContext:
    """Circuit-Wasserstein distance between two compatible probabilistic circuits.

    Returns the ``W_p^p`` objective (additive under the CW recursion). The
    ``p``-th root gives the true CW distance.
    """

    cdef unordered_map[size_t, unordered_map[size_t, double]] couple_memo
    cdef double metric_p
    cdef double scale_factor
    cdef object gurobi_env
    cdef unordered_map[uint64_t, vector[double]] cross_dist_cache

    cdef bint recording
    cdef list tape
    cdef vector[double] tape_adjoints
    cdef unordered_map[size_t, unordered_map[size_t, size_t]] pair_to_tape
    cdef dict sum_grads
    cdef dict cat_grads

    def __cinit__(self):
        self.recording = False
        self.tape = []
        self.sum_grads = {}
        self.cat_grads = {}

    cdef uint64_t cross_dist_cache_key(self, size_t n, size_t m) noexcept nogil:
        return (<uint64_t>n << 32) | <uint64_t>m

    cdef vector[double]* get_cross_distance_matrix(self, size_t n, size_t m) except *:
        cdef uint64_t key = self.cross_dist_cache_key(n, m)
        if self.cross_dist_cache.find(key) == self.cross_dist_cache.end():
            fill_cross_distance(
                self.cross_dist_cache[key], n, m, self.metric_p, self.scale_factor
            )
        return &self.cross_dist_cache[key]

    cdef size_t _lookup_pair_tape_idx(self, CircuitNode P, CircuitNode Q) except *:
        cdef size_t id_p = node_py_id(P)
        cdef size_t id_q = node_py_id(Q)
        cdef size_t key_a
        cdef size_t key_b
        if id_p <= id_q:
            key_a = id_p
            key_b = id_q
        else:
            key_a = id_q
            key_b = id_p
        cdef unordered_map[size_t, unordered_map[size_t, size_t]].iterator outer = self.pair_to_tape.find(key_a)
        if outer == self.pair_to_tape.end():
            return NO_TAPE_IDX
        cdef unordered_map[size_t, size_t].iterator inner = deref(outer).second.find(key_b)
        if inner == deref(outer).second.end():
            return NO_TAPE_IDX
        return deref(inner).second

    cdef size_t _append_tape(self, GCWTapeEntry entry, CircuitNode P, CircuitNode Q) except *:
        cdef size_t idx = <size_t>len(self.tape)
        cdef size_t id_p = node_py_id(P)
        cdef size_t id_q = node_py_id(Q)
        cdef size_t key_a
        cdef size_t key_b
        if id_p <= id_q:
            key_a = id_p
            key_b = id_q
        else:
            key_a = id_q
            key_b = id_p
        self.tape.append(entry)
        self.tape_adjoints.push_back(0.0)
        self.pair_to_tape[key_a][key_b] = idx
        return idx

    cdef object _cat_grad_arr(self, CircuitNode node, size_t n):
        cdef object arr
        cdef int key = int(node.id)
        if key in self.cat_grads:
            return self.cat_grads[key]
        arr = np.zeros(n, dtype=np.float64)
        self.cat_grads[key] = arr
        return arr

    cdef object _sum_grad_arr(self, CircuitNode node, size_t nc):
        cdef object arr
        cdef int key = int(node.id)
        if key in self.sum_grads:
            return self.sum_grads[key]
        arr = np.zeros(nc, dtype=np.float64)
        self.sum_grads[key] = arr
        return arr

    cdef double couple_value(
        self,
        CircuitNode P,
        CircuitNode Q,
        int side_P,
        int side_Q,
    ) except *:
        cdef size_t id_p = node_py_id(P)
        cdef size_t id_q = node_py_id(Q)
        cdef size_t key_a
        cdef size_t key_b
        cdef unordered_map[size_t, double].iterator inner_it
        cdef NodeKind pk
        cdef NodeKind qk
        cdef double res

        if id_p <= id_q:
            key_a = id_p
            key_b = id_q
        else:
            key_a = id_q
            key_b = id_p

        inner_it = self.couple_memo[key_a].find(key_b)
        if inner_it != self.couple_memo[key_a].end():
            return deref(inner_it).second

        pk = node_kind(P)
        qk = node_kind(Q)

        if pk == CATEGORICAL and qk == CATEGORICAL:
            res = self._cw_leaf(
                <CategoricalInputNode>P, <CategoricalInputNode>Q, side_P, side_Q
            )
        elif pk == SUM and qk == SUM:
            res = self._cw_sum_sum(<SumNode>P, <SumNode>Q, side_P, side_Q)
        elif pk == PRODUCT and qk == PRODUCT:
            res = self._cw_prod_prod(<ProductNode>P, <ProductNode>Q, side_P, side_Q)
        else:
            raise ValueError(
                f"CW incompatible: cannot couple {type(P).__name__} with "
                f"{type(Q).__name__}"
            )

        self.couple_memo[key_a][key_b] = res
        return res

    cdef double _cw_leaf(
        self,
        CategoricalInputNode P,
        CategoricalInputNode Q,
        int side_P,
        int side_Q,
    ) except *:
        cdef size_t n = P.num_outcomes()
        cdef size_t m = Q.num_outcomes()
        cdef vector[double]* d_ptr = self.get_cross_distance_matrix(n, m)
        cdef GCWTapeEntry entry
        cdef double value = 0.0
        cdef size_t a
        cdef size_t num_steps
        cdef int i_a
        cdef int j_a
        cdef vector[int] rows_tmp
        cdef vector[int] cols_tmp
        cdef vector[double] vals_tmp
        cdef vector[int] modes_tmp

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_CW_LEAF
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.leaf_n = n
            entry.leaf_m = m
            entry.leaf_p_scope = P.scope_var_c()
            entry.leaf_q_scope = Q.scope_var_c()
            num_steps = nw_plan(
                P.probabilities, Q.probabilities, n, m,
                entry.leaf_rows, entry.leaf_cols, entry.leaf_vals, entry.leaf_modes,
            )
            for a in range(num_steps):
                i_a = entry.leaf_rows[a]
                j_a = entry.leaf_cols[a]
                value += entry.leaf_vals[a] * dist_at_cross(deref(d_ptr), m, i_a, j_a)
            self._append_tape(entry, P, Q)
            return value

        num_steps = nw_plan(
            P.probabilities, Q.probabilities, n, m,
            rows_tmp, cols_tmp, vals_tmp, modes_tmp,
        )
        for a in range(num_steps):
            i_a = rows_tmp[a]
            j_a = cols_tmp[a]
            value += vals_tmp[a] * dist_at_cross(deref(d_ptr), m, i_a, j_a)
        return value

    cdef double _cw_sum_sum(
        self,
        SumNode P,
        SumNode Q,
        int side_P,
        int side_Q,
    ) except *:
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef size_t i
        cdef size_t j
        cdef CircuitNode p_child
        cdef CircuitNode q_child
        cdef vector[vector[double]] value_rows
        cdef object C_np
        cdef object theta_np
        cdef object phi_np
        cdef object weights_matrix
        cdef object pi_row_np
        cdef object pi_col_np
        cdef double cross_term = 0.0
        cdef double w
        cdef GCWTapeEntry entry
        cdef vector[size_t] child_indices
        cdef size_t child_idx

        value_rows.resize(n)
        if self.recording:
            child_indices.resize(n * m)

        for i in range(n):
            value_rows[i].resize(m)
            p_child = P.child_at(i)
            for j in range(m):
                q_child = Q.child_at(j)
                value_rows[i][j] = self.couple_value(p_child, q_child, side_P, side_Q)
                if self.recording:
                    child_idx = self._lookup_pair_tape_idx(p_child, q_child)
                    child_indices[i * m + j] = child_idx

        C_np = np.empty((n, m), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                C_np[i, j] = value_rows[i][j]

        theta_np = np.empty(n, dtype=np.float64)
        phi_np = np.empty(m, dtype=np.float64)
        for i in range(n):
            theta_np[i] = P.parameter_at(i)
        for j in range(m):
            phi_np[j] = Q.parameter_at(j)

        if self.recording:
            weights_matrix, pi_row_np, pi_col_np = solve_transport_lp_with_duals(
                C_np, theta_np, phi_np, self.gurobi_env
            )
        else:
            weights_matrix = solve_transport_lp(
                C_np, theta_np, phi_np, self.gurobi_env
            )

        for i in range(n):
            for j in range(m):
                w = weights_matrix[i, j]
                if w > 0.0:
                    cross_term += w * value_rows[i][j]

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_CW_SUM_SUM
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.ss_n = n
            entry.ss_m = m
            entry.ss_w.resize(n * m)
            entry.ss_V.resize(n * m)
            for i in range(n):
                for j in range(m):
                    entry.ss_w[i * m + j] = weights_matrix[i, j]
                    entry.ss_V[i * m + j] = value_rows[i][j]
            entry.ss_pi.resize(n)
            entry.ss_rho.resize(m)
            for i in range(n):
                entry.ss_pi[i] = pi_row_np[i]
            for j in range(m):
                entry.ss_rho[j] = pi_col_np[j]
            entry.ss_child_pair_indices = child_indices
            self._append_tape(entry, P, Q)

        return cross_term

    cdef double _cw_prod_prod(
        self,
        ProductNode P,
        ProductNode Q,
        int side_P,
        int side_Q,
    ) except *:
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef size_t i
        cdef size_t j
        cdef CircuitNode p_child
        cdef CircuitNode q_child
        cdef double total = 0.0
        cdef GCWTapeEntry entry
        cdef vector[size_t] child_indices
        cdef size_t child_idx
        cdef object scope_to_q
        cdef object p_key
        cdef object q_key
        cdef Py_ssize_t q_idx
        cdef list p_children_list = None
        cdef list q_children_list = None
        cdef vector[int] row_ind_vec
        cdef vector[int] col_ind_vec
        cdef Py_ssize_t k

        if n != m:
            raise ValueError(
                f"CW incompatible: product nodes have different numbers of "
                f"children ({n} vs {m})"
            )

        scope_to_q = {}
        for j in range(m):
            q_child = Q.child_at(j)
            q_key = _scope_frozen(q_child)
            if q_key in scope_to_q:
                raise ValueError(
                    "CW incompatible: duplicate child scope among Q product children"
                )
            scope_to_q[q_key] = j

        row_ind_vec.resize(n)
        col_ind_vec.resize(n)
        if self.recording:
            child_indices.resize(n * m)
            p_children_list = []
            q_children_list = []
            for i in range(n):
                p_children_list.append(P.child_at(i))
            for j in range(m):
                q_children_list.append(Q.child_at(j))

        for i in range(n):
            p_child = P.child_at(i)
            p_key = _scope_frozen(p_child)
            if p_key not in scope_to_q:
                raise ValueError(
                    "CW incompatible: no Q product child with scope matching "
                    f"P child at index {i}"
                )
            q_idx = scope_to_q[p_key]
            del scope_to_q[p_key]
            q_child = Q.child_at(<size_t>q_idx)
            total += self.couple_value(p_child, q_child, side_P, side_Q)
            row_ind_vec[i] = <int>i
            col_ind_vec[i] = <int>q_idx
            if self.recording:
                child_idx = self._lookup_pair_tape_idx(p_child, q_child)
                child_indices[i * m + <size_t>q_idx] = child_idx

        if len(scope_to_q) > 0:
            raise ValueError(
                "CW incompatible: Q product children with unmatched scopes"
            )

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_CW_PROD_PROD
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.pp_n = n
            entry.pp_m = m
            entry.pp_p_children = p_children_list
            entry.pp_q_children = q_children_list
            entry.pp_row_ind.resize(n)
            entry.pp_col_ind.resize(n)
            for k in range(<Py_ssize_t>n):
                entry.pp_row_ind[<size_t>k] = row_ind_vec[<size_t>k]
                entry.pp_col_ind[<size_t>k] = col_ind_vec[<size_t>k]
            entry.pp_child_pair_indices = child_indices
            self._append_tape(entry, P, Q)

        return total

    cdef void _run_backward(self) except *:
        cdef ssize_t k
        cdef double g
        cdef GCWTapeEntry entry
        for k in range(<ssize_t>len(self.tape) - 1, -1, -1):
            g = self.tape_adjoints[<size_t>k]
            if g == 0.0:
                continue
            entry = <GCWTapeEntry>self.tape[k]
            if entry.kind == TAPE_CW_LEAF:
                self._backward_cw_leaf(entry, g)
            elif entry.kind == TAPE_CW_SUM_SUM:
                self._backward_cw_sum_sum(entry, g)
            elif entry.kind == TAPE_CW_PROD_PROD:
                self._backward_cw_prod_prod(entry, g)

    cdef void _backward_cw_leaf(self, GCWTapeEntry entry, double g) except *:
        cdef size_t n = entry.leaf_n
        cdef size_t m = entry.leaf_m
        cdef size_t num_steps = entry.leaf_rows.size()
        cdef vector[double]* d_ptr = self.get_cross_distance_matrix(n, m)
        cdef vector[double] G
        cdef vector[double] adj_p
        cdef vector[double] adj_q
        cdef size_t a
        cdef int i_a
        cdef int j_a
        cdef object grads_arr

        G.resize(num_steps)
        for a in range(num_steps):
            i_a = entry.leaf_rows[a]
            j_a = entry.leaf_cols[a]
            G[a] = g * dist_at_cross(deref(d_ptr), m, i_a, j_a)

        nw_backward_marginals(
            entry.leaf_rows, entry.leaf_cols, entry.leaf_modes,
            G, n, m, adj_p, adj_q,
        )

        if entry.side_P == 1:
            grads_arr = self._cat_grad_arr(entry.P, n)
            for a in range(n):
                grads_arr[a] += adj_p[a]
        if entry.side_Q == 1:
            grads_arr = self._cat_grad_arr(entry.Q, m)
            for a in range(m):
                grads_arr[a] += adj_q[a]

    cdef void _backward_cw_sum_sum(self, GCWTapeEntry entry, double g) except *:
        """value = sum V_ij w*_ij; dV/dV_ij = w*_ij; dV/dtheta = +pi; dV/dphi = +rho."""
        cdef size_t n = entry.ss_n
        cdef size_t m = entry.ss_m
        cdef size_t i
        cdef size_t j
        cdef size_t child_idx
        cdef object grads_arr

        for i in range(n):
            for j in range(m):
                child_idx = entry.ss_child_pair_indices[i * m + j]
                if child_idx != NO_TAPE_IDX:
                    self.tape_adjoints[child_idx] += g * entry.ss_w[i * m + j]

        if entry.side_P == 1:
            grads_arr = self._sum_grad_arr(entry.P, n)
            for i in range(n):
                grads_arr[i] += g * entry.ss_pi[i]

        if entry.side_Q == 1:
            grads_arr = self._sum_grad_arr(entry.Q, m)
            for j in range(m):
                grads_arr[j] += g * entry.ss_rho[j]

    cdef void _backward_cw_prod_prod(self, GCWTapeEntry entry, double g) except *:
        cdef size_t n = entry.pp_n
        cdef size_t m = entry.pp_m
        cdef size_t k_idx
        cdef size_t child_idx
        cdef int r
        cdef int c
        cdef size_t num_matches = entry.pp_row_ind.size()

        for k_idx in range(num_matches):
            r = entry.pp_row_ind[k_idx]
            c = entry.pp_col_ind[k_idx]
            child_idx = entry.pp_child_pair_indices[<size_t>r * m + <size_t>c]
            if child_idx != NO_TAPE_IDX:
                self.tape_adjoints[child_idx] += g

    cdef void _reset(self):
        self.couple_memo.clear()
        self.cross_dist_cache.clear()
        self.tape = []
        self.tape_adjoints.clear()
        self.pair_to_tape.clear()
        self.sum_grads = {}
        self.cat_grads = {}

    cdef double solve(self, CircuitNode circuit1, CircuitNode circuit2) except *:
        self._reset()
        return self.couple_value(circuit1, circuit2, 0, 1)

    cdef tuple solve_with_grad(self, CircuitNode circuit1, CircuitNode circuit2):
        cdef double value
        cdef size_t root_idx
        cdef GCWGradients grads
        self.recording = True
        try:
            value = self.solve(circuit1, circuit2)
            root_idx = self._lookup_pair_tape_idx(circuit1, circuit2)
            if root_idx == NO_TAPE_IDX:
                raise RuntimeError(
                    "internal: root pair has no tape entry; this should not happen"
                )
            self.tape_adjoints[root_idx] = 1.0
            self._run_backward()
        finally:
            self.recording = False

        grads = GCWGradients()
        grads.value = value
        grads.sum_grads = self.sum_grads
        grads.cat_grads = self.cat_grads
        return (value, grads)

cpdef double cw_distance(
    object circuit1,
    object circuit2,
    double metric_p=1.0,
    double scale_factor=1.0,
    object gurobi_env=None,
) except *:
    """Compute the Circuit-Wasserstein ``W_p^p`` objective between two PCs.

    Returns the additive ``W_p^p`` value from the CW recursion (Eq. 2 in
    Ciotinga & Choi, 2025). The ``p``-th root gives the CW distance. Does not
    materialize a coupling circuit.

    Parameters
    ----------
    circuit1, circuit2 : CircuitNode or Circuit
        Compatible circuit roots with propagated scope.
    metric_p : float
        Exponent for separable ground metric ``|i-j|^p / scale_factor``.
    scale_factor : float
        Scale divisor for the ground metric.
    gurobi_env : gurobipy.Env
        Shared Gurobi environment (required for sum x sum transport subproblems).
    """
    if gurobi_env is None:
        raise ValueError("gurobi_env is required for CW distance computation")

    cdef CircuitNode root1 = _unwrap_root(circuit1)
    cdef CircuitNode root2 = _unwrap_root(circuit2)
    cdef CWContext ctx = CWContext()
    ctx.metric_p = metric_p
    ctx.scale_factor = scale_factor
    ctx.gurobi_env = gurobi_env
    return ctx.solve(root1, root2)


cpdef tuple cw_distance_and_grad(
    object circuit1,
    object circuit2,
    double metric_p=1.0,
    double scale_factor=1.0,
    object gurobi_env=None,
):
    """Compute CW ``W_p^p`` and subgradients w.r.t. ``circuit2``.

    Returns ``(value, grads)`` where ``value`` matches ``cw_distance`` and
    ``grads`` is a :class:`GCWGradients` bundle (sum and categorical entries
    for nodes in ``circuit2`` only).
    """
    if gurobi_env is None:
        raise ValueError("gurobi_env is required for CW distance computation")

    cdef CircuitNode root1 = _unwrap_root(circuit1)
    cdef CircuitNode root2 = _unwrap_root(circuit2)
    cdef CWContext ctx = CWContext()
    ctx.metric_p = metric_p
    ctx.scale_factor = scale_factor
    ctx.gurobi_env = gurobi_env
    return ctx.solve_with_grad(root1, root2)
