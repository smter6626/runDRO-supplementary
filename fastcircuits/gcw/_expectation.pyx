# distutils: language = c++

from cython.operator cimport dereference as deref
from libc.math cimport INFINITY, exp, log
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
    TAPE_EXP_LEAF,
    TAPE_EXP_PROD_PROD,
    TAPE_EXP_SUM_SUM,
    TAPE_LOGEXP_LEAF,
    TAPE_LOGEXP_PROD_PROD,
    TAPE_LOGEXP_SUM_SUM,
    node_kind,
    node_py_id,
    _scope_frozen,
    _unwrap_root,
)

cdef inline double _safe_log_prob(double x) noexcept nogil:
    if x > 0.0:
        return log(x)
    return -INFINITY


cdef class LogExpectationContext:
    """Log-space inner product: returns log(E_Q[P(X)]) with log-objective gradients."""

    cdef unordered_map[size_t, unordered_map[size_t, double]] couple_memo
    cdef bint recording
    cdef list tape
    cdef vector[double] tape_adjoints
    cdef unordered_map[size_t, unordered_map[size_t, size_t]] pair_to_tape
    cdef dict c1_sum_grads
    cdef dict c1_cat_grads
    cdef dict c2_sum_grads
    cdef dict c2_cat_grads

    def __cinit__(self):
        self.recording = False
        self.tape = []
        self.c1_sum_grads = {}
        self.c1_cat_grads = {}
        self.c2_sum_grads = {}
        self.c2_cat_grads = {}

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

    cdef object _sum_grad_arr(self, int side, CircuitNode node, size_t nc):
        cdef object arr
        cdef int key = int(node.id)
        if side == 0:
            if key in self.c1_sum_grads:
                return self.c1_sum_grads[key]
            arr = np.zeros(nc, dtype=np.float64)
            self.c1_sum_grads[key] = arr
            return arr
        if key in self.c2_sum_grads:
            return self.c2_sum_grads[key]
        arr = np.zeros(nc, dtype=np.float64)
        self.c2_sum_grads[key] = arr
        return arr

    cdef object _cat_grad_arr(self, int side, CircuitNode node, size_t n):
        cdef object arr
        cdef int key = int(node.id)
        if side == 0:
            if key in self.c1_cat_grads:
                return self.c1_cat_grads[key]
            arr = np.zeros(n, dtype=np.float64)
            self.c1_cat_grads[key] = arr
            return arr
        if key in self.c2_cat_grads:
            return self.c2_cat_grads[key]
        arr = np.zeros(n, dtype=np.float64)
        self.c2_cat_grads[key] = arr
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
            res = self._logexp_leaf(
                <CategoricalInputNode>P, <CategoricalInputNode>Q, side_P, side_Q
            )
        elif pk == SUM and qk == SUM:
            res = self._logexp_sum_sum(<SumNode>P, <SumNode>Q, side_P, side_Q)
        elif pk == PRODUCT and qk == PRODUCT:
            res = self._logexp_prod_prod(<ProductNode>P, <ProductNode>Q, side_P, side_Q)
        else:
            raise ValueError(
                f"expectation incompatible: cannot couple {type(P).__name__} with "
                f"{type(Q).__name__}"
            )

        self.couple_memo[key_a][key_b] = res
        return res

    cdef double _logexp_leaf(
        self,
        CategoricalInputNode P,
        CategoricalInputNode Q,
        int side_P,
        int side_Q,
    ) except *:
        cdef size_t n = P.num_outcomes()
        cdef size_t m = Q.num_outcomes()
        cdef size_t k
        cdef double max_val
        cdef double total = 0.0
        cdef double term
        cdef double ell
        cdef GCWTapeEntry entry

        if P.scope_var_c() != Q.scope_var_c():
            raise ValueError(
                "expectation incompatible: categorical nodes have different scopes"
            )
        if n != m:
            raise ValueError(
                f"expectation incompatible: categorical nodes have different "
                f"cardinalities ({n} vs {m})"
            )

        max_val = -INFINITY
        for k in range(n):
            term = _safe_log_prob(P.probabilities[k]) + _safe_log_prob(Q.probabilities[k])
            if term > max_val:
                max_val = term

        for k in range(n):
            term = _safe_log_prob(P.probabilities[k]) + _safe_log_prob(Q.probabilities[k])
            if term > -INFINITY:
                total += exp(term - max_val)

        if total <= 0.0:
            ell = -INFINITY
        else:
            ell = max_val + log(total)

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_LOGEXP_LEAF
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.leaf_n = n
            entry.logexp_ell = ell
            self._append_tape(entry, P, Q)

        return ell

    cdef double _logexp_sum_sum(
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
        cdef double max_val = -INFINITY
        cdef double total = 0.0
        cdef double log_theta_i
        cdef double log_phi_j
        cdef double ell_ij
        cdef double term
        cdef double ell
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

        for i in range(n):
            log_theta_i = _safe_log_prob(P.parameter_at(i))
            for j in range(m):
                log_phi_j = _safe_log_prob(Q.parameter_at(j))
                term = log_theta_i + log_phi_j + value_rows[i][j]
                if term > max_val:
                    max_val = term

        for i in range(n):
            log_theta_i = _safe_log_prob(P.parameter_at(i))
            for j in range(m):
                log_phi_j = _safe_log_prob(Q.parameter_at(j))
                term = log_theta_i + log_phi_j + value_rows[i][j]
                if term > -INFINITY:
                    total += exp(term - max_val)

        if total <= 0.0:
            ell = -INFINITY
        else:
            ell = max_val + log(total)

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_LOGEXP_SUM_SUM
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.ss_n = n
            entry.ss_m = m
            entry.ss_V.resize(n * m)
            entry.exp_theta.resize(n)
            entry.exp_phi.resize(m)
            for i in range(n):
                entry.exp_theta[i] = _safe_log_prob(P.parameter_at(i))
                for j in range(m):
                    entry.ss_V[i * m + j] = value_rows[i][j]
            for j in range(m):
                entry.exp_phi[j] = _safe_log_prob(Q.parameter_at(j))
            entry.ss_child_pair_indices = child_indices
            entry.logexp_ell = ell
            self._append_tape(entry, P, Q)

        return ell

    cdef double _logexp_prod_prod(
        self,
        ProductNode P,
        ProductNode Q,
        int side_P,
        int side_Q,
    ) except *:
        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef size_t i
        cdef CircuitNode p_child
        cdef CircuitNode q_child
        cdef double ell = 0.0
        cdef double child_ell
        cdef GCWTapeEntry entry
        cdef vector[size_t] child_indices
        cdef size_t child_idx
        cdef object scope_to_q
        cdef object p_key
        cdef object q_key
        cdef Py_ssize_t q_idx
        cdef vector[int] row_ind_vec
        cdef vector[int] col_ind_vec
        cdef Py_ssize_t k

        if n != m:
            raise ValueError(
                f"expectation incompatible: product nodes have different numbers of "
                f"children ({n} vs {m})"
            )

        scope_to_q = {}
        for j in range(m):
            q_child = Q.child_at(j)
            q_key = _scope_frozen(q_child)
            if q_key in scope_to_q:
                raise ValueError(
                    "expectation incompatible: duplicate child scope among Q product children"
                )
            scope_to_q[q_key] = j

        row_ind_vec.resize(n)
        col_ind_vec.resize(n)
        if self.recording:
            child_indices.resize(n * m)

        for i in range(n):
            p_child = P.child_at(i)
            p_key = _scope_frozen(p_child)
            if p_key not in scope_to_q:
                raise ValueError(
                    "expectation incompatible: no Q product child with scope matching "
                    f"P child at index {i}"
                )
            q_idx = scope_to_q[p_key]
            del scope_to_q[p_key]
            q_child = Q.child_at(<size_t>q_idx)
            child_ell = self.couple_value(p_child, q_child, side_P, side_Q)
            ell += child_ell
            row_ind_vec[i] = <int>i
            col_ind_vec[i] = <int>q_idx
            if self.recording:
                child_idx = self._lookup_pair_tape_idx(p_child, q_child)
                child_indices[i * m + <size_t>q_idx] = child_idx

        if len(scope_to_q) > 0:
            raise ValueError(
                "expectation incompatible: Q product children with unmatched scopes"
            )

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_LOGEXP_PROD_PROD
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.pp_n = n
            entry.pp_m = m
            entry.pp_row_ind.resize(n)
            entry.pp_col_ind.resize(n)
            for k in range(<Py_ssize_t>n):
                entry.pp_row_ind[<size_t>k] = row_ind_vec[<size_t>k]
                entry.pp_col_ind[<size_t>k] = col_ind_vec[<size_t>k]
            entry.pp_child_pair_indices = child_indices
            entry.logexp_ell = ell
            self._append_tape(entry, P, Q)

        return ell

    cdef void _run_backward(self) except *:
        cdef ssize_t k
        cdef double g
        cdef GCWTapeEntry entry
        for k in range(<ssize_t>len(self.tape) - 1, -1, -1):
            g = self.tape_adjoints[<size_t>k]
            if g == 0.0:
                continue
            entry = <GCWTapeEntry>self.tape[k]
            if entry.kind == TAPE_LOGEXP_LEAF:
                self._backward_logexp_leaf(entry, g)
            elif entry.kind == TAPE_LOGEXP_SUM_SUM:
                self._backward_logexp_sum_sum(entry, g)
            elif entry.kind == TAPE_LOGEXP_PROD_PROD:
                self._backward_logexp_prod_prod(entry, g)

    cdef void _backward_logexp_leaf(self, GCWTapeEntry entry, double bar) except *:
        cdef CategoricalInputNode P = <CategoricalInputNode>entry.P
        cdef CategoricalInputNode Q = <CategoricalInputNode>entry.Q
        cdef size_t n = entry.leaf_n
        cdef size_t k
        cdef double ell = entry.logexp_ell
        cdef object grads_arr

        grads_arr = self._cat_grad_arr(0, P, n)
        for k in range(n):
            grads_arr[k] += bar * exp(_safe_log_prob(Q.probabilities[k]) - ell)

        grads_arr = self._cat_grad_arr(1, Q, n)
        for k in range(n):
            grads_arr[k] += bar * exp(_safe_log_prob(P.probabilities[k]) - ell)

    cdef void _backward_logexp_sum_sum(self, GCWTapeEntry entry, double bar) except *:
        cdef size_t n = entry.ss_n
        cdef size_t m = entry.ss_m
        cdef size_t i
        cdef size_t j
        cdef size_t child_idx
        cdef double ell = entry.logexp_ell
        cdef double log_theta_i
        cdef double log_phi_j
        cdef double ell_ij
        cdef double weight
        cdef double sum_phi
        cdef double sum_theta
        cdef object grads_arr

        for i in range(n):
            log_theta_i = entry.exp_theta[i]
            for j in range(m):
                log_phi_j = entry.exp_phi[j]
                ell_ij = entry.ss_V[i * m + j]
                weight = bar * exp(log_theta_i + log_phi_j + ell_ij - ell)
                child_idx = entry.ss_child_pair_indices[i * m + j]
                if child_idx != NO_TAPE_IDX:
                    self.tape_adjoints[child_idx] += weight

        grads_arr = self._sum_grad_arr(0, entry.P, n)
        for i in range(n):
            log_theta_i = entry.exp_theta[i]
            sum_phi = 0.0
            for j in range(m):
                log_phi_j = entry.exp_phi[j]
                ell_ij = entry.ss_V[i * m + j]
                sum_phi += exp(log_phi_j + ell_ij - ell)
            grads_arr[i] += bar * sum_phi

        grads_arr = self._sum_grad_arr(1, entry.Q, m)
        for j in range(m):
            log_phi_j = entry.exp_phi[j]
            sum_theta = 0.0
            for i in range(n):
                log_theta_i = entry.exp_theta[i]
                ell_ij = entry.ss_V[i * m + j]
                sum_theta += exp(log_theta_i + ell_ij - ell)
            grads_arr[j] += bar * sum_theta

    cdef void _backward_logexp_prod_prod(self, GCWTapeEntry entry, double bar) except *:
        cdef size_t n = entry.pp_n
        cdef size_t m = entry.pp_m
        cdef size_t k_idx
        cdef size_t child_idx
        cdef int r
        cdef size_t num_matches = entry.pp_row_ind.size()

        for k_idx in range(num_matches):
            r = entry.pp_row_ind[k_idx]
            child_idx = entry.pp_child_pair_indices[<size_t>r * m + <size_t>entry.pp_col_ind[k_idx]]
            if child_idx != NO_TAPE_IDX:
                self.tape_adjoints[child_idx] += bar

    cdef void _reset(self):
        self.couple_memo.clear()
        self.tape = []
        self.tape_adjoints.clear()
        self.pair_to_tape.clear()
        self.c1_sum_grads = {}
        self.c1_cat_grads = {}
        self.c2_sum_grads = {}
        self.c2_cat_grads = {}

    cdef double solve(self, CircuitNode circuit1, CircuitNode circuit2) except *:
        self._reset()
        return self.couple_value(circuit1, circuit2, 0, 1)

    cdef tuple solve_with_grad(self, CircuitNode circuit1, CircuitNode circuit2):
        cdef double value
        cdef size_t root_idx
        cdef GCWGradients grads1
        cdef GCWGradients grads2
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

        grads1 = GCWGradients()
        grads1.value = value
        grads1.sum_grads = self.c1_sum_grads
        grads1.cat_grads = self.c1_cat_grads

        grads2 = GCWGradients()
        grads2.value = value
        grads2.sum_grads = self.c2_sum_grads
        grads2.cat_grads = self.c2_cat_grads

        return (value, grads1, grads2)


cdef class ExpectationContext:
    """Tractable inner product E_Q[P(X)] = sum_x P(x) Q(x) for compatible PCs.

    Forward is the smooth couple recursion (leaf dot product, sum-sum bilinear,
    prod-prod multiplicative). Backward is exact reverse-mode AD with gradients
    for both circuit1 and circuit2 parameters.
    """

    cdef unordered_map[size_t, unordered_map[size_t, double]] couple_memo
    cdef bint recording
    cdef list tape
    cdef vector[double] tape_adjoints
    cdef unordered_map[size_t, unordered_map[size_t, size_t]] pair_to_tape
    cdef dict c1_sum_grads
    cdef dict c1_cat_grads
    cdef dict c2_sum_grads
    cdef dict c2_cat_grads

    def __cinit__(self):
        self.recording = False
        self.tape = []
        self.c1_sum_grads = {}
        self.c1_cat_grads = {}
        self.c2_sum_grads = {}
        self.c2_cat_grads = {}

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

    cdef object _sum_grad_arr(self, int side, CircuitNode node, size_t nc):
        cdef object arr
        cdef int key = int(node.id)
        if side == 0:
            if key in self.c1_sum_grads:
                return self.c1_sum_grads[key]
            arr = np.zeros(nc, dtype=np.float64)
            self.c1_sum_grads[key] = arr
            return arr
        if key in self.c2_sum_grads:
            return self.c2_sum_grads[key]
        arr = np.zeros(nc, dtype=np.float64)
        self.c2_sum_grads[key] = arr
        return arr

    cdef object _cat_grad_arr(self, int side, CircuitNode node, size_t n):
        cdef object arr
        cdef int key = int(node.id)
        if side == 0:
            if key in self.c1_cat_grads:
                return self.c1_cat_grads[key]
            arr = np.zeros(n, dtype=np.float64)
            self.c1_cat_grads[key] = arr
            return arr
        if key in self.c2_cat_grads:
            return self.c2_cat_grads[key]
        arr = np.zeros(n, dtype=np.float64)
        self.c2_cat_grads[key] = arr
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
            res = self._exp_leaf(
                <CategoricalInputNode>P, <CategoricalInputNode>Q, side_P, side_Q
            )
        elif pk == SUM and qk == SUM:
            res = self._exp_sum_sum(<SumNode>P, <SumNode>Q, side_P, side_Q)
        elif pk == PRODUCT and qk == PRODUCT:
            res = self._exp_prod_prod(<ProductNode>P, <ProductNode>Q, side_P, side_Q)
        else:
            raise ValueError(
                f"expectation incompatible: cannot couple {type(P).__name__} with "
                f"{type(Q).__name__}"
            )

        self.couple_memo[key_a][key_b] = res
        return res

    cdef double _exp_leaf(
        self,
        CategoricalInputNode P,
        CategoricalInputNode Q,
        int side_P,
        int side_Q,
    ) except *:
        cdef size_t n = P.num_outcomes()
        cdef size_t m = Q.num_outcomes()
        cdef size_t k
        cdef double total = 0.0
        cdef GCWTapeEntry entry

        if P.scope_var_c() != Q.scope_var_c():
            raise ValueError(
                "expectation incompatible: categorical nodes have different scopes"
            )
        if n != m:
            raise ValueError(
                f"expectation incompatible: categorical nodes have different "
                f"cardinalities ({n} vs {m})"
            )

        for k in range(n):
            total += P.probabilities[k] * Q.probabilities[k]

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_EXP_LEAF
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.leaf_n = n
            self._append_tape(entry, P, Q)

        return total

    cdef double _exp_sum_sum(
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
        cdef double total = 0.0
        cdef double theta_i
        cdef double phi_j
        cdef double v_ij
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

        for i in range(n):
            theta_i = P.parameter_at(i)
            for j in range(m):
                phi_j = Q.parameter_at(j)
                v_ij = value_rows[i][j]
                total += theta_i * phi_j * v_ij

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_EXP_SUM_SUM
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.ss_n = n
            entry.ss_m = m
            entry.ss_V.resize(n * m)
            entry.exp_theta.resize(n)
            entry.exp_phi.resize(m)
            for i in range(n):
                entry.exp_theta[i] = P.parameter_at(i)
                for j in range(m):
                    entry.ss_V[i * m + j] = value_rows[i][j]
            for j in range(m):
                entry.exp_phi[j] = Q.parameter_at(j)
            entry.ss_child_pair_indices = child_indices
            self._append_tape(entry, P, Q)

        return total

    cdef double _exp_prod_prod(
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
        cdef double total = 1.0
        cdef double child_val
        cdef GCWTapeEntry entry
        cdef vector[size_t] child_indices
        cdef size_t child_idx
        cdef object scope_to_q
        cdef object p_key
        cdef object q_key
        cdef Py_ssize_t q_idx
        cdef vector[double] child_vals
        cdef vector[int] row_ind_vec
        cdef vector[int] col_ind_vec
        cdef Py_ssize_t k

        if n != m:
            raise ValueError(
                f"expectation incompatible: product nodes have different numbers of "
                f"children ({n} vs {m})"
            )

        scope_to_q = {}
        for j in range(m):
            q_child = Q.child_at(j)
            q_key = _scope_frozen(q_child)
            if q_key in scope_to_q:
                raise ValueError(
                    "expectation incompatible: duplicate child scope among Q product children"
                )
            scope_to_q[q_key] = j

        child_vals.resize(n)
        row_ind_vec.resize(n)
        col_ind_vec.resize(n)
        if self.recording:
            child_indices.resize(n * m)

        for i in range(n):
            p_child = P.child_at(i)
            p_key = _scope_frozen(p_child)
            if p_key not in scope_to_q:
                raise ValueError(
                    "expectation incompatible: no Q product child with scope matching "
                    f"P child at index {i}"
                )
            q_idx = scope_to_q[p_key]
            del scope_to_q[p_key]
            q_child = Q.child_at(<size_t>q_idx)
            child_val = self.couple_value(p_child, q_child, side_P, side_Q)
            child_vals[i] = child_val
            total *= child_val
            row_ind_vec[i] = <int>i
            col_ind_vec[i] = <int>q_idx
            if self.recording:
                child_idx = self._lookup_pair_tape_idx(p_child, q_child)
                child_indices[i * m + <size_t>q_idx] = child_idx

        if len(scope_to_q) > 0:
            raise ValueError(
                "expectation incompatible: Q product children with unmatched scopes"
            )

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_EXP_PROD_PROD
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.pp_n = n
            entry.pp_m = m
            entry.exp_child_vals = child_vals
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
            if entry.kind == TAPE_EXP_LEAF:
                self._backward_exp_leaf(entry, g)
            elif entry.kind == TAPE_EXP_SUM_SUM:
                self._backward_exp_sum_sum(entry, g)
            elif entry.kind == TAPE_EXP_PROD_PROD:
                self._backward_exp_prod_prod(entry, g)

    cdef void _backward_exp_leaf(self, GCWTapeEntry entry, double g) except *:
        cdef CategoricalInputNode P = <CategoricalInputNode>entry.P
        cdef CategoricalInputNode Q = <CategoricalInputNode>entry.Q
        cdef size_t n = entry.leaf_n
        cdef size_t k
        cdef object grads_arr

        grads_arr = self._cat_grad_arr(0, P, n)
        for k in range(n):
            grads_arr[k] += g * Q.probabilities[k]

        grads_arr = self._cat_grad_arr(1, Q, n)
        for k in range(n):
            grads_arr[k] += g * P.probabilities[k]

    cdef void _backward_exp_sum_sum(self, GCWTapeEntry entry, double g) except *:
        cdef size_t n = entry.ss_n
        cdef size_t m = entry.ss_m
        cdef size_t i
        cdef size_t j
        cdef size_t child_idx
        cdef double theta_i
        cdef double phi_j
        cdef double v_ij
        cdef double sum_phi_v
        cdef double sum_theta_v
        cdef object grads_arr

        for i in range(n):
            theta_i = entry.exp_theta[i]
            for j in range(m):
                phi_j = entry.exp_phi[j]
                child_idx = entry.ss_child_pair_indices[i * m + j]
                if child_idx != NO_TAPE_IDX:
                    self.tape_adjoints[child_idx] += g * theta_i * phi_j

        grads_arr = self._sum_grad_arr(0, entry.P, n)
        for i in range(n):
            sum_phi_v = 0.0
            for j in range(m):
                sum_phi_v += entry.exp_phi[j] * entry.ss_V[i * m + j]
            grads_arr[i] += g * sum_phi_v

        grads_arr = self._sum_grad_arr(1, entry.Q, m)
        for j in range(m):
            sum_theta_v = 0.0
            for i in range(n):
                sum_theta_v += entry.exp_theta[i] * entry.ss_V[i * m + j]
            grads_arr[j] += g * sum_theta_v

    cdef void _backward_exp_prod_prod(self, GCWTapeEntry entry, double g) except *:
        cdef size_t n = entry.pp_n
        cdef size_t m = entry.pp_m
        cdef size_t k_idx
        cdef size_t child_idx
        cdef int r
        cdef int c
        cdef size_t num_matches = entry.pp_row_ind.size()
        cdef vector[double] prefix
        cdef vector[double] suffix
        cdef double factor
        cdef size_t i

        prefix.resize(n)
        suffix.resize(n)
        if n == 0:
            return

        prefix[0] = 1.0
        for i in range(1, n):
            prefix[i] = prefix[i - 1] * entry.exp_child_vals[i - 1]

        suffix[n - 1] = 1.0
        for i in range(<ssize_t>n - 1, 0, -1):
            suffix[<size_t>i - 1] = suffix[<size_t>i] * entry.exp_child_vals[<size_t>i]

        for k_idx in range(num_matches):
            r = entry.pp_row_ind[k_idx]
            factor = g * prefix[<size_t>r] * suffix[<size_t>r]
            child_idx = entry.pp_child_pair_indices[<size_t>r * m + <size_t>entry.pp_col_ind[k_idx]]
            if child_idx != NO_TAPE_IDX:
                self.tape_adjoints[child_idx] += factor

    cdef void _reset(self):
        self.couple_memo.clear()
        self.tape = []
        self.tape_adjoints.clear()
        self.pair_to_tape.clear()
        self.c1_sum_grads = {}
        self.c1_cat_grads = {}
        self.c2_sum_grads = {}
        self.c2_cat_grads = {}

    cdef double solve(self, CircuitNode circuit1, CircuitNode circuit2) except *:
        self._reset()
        return self.couple_value(circuit1, circuit2, 0, 1)

    cdef tuple solve_with_grad(self, CircuitNode circuit1, CircuitNode circuit2):
        cdef double value
        cdef size_t root_idx
        cdef GCWGradients grads1
        cdef GCWGradients grads2
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

        grads1 = GCWGradients()
        grads1.value = value
        grads1.sum_grads = self.c1_sum_grads
        grads1.cat_grads = self.c1_cat_grads

        grads2 = GCWGradients()
        grads2.value = value
        grads2.sum_grads = self.c2_sum_grads
        grads2.cat_grads = self.c2_cat_grads

        return (value, grads1, grads2)

cpdef double exp_query(
    object circuit1,
    object circuit2,
) except *:
    """Compute E_Q[P(X)] = sum_x P(x) Q(x) for compatible structured-decomposable PCs.

    Parameters
    ----------
    circuit1, circuit2 : CircuitNode or Circuit
        Compatible circuit roots with propagated scope (P and Q respectively).
    """
    cdef CircuitNode root1 = _unwrap_root(circuit1)
    cdef CircuitNode root2 = _unwrap_root(circuit2)
    cdef ExpectationContext ctx = ExpectationContext()
    return ctx.solve(root1, root2)


cpdef tuple exp_query_and_grad(
    object circuit1,
    object circuit2,
):
    """Compute E_Q[P(X)] and exact gradients w.r.t. both circuits.

    Returns ``(value, grads1, grads2)`` where ``value`` matches ``exp_query``
    and ``grads1`` / ``grads2`` are :class:`GCWGradients` bundles for
    ``circuit1`` and ``circuit2`` respectively (sum and categorical entries
    keyed by ``node.id``).
    """
    cdef CircuitNode root1 = _unwrap_root(circuit1)
    cdef CircuitNode root2 = _unwrap_root(circuit2)
    cdef ExpectationContext ctx = ExpectationContext()
    return ctx.solve_with_grad(root1, root2)


cpdef double log_exp_query(
    object circuit1,
    object circuit2,
) except *:
    """Compute log(E_Q[P(X)]) for compatible structured-decomposable PCs.

    Parameters
    ----------
    circuit1, circuit2 : CircuitNode or Circuit
        Compatible circuit roots with propagated scope (P and Q respectively).
    """
    cdef CircuitNode root1 = _unwrap_root(circuit1)
    cdef CircuitNode root2 = _unwrap_root(circuit2)
    cdef LogExpectationContext ctx = LogExpectationContext()
    return ctx.solve(root1, root2)


cpdef tuple log_exp_query_and_grad(
    object circuit1,
    object circuit2,
):
    """Compute log(E_Q[P(X)]) and exact log-objective gradients w.r.t. both circuits.

    Returns ``(log_value, grads1, grads2)`` where ``log_value`` matches
    ``log_exp_query`` and ``grads1`` / ``grads2`` are :class:`GCWGradients`
    bundles for ``circuit1`` and ``circuit2`` respectively (sum and categorical
    entries keyed by ``node.id``). Gradients are w.r.t. the linear parameters
    of the log-objective.
    """
    cdef CircuitNode root1 = _unwrap_root(circuit1)
    cdef CircuitNode root2 = _unwrap_root(circuit2)
    cdef LogExpectationContext ctx = LogExpectationContext()
    return ctx.solve_with_grad(root1, root2)
