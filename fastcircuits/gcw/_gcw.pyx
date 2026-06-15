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
    TAPE_LEAF,
    TAPE_MAX_SUM_PROD,
    TAPE_PROD_OTHER,
    TAPE_PROD_PROD,
    TAPE_SUM_OTHER,
    TAPE_SUM_SUM,
    dist_at,
    fill_pairwise_distance,
    node_kind,
    node_py_id,
    nw_backward_marginals,
    nw_run,
    solve_transport_lp,
    solve_transport_lp_with_duals,
    _unwrap_root,
)

cdef class GCWContext:
    cdef unordered_map[size_t, unordered_map[size_t, double]] couple_memo
    cdef unordered_map[size_t, double] d_1
    cdef unordered_map[size_t, double] d_2
    cdef double metric_p_1
    cdef double scale_1
    cdef double metric_p_2
    cdef double scale_2
    cdef object gurobi_env
    cdef unordered_map[uint64_t, vector[double]] dist_cache_1
    cdef unordered_map[uint64_t, vector[double]] dist_cache_2

    # Differentiable-mode state. ``recording == False`` keeps the original
    # forward path with no additional bookkeeping.
    cdef bint recording
    cdef list tape
    cdef vector[double] tape_adjoints
    cdef unordered_map[size_t, unordered_map[size_t, size_t]] pair_to_tape
    cdef unordered_map[size_t, double] ed_adj_2
    cdef list d_2_order
    cdef dict sum_grads
    cdef dict cat_grads

    def __cinit__(self):
        self.recording = False
        self.tape = []
        self.d_2_order = []
        self.sum_grads = {}
        self.cat_grads = {}

    cdef uint64_t dist_cache_key(self, int scope_var, size_t n_bins, int which) noexcept nogil:
        cdef uint64_t key = (<uint64_t>scope_var) << 32
        key |= (<uint64_t>n_bins) << 1
        key |= <uint64_t>which
        return key

    cdef vector[double]* get_distance_matrix(
        self,
        int scope_var,
        size_t n_bins,
        int which,
        double metric_p,
        double scale_factor,
    ) except *:
        cdef uint64_t key = self.dist_cache_key(scope_var, n_bins, which)
        if which == 0:
            if self.dist_cache_1.find(key) == self.dist_cache_1.end():
                fill_pairwise_distance(
                    self.dist_cache_1[key], n_bins, metric_p, scale_factor
                )
            return &self.dist_cache_1[key]
        if self.dist_cache_2.find(key) == self.dist_cache_2.end():
            fill_pairwise_distance(
                self.dist_cache_2[key], n_bins, metric_p, scale_factor
            )
        return &self.dist_cache_2[key]

    cdef double d_lookup_side(self, int side, CircuitNode node) except *:
        cdef size_t nid = node_py_id(node)
        cdef unordered_map[size_t, double].iterator it
        if side == 0:
            it = self.d_1.find(nid)
            if it == self.d_1.end():
                raise KeyError(
                    f"expected distance missing for node (py_id={nid}, "
                    f"node.id={node.id}) on side 0"
                )
        else:
            it = self.d_2.find(nid)
            if it == self.d_2.end():
                raise KeyError(
                    f"expected distance missing for node (py_id={nid}, "
                    f"node.id={node.id}) on side 1"
                )
        return deref(it).second

    cdef double metric_p_for_side(self, int side) noexcept nogil:
        if side == 0:
            return self.metric_p_1
        return self.metric_p_2

    cdef double scale_for_side(self, int side) noexcept nogil:
        if side == 0:
            return self.scale_1
        return self.scale_2

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

    # --- Expected distance (with optional topological order capture) ---

    cdef double _ed_node(
        self,
        CircuitNode node,
        int side,
        unordered_map[size_t, double]& cache,
        list order,
    ) except *:
        cdef size_t nid = node_py_id(node)
        cdef unordered_map[size_t, double].iterator it = cache.find(nid)
        if it != cache.end():
            return deref(it).second

        cdef double value
        cdef double total
        cdef size_t i
        cdef size_t nc
        cdef size_t a
        cdef size_t b
        cdef size_t n_outcomes
        cdef double pa
        cdef double pb
        cdef CircuitNode child
        cdef vector[double]* d_x_ptr
        cdef CategoricalInputNode cat_node
        cdef SumNode sum_node
        cdef ProductNode prod_node

        if isinstance(node, CategoricalInputNode):
            cat_node = <CategoricalInputNode>node
            n_outcomes = cat_node.probabilities.size()
            d_x_ptr = self.get_distance_matrix(
                cat_node.scope_var_c(),
                n_outcomes,
                side,
                self.metric_p_for_side(side),
                self.scale_for_side(side),
            )
            total = 0.0
            for a in range(n_outcomes):
                pa = cat_node.probabilities[a]
                for b in range(n_outcomes):
                    pb = cat_node.probabilities[b]
                    total += pa * deref(d_x_ptr)[a * n_outcomes + b] * pb
            value = total
        elif isinstance(node, SumNode):
            sum_node = <SumNode>node
            total = 0.0
            nc = sum_node.num_children()
            for i in range(nc):
                child = sum_node.child_at(i)
                total += sum_node.parameter_at(i) * self._ed_node(
                    child, side, cache, order
                )
            value = total
        elif isinstance(node, ProductNode):
            prod_node = <ProductNode>node
            total = 0.0
            nc = prod_node.num_children()
            for i in range(nc):
                child = prod_node.child_at(i)
                total += self._ed_node(child, side, cache, order)
            value = total
        else:
            raise TypeError(f"unsupported node type: {type(node)}")

        cache[nid] = value
        if order is not None:
            order.append(node)
        return value

    # --- Coupling forward (records tape entries when ``self.recording``) ---

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
        cdef ProductNode P_prod
        cdef ProductNode Q_prod

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
            res = self._compute_leaf(
                <CategoricalInputNode>P, <CategoricalInputNode>Q, side_P, side_Q
            )
        elif pk == SUM and qk == SUM:
            res = self._compute_sum_sum(<SumNode>P, <SumNode>Q, side_P, side_Q)
        elif pk == PRODUCT and qk == PRODUCT:
            P_prod = <ProductNode>P
            Q_prod = <ProductNode>Q
            if P_prod.num_children() < Q_prod.num_children():
                res = self._compute_prod_prod(Q_prod, P_prod, side_Q, side_P)
            else:
                res = self._compute_prod_prod(P_prod, Q_prod, side_P, side_Q)
        elif pk == PRODUCT and qk == SUM:
            res = self._compute_max_sum_prod(
                <SumNode>Q, <ProductNode>P, side_Q, side_P
            )
        elif pk == SUM and qk == PRODUCT:
            res = self._compute_max_sum_prod(
                <SumNode>P, <ProductNode>Q, side_P, side_Q
            )
        elif pk == SUM and qk == CATEGORICAL:
            res = self._compute_sum_other(<SumNode>P, Q, side_P, side_Q)
        elif pk == CATEGORICAL and qk == SUM:
            res = self._compute_sum_other(<SumNode>Q, P, side_Q, side_P)
        elif pk == CATEGORICAL and qk == PRODUCT:
            res = self._compute_prod_other(<ProductNode>Q, P, side_Q, side_P)
        elif pk == PRODUCT and qk == CATEGORICAL:
            res = self._compute_prod_other(<ProductNode>P, Q, side_P, side_Q)
        else:
            raise NotImplementedError("Coupling not implemented")

        self.couple_memo[key_a][key_b] = res
        return res

    cdef double _compute_leaf(
        self,
        CategoricalInputNode P,
        CategoricalInputNode Q,
        int side_P,
        int side_Q,
    ) except *:
        cdef size_t n = P.num_outcomes()
        cdef size_t m = Q.num_outcomes()
        cdef int p_scope = P.scope_var_c()
        cdef int q_scope = Q.scope_var_c()
        cdef vector[double]* d_p_ptr = self.get_distance_matrix(
            p_scope, n, side_P,
            self.metric_p_for_side(side_P), self.scale_for_side(side_P),
        )
        cdef vector[double]* d_q_ptr = self.get_distance_matrix(
            q_scope, m, side_Q,
            self.metric_p_for_side(side_Q), self.scale_for_side(side_Q),
        )
        cdef GCWTapeEntry entry
        cdef double value
        cdef vector[int] rows_tmp
        cdef vector[int] cols_tmp
        cdef vector[double] vals_tmp
        cdef vector[int] modes_tmp

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_LEAF
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.leaf_n = n
            entry.leaf_m = m
            entry.leaf_p_scope = p_scope
            entry.leaf_q_scope = q_scope
            value = nw_run(
                P.probabilities, Q.probabilities,
                deref(d_p_ptr), deref(d_q_ptr), n, m,
                entry.leaf_rows, entry.leaf_cols, entry.leaf_vals, entry.leaf_modes,
            )
            self._append_tape(entry, P, Q)
            return value

        return nw_run(
            P.probabilities, Q.probabilities,
            deref(d_p_ptr), deref(d_q_ptr), n, m,
            rows_tmp, cols_tmp, vals_tmp, modes_tmp,
        )

    cdef double _compute_sum_sum(
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
                C_np[i, j] = -value_rows[i][j]

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
            entry.kind = TAPE_SUM_SUM
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

    cdef double _compute_prod_prod(
        self,
        ProductNode P,
        ProductNode Q,
        int side_P,
        int side_Q,
    ) except *:
        from scipy.optimize import linear_sum_assignment

        cdef size_t n = P.num_children()
        cdef size_t m = Q.num_children()
        cdef size_t i
        cdef size_t j
        cdef CircuitNode p_child
        cdef CircuitNode q_child
        cdef double c_cost
        cdef double ed_val
        cdef double base_cost = 0.0
        cdef double gw_cost
        cdef object pairwise_costs_np
        cdef object row_ind
        cdef object col_ind
        cdef Py_ssize_t k
        cdef GCWTapeEntry entry
        cdef vector[size_t] child_indices
        cdef size_t child_idx
        cdef list p_children_list = None
        cdef list q_children_list = None
        cdef Py_ssize_t num_matches

        pairwise_costs_np = np.empty((n, m), dtype=np.float64)
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
            for j in range(m):
                q_child = Q.child_at(j)
                c_cost = self.couple_value(p_child, q_child, side_P, side_Q)
                ed_val = self.d_lookup_side(side_P, p_child) * self.d_lookup_side(
                    side_Q, q_child
                )
                pairwise_costs_np[i, j] = c_cost - ed_val
                base_cost += ed_val
                if self.recording:
                    child_idx = self._lookup_pair_tape_idx(p_child, q_child)
                    child_indices[i * m + j] = child_idx

        row_ind, col_ind = linear_sum_assignment(-pairwise_costs_np)
        gw_cost = base_cost
        num_matches = len(row_ind)
        for k in range(num_matches):
            gw_cost += pairwise_costs_np[row_ind[k], col_ind[k]]

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_PROD_PROD
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.pp_n = n
            entry.pp_m = m
            entry.pp_d_p.resize(n)
            entry.pp_d_q.resize(m)
            for i in range(n):
                entry.pp_d_p[i] = self.d_lookup_side(side_P, P.child_at(i))
            for j in range(m):
                entry.pp_d_q[j] = self.d_lookup_side(side_Q, Q.child_at(j))
            entry.pp_p_children = p_children_list
            entry.pp_q_children = q_children_list
            entry.pp_row_ind.resize(<size_t>num_matches)
            entry.pp_col_ind.resize(<size_t>num_matches)
            for k in range(num_matches):
                entry.pp_row_ind[<size_t>k] = int(row_ind[k])
                entry.pp_col_ind[<size_t>k] = int(col_ind[k])
            entry.pp_child_pair_indices = child_indices
            self._append_tape(entry, P, Q)

        return gw_cost

    cdef double _compute_sum_other(
        self,
        SumNode P,
        CircuitNode Q,
        int side_P,
        int side_Q,
    ) except *:
        cdef size_t nc = P.num_children()
        cdef size_t i
        cdef CircuitNode p_child
        cdef double total = 0.0
        cdef double theta_i
        cdef double v_i
        cdef GCWTapeEntry entry
        cdef vector[double] V_vec
        cdef vector[double] theta_vec
        cdef vector[size_t] child_indices

        if self.recording:
            V_vec.resize(nc)
            theta_vec.resize(nc)
            child_indices.resize(nc)

        for i in range(nc):
            p_child = P.child_at(i)
            v_i = self.couple_value(p_child, Q, side_P, side_Q)
            theta_i = P.parameter_at(i)
            total += theta_i * v_i
            if self.recording:
                V_vec[i] = v_i
                theta_vec[i] = theta_i
                child_indices[i] = self._lookup_pair_tape_idx(p_child, Q)

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_SUM_OTHER
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.so_nc = nc
            entry.so_theta = theta_vec
            entry.so_V = V_vec
            entry.so_child_pair_indices = child_indices
            self._append_tape(entry, P, Q)

        return total

    cdef double _compute_prod_other(
        self,
        ProductNode P,
        CircuitNode Q,
        int side_P,
        int side_Q,
    ) except *:
        cdef size_t nc = P.num_children()
        cdef size_t i
        cdef size_t best_idx = 0
        cdef CircuitNode p_child
        cdef vector[double] V_vec
        cdef vector[double] d_p_vec
        cdef double d_q = self.d_lookup_side(side_Q, Q)
        cdef double total_cost
        cdef double best_val = -1e300
        cdef double adjusted
        cdef GCWTapeEntry entry
        cdef vector[size_t] child_indices

        V_vec.resize(nc)
        d_p_vec.resize(nc)
        if self.recording:
            child_indices.resize(nc)

        for i in range(nc):
            p_child = P.child_at(i)
            V_vec[i] = self.couple_value(p_child, Q, side_P, side_Q)
            d_p_vec[i] = self.d_lookup_side(side_P, p_child)
            if self.recording:
                child_indices[i] = self._lookup_pair_tape_idx(p_child, Q)

        for i in range(nc):
            adjusted = V_vec[i] - d_p_vec[i] * d_q
            if adjusted > best_val:
                best_val = adjusted
                best_idx = i

        total_cost = V_vec[best_idx]
        for i in range(nc):
            if i != best_idx:
                total_cost += d_p_vec[i] * d_q

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_PROD_OTHER
            entry.side_P = side_P
            entry.side_Q = side_Q
            entry.P = P
            entry.Q = Q
            entry.po_nc = nc
            entry.po_V = V_vec
            entry.po_d_p = d_p_vec
            entry.po_d_q = d_q
            entry.po_best_idx = best_idx
            entry.po_child_pair_indices = child_indices
            self._append_tape(entry, P, Q)

        return total_cost

    cdef double _compute_max_sum_prod(
        self,
        SumNode P_sum,
        ProductNode Q_prod,
        int sP_sum,
        int sQ_prod,
    ) except *:
        """Coupling between a SumNode and a ProductNode.

        Forward computes both branches and returns ``max(res_sum_other,
        res_prod_other)``. The tape entry stores both branches' state and a
        ``max_winner`` flag; backward routes the upstream adjoint into the
        winning branch only (active-set subgradient at the max).
        """
        cdef size_t nc_sum = P_sum.num_children()
        cdef size_t nc_prod = Q_prod.num_children()
        cdef size_t i
        cdef size_t best_idx = 0
        cdef CircuitNode sum_child
        cdef CircuitNode prod_child
        cdef vector[double] so_V
        cdef vector[double] so_theta
        cdef vector[size_t] so_indices
        cdef vector[double] po_V
        cdef vector[double] po_d_p
        cdef vector[size_t] po_indices
        cdef double d_q_sum
        cdef double theta_i
        cdef double v_i
        cdef double adjusted
        cdef double best_val = -1e300
        cdef double res1 = 0.0
        cdef double res2
        cdef GCWTapeEntry entry
        cdef int max_winner

        so_V.resize(nc_sum)
        so_theta.resize(nc_sum)
        if self.recording:
            so_indices.resize(nc_sum)
        for i in range(nc_sum):
            sum_child = P_sum.child_at(i)
            v_i = self.couple_value(sum_child, Q_prod, sP_sum, sQ_prod)
            theta_i = P_sum.parameter_at(i)
            so_V[i] = v_i
            so_theta[i] = theta_i
            res1 += theta_i * v_i
            if self.recording:
                so_indices[i] = self._lookup_pair_tape_idx(sum_child, Q_prod)

        po_V.resize(nc_prod)
        po_d_p.resize(nc_prod)
        if self.recording:
            po_indices.resize(nc_prod)
        d_q_sum = self.d_lookup_side(sP_sum, P_sum)
        for i in range(nc_prod):
            prod_child = Q_prod.child_at(i)
            po_V[i] = self.couple_value(prod_child, P_sum, sQ_prod, sP_sum)
            po_d_p[i] = self.d_lookup_side(sQ_prod, prod_child)
            if self.recording:
                po_indices[i] = self._lookup_pair_tape_idx(prod_child, P_sum)

        for i in range(nc_prod):
            adjusted = po_V[i] - po_d_p[i] * d_q_sum
            if adjusted > best_val:
                best_val = adjusted
                best_idx = i

        res2 = po_V[best_idx]
        for i in range(nc_prod):
            if i != best_idx:
                res2 += po_d_p[i] * d_q_sum

        if res1 >= res2:
            max_winner = 0
        else:
            max_winner = 1

        if self.recording:
            entry = GCWTapeEntry()
            entry.kind = TAPE_MAX_SUM_PROD
            entry.P = P_sum
            entry.Q = Q_prod
            entry.side_P = sP_sum
            entry.side_Q = sQ_prod
            entry.max_winner = max_winner
            entry.max_sum_node = P_sum
            entry.max_prod_node = Q_prod
            entry.max_sum_side = sP_sum
            entry.max_prod_side = sQ_prod
            entry.max_so_nc = nc_sum
            entry.max_so_theta = so_theta
            entry.max_so_V = so_V
            entry.max_so_child_pair_indices = so_indices
            entry.max_po_nc = nc_prod
            entry.max_po_V = po_V
            entry.max_po_d_p = po_d_p
            entry.max_po_d_q = d_q_sum
            entry.max_po_best_idx = best_idx
            entry.max_po_child_pair_indices = po_indices
            self._append_tape(entry, P_sum, Q_prod)

        if res1 >= res2:
            return res1
        return res2

    # --- Backward pass ---

    cdef void _run_backward(self) except *:
        cdef ssize_t k
        cdef double g
        cdef GCWTapeEntry entry
        for k in range(<ssize_t>len(self.tape) - 1, -1, -1):
            g = self.tape_adjoints[<size_t>k]
            if g == 0.0:
                continue
            entry = <GCWTapeEntry>self.tape[k]
            if entry.kind == TAPE_LEAF:
                self._backward_leaf(entry, g)
            elif entry.kind == TAPE_SUM_SUM:
                self._backward_sum_sum(entry, g)
            elif entry.kind == TAPE_PROD_PROD:
                self._backward_prod_prod(entry, g)
            elif entry.kind == TAPE_SUM_OTHER:
                self._backward_sum_other(entry, g)
            elif entry.kind == TAPE_PROD_OTHER:
                self._backward_prod_other(entry, g)
            elif entry.kind == TAPE_MAX_SUM_PROD:
                self._backward_max_sum_prod(entry, g)

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

    cdef void _backward_leaf(self, GCWTapeEntry entry, double g) except *:
        """V = sum_{a, b} w_a w_b d_p[i_a, i_b] d_q[j_a, j_b]; sym. d => G_a = 2 sum_b w_b d_p d_q."""
        cdef size_t n = entry.leaf_n
        cdef size_t m = entry.leaf_m
        cdef size_t num_steps = entry.leaf_rows.size()
        cdef vector[double]* d_p_ptr = self.get_distance_matrix(
            entry.leaf_p_scope, n, entry.side_P,
            self.metric_p_for_side(entry.side_P), self.scale_for_side(entry.side_P),
        )
        cdef vector[double]* d_q_ptr = self.get_distance_matrix(
            entry.leaf_q_scope, m, entry.side_Q,
            self.metric_p_for_side(entry.side_Q), self.scale_for_side(entry.side_Q),
        )
        cdef vector[double] G
        cdef vector[double] adj_p
        cdef vector[double] adj_q
        cdef size_t a
        cdef size_t b
        cdef int i_a
        cdef int j_a
        cdef int i_b
        cdef int j_b
        cdef double sum_b
        cdef size_t k
        cdef object grads_arr

        G.resize(num_steps)
        for a in range(num_steps):
            i_a = entry.leaf_rows[a]
            j_a = entry.leaf_cols[a]
            sum_b = 0.0
            for b in range(num_steps):
                i_b = entry.leaf_rows[b]
                j_b = entry.leaf_cols[b]
                sum_b += entry.leaf_vals[b] * dist_at(deref(d_p_ptr), n, i_a, i_b) * dist_at(deref(d_q_ptr), m, j_a, j_b)
            G[a] = 2.0 * sum_b * g

        nw_backward_marginals(
            entry.leaf_rows, entry.leaf_cols, entry.leaf_modes,
            G, n, m, adj_p, adj_q,
        )

        if entry.side_P == 1:
            grads_arr = self._cat_grad_arr(entry.P, n)
            for k in range(n):
                grads_arr[k] += adj_p[k]
        if entry.side_Q == 1:
            grads_arr = self._cat_grad_arr(entry.Q, m)
            for k in range(m):
                grads_arr[k] += adj_q[k]

    cdef void _backward_sum_sum(self, GCWTapeEntry entry, double g) except *:
        """value = sum V_ij w*_ij. dV/dV_ij = w*_ij, dV/dtheta = -pi, dV/dphi = -rho."""
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
                grads_arr[i] += -g * entry.ss_pi[i]

        if entry.side_Q == 1:
            grads_arr = self._sum_grad_arr(entry.Q, m)
            for j in range(m):
                grads_arr[j] += -g * entry.ss_rho[j]

    cdef void _backward_prod_prod(self, GCWTapeEntry entry, double g) except *:
        """value = (sum d_p)(sum d_q) + sum_{(i,j) in sigma} [couple - d_p*d_q]."""
        cdef size_t n = entry.pp_n
        cdef size_t m = entry.pp_m
        cdef size_t i
        cdef size_t j
        cdef size_t k_idx
        cdef size_t child_idx
        cdef CircuitNode p_child
        cdef CircuitNode q_child
        cdef double sum_d_p = 0.0
        cdef double sum_d_q = 0.0
        cdef double d_q_match
        cdef double d_p_match
        cdef size_t num_matches = entry.pp_row_ind.size()
        cdef vector[int] matched_q_for_p
        cdef vector[int] matched_p_for_q
        cdef int r
        cdef int c

        matched_q_for_p.assign(n, -1)
        matched_p_for_q.assign(m, -1)
        for k_idx in range(num_matches):
            r = entry.pp_row_ind[k_idx]
            c = entry.pp_col_ind[k_idx]
            matched_q_for_p[r] = c
            matched_p_for_q[c] = r

            child_idx = entry.pp_child_pair_indices[<size_t>r * m + <size_t>c]
            if child_idx != NO_TAPE_IDX:
                self.tape_adjoints[child_idx] += g

        for j in range(m):
            sum_d_q += entry.pp_d_q[j]
        for i in range(n):
            sum_d_p += entry.pp_d_p[i]

        if entry.side_P == 1:
            for i in range(n):
                p_child = <CircuitNode>entry.pp_p_children[i]
                if matched_q_for_p[i] >= 0:
                    d_q_match = entry.pp_d_q[<size_t>matched_q_for_p[i]]
                    self.ed_adj_2[node_py_id(p_child)] += g * (sum_d_q - d_q_match)
                else:
                    self.ed_adj_2[node_py_id(p_child)] += g * sum_d_q

        if entry.side_Q == 1:
            for j in range(m):
                q_child = <CircuitNode>entry.pp_q_children[j]
                if matched_p_for_q[j] >= 0:
                    d_p_match = entry.pp_d_p[<size_t>matched_p_for_q[j]]
                    self.ed_adj_2[node_py_id(q_child)] += g * (sum_d_p - d_p_match)
                else:
                    self.ed_adj_2[node_py_id(q_child)] += g * sum_d_p

    cdef void _backward_sum_other_state(
        self,
        vector[double]& theta,
        vector[double]& V,
        vector[size_t]& child_indices,
        size_t nc,
        CircuitNode P_sum,
        int side_P_sum,
        double g,
    ) except *:
        cdef size_t i
        cdef size_t child_idx
        cdef object grads_arr

        for i in range(nc):
            child_idx = child_indices[i]
            if child_idx != NO_TAPE_IDX:
                self.tape_adjoints[child_idx] += g * theta[i]

        if side_P_sum == 1:
            grads_arr = self._sum_grad_arr(P_sum, nc)
            for i in range(nc):
                grads_arr[i] += g * V[i]

    cdef void _backward_sum_other(self, GCWTapeEntry entry, double g) except *:
        self._backward_sum_other_state(
            entry.so_theta, entry.so_V, entry.so_child_pair_indices,
            entry.so_nc, entry.P, entry.side_P, g,
        )

    cdef void _backward_prod_other_state(
        self,
        vector[double]& V,
        vector[double]& d_p,
        double d_q,
        size_t best_idx,
        vector[size_t]& child_indices,
        size_t nc,
        ProductNode P_prod,
        CircuitNode Q_other,
        int side_P_prod,
        int side_Q_other,
        double g,
    ) except *:
        """value = V[best] + sum_{i != best} d_p[i] * d_q."""
        cdef size_t i
        cdef size_t child_idx
        cdef CircuitNode p_child
        cdef double sum_others_d_p = 0.0

        child_idx = child_indices[best_idx]
        if child_idx != NO_TAPE_IDX:
            self.tape_adjoints[child_idx] += g

        for i in range(nc):
            if i != best_idx:
                sum_others_d_p += d_p[i]

        if side_P_prod == 1:
            for i in range(nc):
                if i != best_idx:
                    p_child = P_prod.child_at(i)
                    self.ed_adj_2[node_py_id(p_child)] += g * d_q

        if side_Q_other == 1:
            self.ed_adj_2[node_py_id(Q_other)] += g * sum_others_d_p

    cdef void _backward_prod_other(self, GCWTapeEntry entry, double g) except *:
        self._backward_prod_other_state(
            entry.po_V, entry.po_d_p, entry.po_d_q, entry.po_best_idx,
            entry.po_child_pair_indices, entry.po_nc,
            <ProductNode>entry.P, entry.Q,
            entry.side_P, entry.side_Q, g,
        )

    cdef void _backward_max_sum_prod(self, GCWTapeEntry entry, double g) except *:
        if entry.max_winner == 0:
            # Sum-other branch wins: SumNode is "P", ProductNode is "Other";
            # effective sides are (max_sum_side, max_prod_side).
            self._backward_sum_other_state(
                entry.max_so_theta, entry.max_so_V,
                entry.max_so_child_pair_indices, entry.max_so_nc,
                entry.max_sum_node, entry.max_sum_side, g,
            )
        else:
            # Prod-other branch wins: ProductNode is "P", SumNode is "Other";
            # effective sides are (max_prod_side, max_sum_side).
            self._backward_prod_other_state(
                entry.max_po_V, entry.max_po_d_p, entry.max_po_d_q,
                entry.max_po_best_idx, entry.max_po_child_pair_indices,
                entry.max_po_nc, entry.max_prod_node, entry.max_sum_node,
                entry.max_prod_side, entry.max_sum_side, g,
            )

    # --- Expected-distance backward (top-down through circuit2) ---

    cdef void _ed_backward(self) except *:
        cdef ssize_t k
        cdef CircuitNode node
        cdef double adj
        cdef size_t nid
        cdef unordered_map[size_t, double].iterator it
        cdef size_t i
        cdef size_t nc
        cdef size_t n_outcomes
        cdef size_t kk
        cdef size_t j
        cdef CircuitNode child
        cdef object grads_arr
        cdef vector[double]* d_x_ptr
        cdef double accum
        cdef double child_E
        cdef double metric_p
        cdef double scale
        cdef CategoricalInputNode cat_node
        cdef SumNode sum_node
        cdef ProductNode prod_node
        cdef unordered_map[size_t, double].iterator d_it

        for k in range(<ssize_t>len(self.d_2_order) - 1, -1, -1):
            node = <CircuitNode>self.d_2_order[k]
            nid = node_py_id(node)
            it = self.ed_adj_2.find(nid)
            if it == self.ed_adj_2.end():
                continue
            adj = deref(it).second
            if adj == 0.0:
                continue

            if isinstance(node, CategoricalInputNode):
                cat_node = <CategoricalInputNode>node
                n_outcomes = cat_node.probabilities.size()
                metric_p = self.metric_p_for_side(1)
                scale = self.scale_for_side(1)
                d_x_ptr = self.get_distance_matrix(
                    cat_node.scope_var_c(), n_outcomes, 1, metric_p, scale,
                )
                grads_arr = self._cat_grad_arr(cat_node, n_outcomes)
                for kk in range(n_outcomes):
                    accum = 0.0
                    for j in range(n_outcomes):
                        accum += deref(d_x_ptr)[kk * n_outcomes + j] * cat_node.probabilities[j]
                    grads_arr[kk] += 2.0 * adj * accum
            elif isinstance(node, SumNode):
                sum_node = <SumNode>node
                nc = sum_node.num_children()
                grads_arr = self._sum_grad_arr(sum_node, nc)
                for i in range(nc):
                    child = sum_node.child_at(i)
                    d_it = self.d_2.find(node_py_id(child))
                    child_E = 0.0 if d_it == self.d_2.end() else deref(d_it).second
                    grads_arr[i] += adj * child_E
                    self.ed_adj_2[node_py_id(child)] = self.ed_adj_2[node_py_id(child)] + adj * sum_node.parameter_at(i)
            elif isinstance(node, ProductNode):
                prod_node = <ProductNode>node
                nc = prod_node.num_children()
                for i in range(nc):
                    child = prod_node.child_at(i)
                    self.ed_adj_2[node_py_id(child)] = self.ed_adj_2[node_py_id(child)] + adj

    # --- Public solve methods ---

    cdef void _reset(self):
        self.couple_memo.clear()
        self.d_1.clear()
        self.d_2.clear()
        self.dist_cache_1.clear()
        self.dist_cache_2.clear()
        self.tape = []
        self.tape_adjoints.clear()
        self.pair_to_tape.clear()
        self.ed_adj_2.clear()
        self.d_2_order = []
        self.sum_grads = {}
        self.cat_grads = {}

    cdef double solve(self, CircuitNode circuit1, CircuitNode circuit2) except *:
        self._reset()
        self._ed_node(circuit1, 0, self.d_1, None)
        self._ed_node(circuit2, 1, self.d_2, self.d_2_order)
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
            self._ed_backward()
        finally:
            self.recording = False

        grads = GCWGradients()
        grads.value = value
        grads.sum_grads = self.sum_grads
        grads.cat_grads = self.cat_grads
        return (value, grads)

cpdef double gcw_crossterm(
    object circuit1,
    object circuit2,
    double metric_p=1.0,
    double scale_factor_1=1.0,
    double scale_factor_2=1.0,
    object gurobi_env=None,
) except *:
    """Compute the GCW cross-term between two probabilistic circuits.

    Returns the structurally coupled Gromov-Wasserstein cross-term scalar only
    (not the full GCW distance and not a coupling circuit).

    Parameters
    ----------
    circuit1, circuit2 : CircuitNode or Circuit
        Circuit roots with propagated scope.
    metric_p : float
        Exponent for separable L_p distance on categorical indices.
    scale_factor_1, scale_factor_2 : float
        Scale divisors for circuit1 and circuit2 distance metrics.
    gurobi_env : gurobipy.Env
        Shared Gurobi environment (required for sum x sum transport subproblems).
    """
    if gurobi_env is None:
        raise ValueError("gurobi_env is required for GCW cross-term computation")

    cdef CircuitNode root1 = _unwrap_root(circuit1)
    cdef CircuitNode root2 = _unwrap_root(circuit2)
    cdef GCWContext ctx = GCWContext()
    ctx.metric_p_1 = metric_p
    ctx.scale_1 = scale_factor_1
    ctx.metric_p_2 = metric_p
    ctx.scale_2 = scale_factor_2
    ctx.gurobi_env = gurobi_env
    return ctx.solve(root1, root2)


cpdef tuple gcw_crossterm_and_grad(
    object circuit1,
    object circuit2,
    double metric_p=1.0,
    double scale_factor_1=1.0,
    double scale_factor_2=1.0,
    object gurobi_env=None,
):
    """Compute the GCW cross-term and its subgradients w.r.t. ``circuit2``.

    Returns ``(value, grads)`` where ``value`` matches ``gcw_crossterm`` to
    floating-point tolerance and ``grads`` is a :class:`GCWGradients` bundle
    containing per-node gradient vectors for every ``SumNode`` and
    ``CategoricalInputNode`` in ``circuit2``. Nodes that did not contribute to
    the crossterm receive no entry.

    The gradient is an exact subgradient: it is computed by fixing the
    discrete choices made during the forward solve (NW plan, sum-sum LP basis
    with constraint duals, product-product Hungarian assignment, argmax index,
    and max-of-two branch) and applying reverse-mode AD on the remaining
    smooth structure. At kinks (LP degeneracy, argmax ties, NW mass ties) any
    valid subgradient may be returned.
    """
    if gurobi_env is None:
        raise ValueError("gurobi_env is required for GCW cross-term computation")

    cdef CircuitNode root1 = _unwrap_root(circuit1)
    cdef CircuitNode root2 = _unwrap_root(circuit2)
    cdef GCWContext ctx = GCWContext()
    ctx.metric_p_1 = metric_p
    ctx.scale_1 = scale_factor_1
    ctx.metric_p_2 = metric_p
    ctx.scale_2 = scale_factor_2
    ctx.gurobi_env = gurobi_env
    return ctx.solve_with_grad(root1, root2)
