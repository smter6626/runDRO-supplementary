from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

from fastcircuits.nodes cimport CircuitNode

cdef double PROB_EPS
cdef size_t NO_TAPE_IDX

cdef enum NodeKind:
    SUM = 0
    PRODUCT = 1
    CATEGORICAL = 2

cdef enum TapeKind:
    TAPE_LEAF = 0
    TAPE_SUM_SUM = 1
    TAPE_PROD_PROD = 2
    TAPE_SUM_OTHER = 3
    TAPE_PROD_OTHER = 4
    TAPE_MAX_SUM_PROD = 5
    TAPE_CW_LEAF = 6
    TAPE_CW_SUM_SUM = 7
    TAPE_CW_PROD_PROD = 8
    TAPE_EXP_LEAF = 9
    TAPE_EXP_SUM_SUM = 10
    TAPE_EXP_PROD_PROD = 11
    TAPE_LOGEXP_LEAF = 12
    TAPE_LOGEXP_SUM_SUM = 13
    TAPE_LOGEXP_PROD_PROD = 14

cdef NodeKind node_kind(CircuitNode node) except *

cdef inline size_t node_py_id(CircuitNode node):
    return id(node)

cdef inline double dist_at(const vector[double]& mat, size_t n, size_t i, size_t j) noexcept nogil:
    return mat[i * n + j]

cdef inline double dist_at_cross(
    const vector[double]& mat, size_t m, size_t i, size_t j
) noexcept nogil:
    return mat[i * m + j]

cdef void fill_cross_distance(
    vector[double]& mat,
    size_t n,
    size_t m,
    double metric_p,
    double scale_factor,
) noexcept nogil

cdef void fill_pairwise_distance(
    vector[double]& mat,
    size_t n,
    double metric_p,
    double scale_factor,
) noexcept nogil

cdef size_t nw_plan(
    const vector[double]& p,
    const vector[double]& q,
    size_t n,
    size_t m,
    vector[int]& rows_out,
    vector[int]& cols_out,
    vector[double]& vals_out,
    vector[int]& modes_out,
) noexcept nogil

cdef double nw_run(
    const vector[double]& p,
    const vector[double]& q,
    const vector[double]& d_p,
    const vector[double]& d_q,
    size_t n,
    size_t m,
    vector[int]& rows_out,
    vector[int]& cols_out,
    vector[double]& vals_out,
    vector[int]& modes_out,
) noexcept nogil

cdef void nw_backward_marginals(
    const vector[int]& rows,
    const vector[int]& cols,
    const vector[int]& modes,
    const vector[double]& G,
    size_t n,
    size_t m,
    vector[double]& adj_p_out,
    vector[double]& adj_q_out,
) noexcept nogil

cdef tuple solve_transport_lp_with_duals(
    object cost,
    object theta,
    object phi,
    object gurobi_env,
)

cdef object solve_transport_lp(
    object cost,
    object theta,
    object phi,
    object gurobi_env,
)

cdef inline object _scope_frozen(CircuitNode node):
    return frozenset(node.scope_as_list())

cdef CircuitNode _unwrap_root(object root) except *

cdef class GCWTapeEntry:
    cdef int kind
    cdef int side_P
    cdef int side_Q
    cdef CircuitNode P
    cdef CircuitNode Q
    cdef vector[int] leaf_rows
    cdef vector[int] leaf_cols
    cdef vector[double] leaf_vals
    cdef vector[int] leaf_modes
    cdef size_t leaf_n
    cdef size_t leaf_m
    cdef int leaf_p_scope
    cdef int leaf_q_scope
    cdef vector[double] ss_w
    cdef vector[double] ss_pi
    cdef vector[double] ss_rho
    cdef vector[double] ss_V
    cdef size_t ss_n
    cdef size_t ss_m
    cdef vector[size_t] ss_child_pair_indices
    cdef vector[int] pp_row_ind
    cdef vector[int] pp_col_ind
    cdef vector[double] pp_d_p
    cdef vector[double] pp_d_q
    cdef list pp_p_children
    cdef list pp_q_children
    cdef vector[size_t] pp_child_pair_indices
    cdef size_t pp_n
    cdef size_t pp_m
    cdef vector[double] so_theta
    cdef vector[double] so_V
    cdef vector[size_t] so_child_pair_indices
    cdef size_t so_nc
    cdef vector[double] po_V
    cdef vector[double] po_d_p
    cdef double po_d_q
    cdef size_t po_best_idx
    cdef vector[size_t] po_child_pair_indices
    cdef size_t po_nc
    cdef int max_winner
    cdef CircuitNode max_sum_node
    cdef CircuitNode max_prod_node
    cdef int max_sum_side
    cdef int max_prod_side
    cdef vector[double] max_so_theta
    cdef vector[double] max_so_V
    cdef vector[size_t] max_so_child_pair_indices
    cdef size_t max_so_nc
    cdef vector[double] max_po_V
    cdef vector[double] max_po_d_p
    cdef double max_po_d_q
    cdef size_t max_po_best_idx
    cdef vector[size_t] max_po_child_pair_indices
    cdef size_t max_po_nc
    cdef vector[double] exp_theta
    cdef vector[double] exp_phi
    cdef vector[double] exp_child_vals
    cdef double logexp_ell

cdef class GCWGradients:
    cdef public double value
    cdef public dict sum_grads
    cdef public dict cat_grads
