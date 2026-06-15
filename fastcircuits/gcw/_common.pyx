# distutils: language = c++

from cython.operator cimport dereference as deref
from libc.math cimport fabs, pow
from libcpp.vector cimport vector

import numpy as np

from fastcircuits.nodes cimport (
    CategoricalInputNode,
    CircuitNode,
    ProductNode,
    SumNode,
)

cdef double PROB_EPS = 1e-8
cdef size_t NO_TAPE_IDX = <size_t>-1


cdef NodeKind node_kind(CircuitNode node) except *:
    if isinstance(node, CategoricalInputNode):
        return CATEGORICAL
    if isinstance(node, SumNode):
        return SUM
    if isinstance(node, ProductNode):
        return PRODUCT
    raise TypeError(f"unsupported node type: {type(node)}")


cdef void fill_cross_distance(
    vector[double]& mat,
    size_t n,
    size_t m,
    double metric_p,
    double scale_factor,
) noexcept nogil:
    cdef size_t i
    cdef size_t j
    cdef double xi
    cdef double xj
    mat.resize(n * m)
    for i in range(n):
        xi = <double>i
        for j in range(m):
            xj = <double>j
            mat[i * m + j] = pow(fabs(xi - xj), metric_p) / scale_factor


cdef void fill_pairwise_distance(
    vector[double]& mat,
    size_t n,
    double metric_p,
    double scale_factor,
) noexcept nogil:
    cdef size_t i
    cdef size_t j
    cdef double xi
    cdef double xj
    mat.resize(n * n)
    for i in range(n):
        xi = <double>i
        for j in range(n):
            xj = <double>j
            mat[i * n + j] = pow(fabs(xi - xj), metric_p) / scale_factor


cdef size_t nw_plan(
    const vector[double]& p,
    const vector[double]& q,
    size_t n,
    size_t m,
    vector[int]& rows_out,
    vector[int]& cols_out,
    vector[double]& vals_out,
    vector[int]& modes_out,
) noexcept nogil:
    """Northwest-corner monotone coupling plan (no cost aggregation).

    ``modes_out[a]`` is 0 if ``p_rem`` was strictly smaller than ``q_rem`` at
    step ``a`` (P binding) and 1 otherwise (Q binding or tie).
    """
    cdef size_t max_entries = n + m - 1
    rows_out.resize(max_entries)
    cols_out.resize(max_entries)
    vals_out.resize(max_entries)
    modes_out.resize(max_entries)

    cdef size_t i = 0
    cdef size_t j = 0
    cdef size_t idx = 0
    cdef double p_rem = p[0]
    cdef double q_rem = q[0]
    cdef double flow
    cdef int mode

    while i < n and j < m:
        if p_rem < q_rem:
            flow = p_rem
            mode = 0
        else:
            flow = q_rem
            mode = 1

        rows_out[idx] = <int>i
        cols_out[idx] = <int>j
        vals_out[idx] = flow
        modes_out[idx] = mode
        idx += 1

        p_rem -= flow
        q_rem -= flow

        if p_rem < PROB_EPS:
            i += 1
            if i < n:
                p_rem = p[i]

        if q_rem < PROB_EPS:
            j += 1
            if j < m:
                q_rem = q[j]

    rows_out.resize(idx)
    cols_out.resize(idx)
    vals_out.resize(idx)
    modes_out.resize(idx)
    return idx


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
) noexcept nogil:
    """Northwest-corner plan plus GCW bilinear cross-term aggregation."""
    cdef size_t idx
    cdef size_t a
    cdef size_t b
    cdef int i_a
    cdef int j_a
    cdef int i_b
    cdef int j_b
    cdef double v_a
    cdef double cross_term = 0.0

    idx = nw_plan(p, q, n, m, rows_out, cols_out, vals_out, modes_out)

    for a in range(idx):
        i_a = rows_out[a]
        j_a = cols_out[a]
        v_a = vals_out[a]
        cross_term += v_a * v_a * dist_at(d_p, n, i_a, i_a) * dist_at(d_q, m, j_a, j_a)
        for b in range(a):
            i_b = rows_out[b]
            j_b = cols_out[b]
            cross_term += 2.0 * v_a * vals_out[b] * dist_at(d_p, n, i_a, i_b) * dist_at(d_q, m, j_b, j_a)

    return cross_term


cdef void nw_backward_marginals(
    const vector[int]& rows,
    const vector[int]& cols,
    const vector[int]& modes,
    const vector[double]& G,
    size_t n,
    size_t m,
    vector[double]& adj_p_out,
    vector[double]& adj_q_out,
) noexcept nogil:
    """Reverse-mode subgradient of NW plan: given ``G_a = dL/dw_a``, write
    ``dL/dp`` and ``dL/dq`` to the output vectors.

    Uses the local affine expressions of the NW plan on its (fixed) support:
        mode 0 (P binding): w_a + sum_{a' < a, i_a' == i_a} w_a' = p[i_a]
        mode 1 (Q binding): w_a + sum_{a' < a, j_a' == j_a} w_a' = q[j_a]
    Processing in reverse a propagates the linear sensitivities exactly.
    """
    adj_p_out.assign(n, 0.0)
    adj_q_out.assign(m, 0.0)
    cdef vector[double] G_work = G
    cdef size_t num_steps = rows.size()
    cdef ssize_t a
    cdef size_t a_prime
    cdef int i_a
    cdef int j_a
    cdef double g_a

    if num_steps == 0:
        return

    for a in range(<ssize_t>num_steps - 1, -1, -1):
        g_a = G_work[<size_t>a]
        if g_a == 0.0:
            continue
        i_a = rows[<size_t>a]
        j_a = cols[<size_t>a]
        if modes[<size_t>a] == 0:
            adj_p_out[i_a] += g_a
            for a_prime in range(<size_t>a):
                if rows[a_prime] == i_a:
                    G_work[a_prime] -= g_a
        else:
            adj_q_out[j_a] += g_a
            for a_prime in range(<size_t>a):
                if cols[a_prime] == j_a:
                    G_work[a_prime] -= g_a


cdef tuple solve_transport_lp_with_duals(
    object cost,
    object theta,
    object phi,
    object gurobi_env,
):
    """Solve ``min sum c_ij x_ij`` over the transport polytope and return
    ``(x*, pi_row, pi_col)`` where the latter two are the optimal duals of the
    row and column marginal equality constraints.

    For our usage ``c = -V``; the max value ``sum V_ij w_ij`` then has
    sensitivities ``+w*`` w.r.t. ``V``, ``-pi_row`` w.r.t. ``theta`` and
    ``-pi_col`` w.r.t. ``phi`` (envelope theorem on the equivalent minimisation).
    """
    import gurobipy as gp
    from gurobipy import GRB

    cdef Py_ssize_t n = cost.shape[0]
    cdef Py_ssize_t m = cost.shape[1]

    model = gp.Model("", env=gurobi_env)
    model.Params.OutputFlag = 0

    x = model.addMVar((n, m), lb=0.0, ub=1.0)
    row_cons = model.addConstr(x.sum(axis=1) == theta)
    col_cons = model.addConstr(x.sum(axis=0) == phi)
    model.setObjective((cost * x).sum(), GRB.MINIMIZE)
    model.optimize()

    cdef bint relaxed = False
    if model.status == GRB.INFEASIBLE:
        model.feasRelaxS(1, False, False, True)
        model.optimize()
        relaxed = True

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Gurobi failed to find an optimal solution. Status: {model.Status}"
        )

    sol = np.asarray(x.X, dtype=np.float64).reshape(n, m)

    if relaxed:
        # Duals of the relaxed model are not meaningful for the original
        # constraints; surface zeros so the caller's subgradient drops the
        # marginal sensitivity at that node (the recursion continues to use
        # the optimal plan from the relaxed LP).
        pi_row = np.zeros(n, dtype=np.float64)
        pi_col = np.zeros(m, dtype=np.float64)
    else:
        pi_row = np.asarray(row_cons.Pi, dtype=np.float64).reshape(n)
        pi_col = np.asarray(col_cons.Pi, dtype=np.float64).reshape(m)

    return sol, pi_row, pi_col


cdef object solve_transport_lp(
    object cost,
    object theta,
    object phi,
    object gurobi_env,
):
    sol, _, _ = solve_transport_lp_with_duals(cost, theta, phi, gurobi_env)
    return sol


cdef class GCWTapeEntry:
    """Forward state for one ``couple_value(P, Q)`` call, used in backward.

    Each ``couple_value`` invocation that actually computes (not memo-hit)
    appends exactly one tape entry. The ``kind`` field selects which of the
    case-specific fields below are populated.
    """

    def __cinit__(self):
        self.kind = -1
        self.side_P = -1
        self.side_Q = -1


cdef class GCWGradients:
    """Subgradient bundle for ``circuit2`` parameters.

    Attributes
    ----------
    value : float
        The crossterm scalar (same as ``gcw_crossterm``).
    sum_grads : dict[int, numpy.ndarray]
        Maps ``SumNode.id`` to a vector ``g`` of the same length as that node's
        ``parameters``; ``g[k] = d(value)/d(theta_k)``.
    cat_grads : dict[int, numpy.ndarray]
        Maps ``CategoricalInputNode.id`` to a vector ``g`` of the same length
        as ``probabilities``; ``g[k] = d(value)/d(p_k)``.

    The gradients are treated as unconstrained (no simplex projection). To use
    them for a feasible optimization step, project onto the simplex tangent
    (e.g. subtract the mean before stepping, then renormalize).
    """

    def __cinit__(self):
        self.value = 0.0
        self.sum_grads = {}
        self.cat_grads = {}


cdef CircuitNode _unwrap_root(object root) except *:
    if hasattr(root, "root"):
        root = root.root
    if not isinstance(root, CircuitNode):
        raise TypeError("root must be a CircuitNode or Circuit instance")
    return <CircuitNode>root
