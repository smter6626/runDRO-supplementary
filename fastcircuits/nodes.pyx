# distutils: language = c++

from cpython.ref cimport PyObject

from libcpp cimport bool as cpp_bool
from libcpp.random cimport mt19937_64, uniform_real_distribution
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.math cimport exp, fabs, INFINITY, isfinite, log

import numpy as np

cdef double PROB_TOL = 1e-6
cdef double NEG_INF = -INFINITY

cdef double safe_log(double x) noexcept nogil:
    if x > 0.0:
        return log(x)
    return NEG_INF

cdef double logsumexp(const vector[double]& log_vals) noexcept nogil:
    cdef size_t n = log_vals.size()
    cdef size_t i
    cdef double max_log
    cdef double sum_exp = 0.0
    cdef double x
    if n == 0:
        return NEG_INF
    max_log = log_vals[0]
    for i in range(1, n):
        if log_vals[i] > max_log:
            max_log = log_vals[i]
    if not isfinite(max_log) or max_log == NEG_INF:
        return NEG_INF
    for i in range(n):
        x = log_vals[i]
        if isfinite(x) and x > NEG_INF:
            sum_exp += exp(x - max_log)
    if sum_exp <= 0.0:
        return NEG_INF
    return max_log + log(sum_exp)

cdef void scope_clear(unordered_set[int]& s):
    s.clear()

cdef void scope_union_from(unordered_set[int]& dest, unordered_set[int]& src):
    dest.insert(src.begin(), src.end())

cdef void validate_non_negative_scope(unordered_set[int]& scope):
    cdef int v
    for v in sorted(scope):
        if v < 0:
            raise ValueError(f"scope indices must be non-negative, got {v}")

cdef void validate_probabilities(
    const vector[double]& p,
    cpp_bool normalize_check,
):
    cdef size_t i
    cdef size_t n = p.size()
    cdef double total = 0.0
    cdef double x
    if n == 0:
        raise ValueError("probability vector must not be empty")
    for i in range(n):
        x = p[i]
        if not isfinite(x) or x < 0.0:
            raise ValueError("probabilities must be finite and non-negative")
        total += x
    if normalize_check and fabs(total - 1.0) > PROB_TOL:
        raise ValueError(f"probabilities must sum to 1, got {total}")

cdef void fill_vector_double(vector[double]& dest, object values) except *:
    cdef object item
    dest.clear()
    for item in values:
        dest.push_back(float(item))

cdef class CircuitNode:
    def __init__(self, size_t id):
        self.id = id
        scope_clear(self.scope)

    cdef void _propagate_scope_impl(self, unordered_set[size_t]& visited) except *:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _propagate_scope_impl"
        )

    cpdef void propagate_scope(self) except *:
        cdef unordered_set[size_t] visited
        self._propagate_scope_impl(visited)

    cpdef list scope_as_list(self):
        return sorted(self.scope)

    cpdef void set_scope_from_iterable(self, object indices) except *:
        cdef int v
        scope_clear(self.scope)
        for v in indices:
            if v < 0:
                raise ValueError(f"scope indices must be non-negative, got {v}")
            self.scope.insert(<int>v)
        validate_non_negative_scope(self.scope)

cdef CircuitNode node_from_ptr(PyObject* obj):
    return <CircuitNode><object>obj

cdef void scope_union_from_ptrs(
    unordered_set[int]& dest,
    const vector[PyObject*]& children,
):
    cdef PyObject* raw
    cdef CircuitNode node
    cdef size_t i
    cdef size_t n = children.size()
    for i in range(n):
        raw = children[i]
        if raw != NULL:
            node = node_from_ptr(raw)
            scope_union_from(dest, node.scope)

cdef void fill_children(
    vector[PyObject*]& ptrs,
    list refs,
    object children,
) except *:
    cdef object child
    ptrs.clear()
    refs.clear()
    for child in children:
        if not isinstance(child, CircuitNode):
            raise TypeError("children must be CircuitNode instances")
        refs.append(child)
        ptrs.push_back(<PyObject*>child)

cdef class CategoricalInputNode(CircuitNode):
    def __init__(self, size_t id, int scope_var, object probabilities):
        if scope_var < 0:
            raise ValueError(f"scope_var must be non-negative, got {scope_var}")
        CircuitNode.__init__(self, id)
        scope_clear(self.scope)
        self.scope.insert(scope_var)
        fill_vector_double(self.probabilities, probabilities)
        if self.probabilities.size() < 2:
            raise ValueError("categorical distribution must have at least 2 outcomes")
        validate_probabilities(self.probabilities, True)

    cdef int scope_var_c(self) except *:
        cdef int v
        if self.scope.size() != 1:
            raise ValueError(
                f"CategoricalInputNode {self.id} must have scope of size 1"
            )
        for v in sorted(self.scope):
            return v
        raise ValueError(f"CategoricalInputNode {self.id} has empty scope")

    cdef double probability_at(self, size_t index) except *:
        if index >= self.probabilities.size():
            raise IndexError(
                f"outcome index {index} out of range for node {self.id}"
            )
        return self.probabilities[index]

    cdef size_t num_outcomes(self):
        return self.probabilities.size()

    cdef void _propagate_scope_impl(self, unordered_set[size_t]& visited) except *:
        cdef pair[unordered_set[size_t].iterator, cpp_bool] inserted
        inserted = visited.insert(self.id)
        if not inserted.second:
            return
        if self.scope.size() != 1:
            raise ValueError(
                f"CategoricalInputNode {self.id} must have scope of size 1, "
                f"got {self.scope.size()}"
            )

    cpdef list probabilities_list(self):
        cdef size_t i
        cdef size_t n = self.probabilities.size()
        cdef list out = []
        for i in range(n):
            out.append(self.probabilities[i])
        return out

    cpdef void set_probabilities_list(self, object probabilities) except *:
        fill_vector_double(self.probabilities, probabilities)
        if self.probabilities.size() < 2:
            raise ValueError("categorical distribution must have at least 2 outcomes")
        validate_probabilities(self.probabilities, True)

    cpdef Py_ssize_t cardinality(self):
        return <Py_ssize_t>self.probabilities.size()

cdef class ProductNode(CircuitNode):
    def __init__(self, size_t id, object children):
        CircuitNode.__init__(self, id)
        self._child_refs = []
        if len(children) < 1:
            raise ValueError("ProductNode must have at least one child")
        fill_children(self._children, self._child_refs, children)

    cdef size_t num_children(self):
        return self._children.size()

    cdef CircuitNode child_at(self, size_t index) except *:
        if index >= self._children.size():
            raise IndexError(f"child index {index} out of range")
        return node_from_ptr(self._children[index])

    cdef void _propagate_scope_impl(self, unordered_set[size_t]& visited) except *:
        cdef pair[unordered_set[size_t].iterator, cpp_bool] inserted
        cdef size_t i
        cdef size_t n
        cdef CircuitNode child
        inserted = visited.insert(self.id)
        if not inserted.second:
            return
        n = self._children.size()
        for i in range(n):
            child = self.child_at(i)
            child._propagate_scope_impl(visited)
        scope_clear(self.scope)
        scope_union_from_ptrs(self.scope, self._children)
        validate_non_negative_scope(self.scope)

    cpdef list children(self):
        return list(self._child_refs)

cdef class SumNode(CircuitNode):
    def __init__(self, size_t id, object children, object parameters):
        CircuitNode.__init__(self, id)
        self._child_refs = []
        if len(children) < 1:
            raise ValueError("SumNode must have at least one child")
        if len(children) != len(parameters):
            raise ValueError(
                f"children and parameters length mismatch: "
                f"{len(children)} vs {len(parameters)}"
            )
        fill_children(self._children, self._child_refs, children)
        fill_vector_double(self.parameters, parameters)
        validate_probabilities(self.parameters, True)

    cdef size_t num_children(self):
        return self._children.size()

    cdef CircuitNode child_at(self, size_t index) except *:
        if index >= self._children.size():
            raise IndexError(f"child index {index} out of range")
        return node_from_ptr(self._children[index])

    cdef double parameter_at(self, size_t index) except *:
        if index >= self.parameters.size():
            raise IndexError(f"parameter index {index} out of range")
        return self.parameters[index]

    cdef void _propagate_scope_impl(self, unordered_set[size_t]& visited) except *:
        cdef pair[unordered_set[size_t].iterator, cpp_bool] inserted
        cdef size_t i
        cdef size_t n
        cdef CircuitNode child
        inserted = visited.insert(self.id)
        if not inserted.second:
            return
        n = self._children.size()
        for i in range(n):
            child = self.child_at(i)
            child._propagate_scope_impl(visited)
        scope_clear(self.scope)
        scope_union_from_ptrs(self.scope, self._children)
        validate_non_negative_scope(self.scope)

    cpdef list children(self):
        return list(self._child_refs)

    cpdef list parameters_list(self):
        cdef size_t i
        cdef size_t n = self.parameters.size()
        cdef list out = []
        for i in range(n):
            out.append(self.parameters[i])
        return out

    cpdef void set_parameters_list(self, object parameters) except *:
        cdef size_t n_old = self.parameters.size()
        cdef object params_list = list(parameters)
        if len(params_list) != n_old:
            raise ValueError(
                f"parameter length mismatch: expected {n_old}, "
                f"got {len(params_list)}"
            )
        fill_vector_double(self.parameters, params_list)
        validate_probabilities(self.parameters, True)

# --- Evidence ---

cdef class Evidence:
    def __init__(self, object assignment=None):
        self._values.clear()
        if assignment is not None:
            self.update_from_mapping(assignment)

    cpdef void update_from_mapping(self, object assignment) except *:
        cdef object key
        cdef object value
        cdef int var
        cdef int outcome
        self._values.clear()
        for key, value in assignment.items():
            var = int(key)
            outcome = int(value)
            if var < 0:
                raise ValueError(f"variable index must be non-negative, got {var}")
            if outcome < 0:
                raise ValueError(f"outcome value must be non-negative, got {outcome}")
            self._values[var] = outcome

    cdef int get(self, int var) except *:
        if self._values.find(var) == self._values.end():
            raise ValueError(f"missing evidence for variable {var}")
        return self._values[var]

    cdef void require_vars(self, unordered_set[int]& scope_vars) except *:
        cdef int v
        for v in sorted(scope_vars):
            self.get(v)

    cdef void validate_value(self, int var, int value, Py_ssize_t cardinality) except *:
        if value < 0 or value >= cardinality:
            raise ValueError(
                f"evidence for variable {var}: outcome {value} out of range "
                f"[0, {cardinality})"
            )

# --- Queries ---

cdef class QueryContext:
    def __cinit__(self):
        self.memo.clear()
        self.query_kind = QueryKind.LIKELIHOOD

    cdef void clear_memo(self):
        self.memo.clear()

    cdef bint memo_get(self, size_t node_id, double* out) except *:
        if self.memo.find(node_id) == self.memo.end():
            return False
        out[0] = self.memo[node_id]
        return True

    cdef void memo_put(self, size_t node_id, double value) except *:
        self.memo[node_id] = value

cdef double evaluate_likelihood_categorical(
    CategoricalInputNode node,
    QueryContext ctx,
) except *:
    cdef int var = node.scope_var_c()
    cdef int value = ctx.evidence.get(var)
    cdef Py_ssize_t card = node.cardinality()
    ctx.evidence.validate_value(var, value, card)
    return node.probability_at(<size_t>value)

cdef double evaluate_likelihood_product(
    ProductNode node,
    QueryContext ctx,
) except *:
    cdef size_t i
    cdef size_t n = node.num_children()
    cdef double result = 1.0
    cdef CircuitNode child
    for i in range(n):
        child = node.child_at(i)
        result *= evaluate(child, ctx)
    return result

cdef double evaluate_likelihood_sum(
    SumNode node,
    QueryContext ctx,
) except *:
    cdef size_t i
    cdef size_t n = node.num_children()
    cdef double result = 0.0
    cdef CircuitNode child
    cdef double weight
    for i in range(n):
        child = node.child_at(i)
        weight = node.parameter_at(i)
        result += weight * evaluate(child, ctx)
    return result

cdef double evaluate_log_likelihood_categorical(
    CategoricalInputNode node,
    QueryContext ctx,
) except *:
    cdef int var = node.scope_var_c()
    cdef int value = ctx.evidence.get(var)
    cdef Py_ssize_t card = node.cardinality()
    ctx.evidence.validate_value(var, value, card)
    return safe_log(node.probability_at(<size_t>value))

cdef double evaluate_log_likelihood_product(
    ProductNode node,
    QueryContext ctx,
) except *:
    cdef size_t i
    cdef size_t n = node.num_children()
    cdef double result = 0.0
    cdef CircuitNode child
    for i in range(n):
        child = node.child_at(i)
        result += evaluate(child, ctx)
    return result

cdef double evaluate_log_likelihood_sum(
    SumNode node,
    QueryContext ctx,
) except *:
    cdef size_t i
    cdef size_t n = node.num_children()
    cdef vector[double] terms
    cdef CircuitNode child
    cdef double weight
    cdef double child_log
    terms.resize(n)
    for i in range(n):
        child = node.child_at(i)
        weight = node.parameter_at(i)
        child_log = evaluate(child, ctx)
        terms[i] = safe_log(weight) + child_log
    return logsumexp(terms)

cdef double evaluate_impl(CircuitNode node, QueryContext ctx) except *:
    if ctx.query_kind == QueryKind.LOG_LIKELIHOOD:
        if isinstance(node, CategoricalInputNode):
            return evaluate_log_likelihood_categorical(
                <CategoricalInputNode>node, ctx
            )
        if isinstance(node, ProductNode):
            return evaluate_log_likelihood_product(<ProductNode>node, ctx)
        if isinstance(node, SumNode):
            return evaluate_log_likelihood_sum(<SumNode>node, ctx)
        raise TypeError(
            f"unsupported node type for query: {type(node).__name__}"
        )
    if isinstance(node, CategoricalInputNode):
        return evaluate_likelihood_categorical(<CategoricalInputNode>node, ctx)
    if isinstance(node, ProductNode):
        return evaluate_likelihood_product(<ProductNode>node, ctx)
    if isinstance(node, SumNode):
        return evaluate_likelihood_sum(<SumNode>node, ctx)
    raise TypeError(f"unsupported node type for query: {type(node).__name__}")

cdef double evaluate(CircuitNode node, QueryContext ctx) except *:
    cdef double cached
    cdef double result
    cdef size_t node_id = node.id
    if ctx.memo_get(node_id, &cached):
        return cached
    result = evaluate_impl(node, ctx)
    ctx.memo_put(node_id, result)
    return result

cdef double _run_query(CircuitNode root, object assignment, QueryKind kind) except *:
    cdef Evidence ev = Evidence(assignment)
    cdef QueryContext ctx = QueryContext()
    ctx.evidence = ev
    ctx.query_kind = kind
    if root.scope.size() == 0:
        raise ValueError(
            "root scope is empty; call propagate_scope() on the circuit first"
        )
    ev.require_vars(root.scope)
    return evaluate(root, ctx)

cpdef double likelihood(CircuitNode root, object assignment) except *:
    return _run_query(root, assignment, QueryKind.LIKELIHOOD)

cpdef double log_likelihood(CircuitNode root, object assignment) except *:
    return _run_query(root, assignment, QueryKind.LOG_LIKELIHOOD)

# --- Sampling ---

cdef size_t _draw_index(const vector[double]& probs, double u) noexcept nogil:
    cdef size_t n = probs.size()
    cdef size_t i
    cdef double cum = 0.0
    for i in range(n):
        cum += probs[i]
        if u < cum:
            return i
    return n - 1

cdef void _sample_node(
    CircuitNode node,
    mt19937_64& rng,
    uniform_real_distribution[double]& dist,
    dict out,
) except *:
    cdef size_t i
    cdef size_t idx
    cdef double u
    cdef CategoricalInputNode cat
    cdef ProductNode prod
    cdef SumNode s
    if isinstance(node, CategoricalInputNode):
        cat = <CategoricalInputNode>node
        u = dist(rng)
        idx = _draw_index(cat.probabilities, u)
        out[cat.scope_var_c()] = <int>idx
    elif isinstance(node, ProductNode):
        prod = <ProductNode>node
        for i in range(prod.num_children()):
            _sample_node(prod.child_at(i), rng, dist, out)
    elif isinstance(node, SumNode):
        s = <SumNode>node
        u = dist(rng)
        idx = _draw_index(s.parameters, u)
        _sample_node(s.child_at(idx), rng, dist, out)
    else:
        raise TypeError(
            f"unsupported node type for sampling: {type(node).__name__}"
        )

cpdef list sample(CircuitNode root, Py_ssize_t n_samples, object seed=None) except *:
    if root.scope.size() == 0:
        raise ValueError(
            "root scope is empty; call propagate_scope() on the circuit first"
        )
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    cdef mt19937_64 rng
    cdef unsigned long long rng_seed
    if seed is None:
        import time
        rng_seed = <unsigned long long>time.time_ns()
    else:
        rng_seed = <unsigned long long>int(seed)
    rng = mt19937_64(rng_seed)
    cdef uniform_real_distribution[double] dist = uniform_real_distribution[double](
        0.0, 1.0
    )
    cdef list results = []
    cdef Py_ssize_t s
    cdef dict row
    for s in range(n_samples):
        row = {}
        _sample_node(root, rng, dist, row)
        results.append(row)
    return results

# --- Differentiable mean log-likelihood ---

cdef class LogLikelihoodGradients:
    """Gradient bundle for the mean log-likelihood objective.

    Attributes
    ----------
    value : float
        The mean log-likelihood of the dataset.
    sum_grads : dict[int, numpy.ndarray]
        Maps ``SumNode.id`` to a vector ``g`` of the same length as that node's
        ``parameters``; ``g[k] = d(mean_ll)/d(theta_k)``.
    cat_grads : dict[int, numpy.ndarray]
        Maps ``CategoricalInputNode.id`` to a vector ``g`` of the same length as
        ``probabilities``; ``g[k] = d(mean_ll)/d(p_k)``.

    Gradients are taken w.r.t. the linear parameters (no simplex projection).
    To take a feasible step, project onto the simplex tangent (subtract the mean
    before stepping, then renormalize).
    """

    def __cinit__(self):
        self.value = 0.0
        self.sum_grads = {}
        self.cat_grads = {}


cdef class _LLGradContext:
    """Reverse-mode AD over the log-likelihood DAG, accumulated over a dataset."""

    cdef Evidence evidence
    cdef unordered_map[size_t, double] memo
    cdef unordered_map[size_t, double] adjoints
    cdef list tape
    cdef dict sum_grads
    cdef dict cat_grads

    def __cinit__(self):
        self.tape = []
        self.sum_grads = {}
        self.cat_grads = {}

    cdef void _reset(self):
        self.memo.clear()
        self.adjoints.clear()
        self.tape = []

    cdef inline void _add_adjoint(self, size_t nid, double val):
        self.adjoints[nid] = self.adjoints[nid] + val

    cdef object _sum_grad_arr(self, CircuitNode node, size_t n):
        cdef object key = node.id
        cdef object arr
        if key in self.sum_grads:
            return self.sum_grads[key]
        arr = np.zeros(n, dtype=np.float64)
        self.sum_grads[key] = arr
        return arr

    cdef object _cat_grad_arr(self, CircuitNode node, size_t n):
        cdef object key = node.id
        cdef object arr
        if key in self.cat_grads:
            return self.cat_grads[key]
        arr = np.zeros(n, dtype=np.float64)
        self.cat_grads[key] = arr
        return arr

    cdef double _forward(self, CircuitNode node) except *:
        cdef size_t nid = node.id
        cdef double result
        if self.memo.count(nid):
            return self.memo[nid]
        result = self._forward_impl(node)
        self.memo[nid] = result
        self.tape.append(node)
        return result

    cdef double _forward_impl(self, CircuitNode node) except *:
        cdef CategoricalInputNode cat
        cdef ProductNode prod
        cdef SumNode s
        cdef int var
        cdef int value
        cdef Py_ssize_t card
        cdef size_t i
        cdef size_t n
        cdef double total
        cdef vector[double] terms
        cdef CircuitNode child
        cdef double weight
        cdef double child_log
        if isinstance(node, CategoricalInputNode):
            cat = <CategoricalInputNode>node
            var = cat.scope_var_c()
            value = self.evidence.get(var)
            card = cat.cardinality()
            self.evidence.validate_value(var, value, card)
            return safe_log(cat.probability_at(<size_t>value))
        if isinstance(node, ProductNode):
            prod = <ProductNode>node
            n = prod.num_children()
            total = 0.0
            for i in range(n):
                child = prod.child_at(i)
                total += self._forward(child)
            return total
        if isinstance(node, SumNode):
            s = <SumNode>node
            n = s.num_children()
            terms.resize(n)
            for i in range(n):
                child = s.child_at(i)
                weight = s.parameter_at(i)
                child_log = self._forward(child)
                terms[i] = safe_log(weight) + child_log
            return logsumexp(terms)
        raise TypeError(f"unsupported node type for query: {type(node).__name__}")

    cdef void _run_backward(self) except *:
        cdef ssize_t k
        cdef CircuitNode node
        cdef size_t nid
        cdef double bar
        for k in range(<ssize_t>len(self.tape) - 1, -1, -1):
            node = <CircuitNode>self.tape[k]
            nid = node.id
            if self.adjoints.count(nid) == 0:
                continue
            bar = self.adjoints[nid]
            if bar == 0.0:
                continue
            if isinstance(node, CategoricalInputNode):
                self._backward_cat(<CategoricalInputNode>node, bar)
            elif isinstance(node, ProductNode):
                self._backward_prod(<ProductNode>node, bar)
            elif isinstance(node, SumNode):
                self._backward_sum(<SumNode>node, bar)

    cdef void _backward_cat(self, CategoricalInputNode node, double bar) except *:
        cdef int var = node.scope_var_c()
        cdef int value = self.evidence.get(var)
        cdef double p_v = node.probability_at(<size_t>value)
        cdef object arr
        if p_v <= 0.0:
            return
        arr = self._cat_grad_arr(node, node.num_outcomes())
        arr[value] += bar / p_v

    cdef void _backward_prod(self, ProductNode node, double bar) except *:
        cdef size_t i
        cdef size_t n = node.num_children()
        cdef CircuitNode child
        for i in range(n):
            child = node.child_at(i)
            self._add_adjoint(child.id, bar)

    cdef void _backward_sum(self, SumNode node, double bar) except *:
        cdef size_t i
        cdef size_t n = node.num_children()
        cdef double ll = self.memo[node.id]
        cdef CircuitNode child
        cdef double weight
        cdef double child_ll
        cdef double log_w
        cdef object arr
        if ll == NEG_INF:
            return
        arr = self._sum_grad_arr(node, n)
        for i in range(n):
            child = node.child_at(i)
            weight = node.parameter_at(i)
            child_ll = self.memo[child.id]
            log_w = safe_log(weight)
            if log_w > NEG_INF and child_ll > NEG_INF:
                self._add_adjoint(child.id, bar * exp(log_w + child_ll - ll))
            if child_ll > NEG_INF:
                arr[i] += bar * exp(child_ll - ll)

    cdef tuple solve_dataset(self, CircuitNode root, list dataset):
        cdef Py_ssize_t n = len(dataset)
        cdef Py_ssize_t idx
        cdef double total_ll = 0.0
        cdef double inv_n
        cdef double ll
        cdef object datapoint
        cdef LogLikelihoodGradients grads
        if root.scope.size() == 0:
            raise ValueError(
                "root scope is empty; call propagate_scope() on the circuit first"
            )
        if n == 0:
            raise ValueError("dataset must contain at least one datapoint")
        inv_n = 1.0 / <double>n
        for idx in range(n):
            datapoint = dataset[idx]
            self.evidence = Evidence(datapoint)
            self.evidence.require_vars(root.scope)
            self._reset()
            ll = self._forward(root)
            total_ll += ll
            if ll > NEG_INF:
                self._add_adjoint(root.id, inv_n)
                self._run_backward()
        grads = LogLikelihoodGradients()
        grads.value = total_ll * inv_n
        grads.sum_grads = self.sum_grads
        grads.cat_grads = self.cat_grads
        return (grads.value, grads)


cpdef tuple mean_log_likelihood_and_grad(CircuitNode root, object dataset):
    """Mean log-likelihood of a dataset and its gradient w.r.t. circuit params.

    Parameters
    ----------
    root : CircuitNode
        Circuit root with propagated scope.
    dataset : iterable of dict[int, int]
        Each datapoint is a full ``{var: value}`` assignment over the scope.

    Returns
    -------
    (mean_ll, grads) : tuple[float, LogLikelihoodGradients]
        ``mean_ll`` is the average log-likelihood; ``grads`` carries
        ``sum_grads`` / ``cat_grads`` keyed by ``node.id`` (gradients of
        ``mean_ll`` w.r.t. the linear parameters). Use with simplex-tangent
        ascent for maximum-likelihood learning.
    """
    cdef _LLGradContext ctx = _LLGradContext()
    cdef list data_list = list(dataset)
    return ctx.solve_dataset(root, data_list)
