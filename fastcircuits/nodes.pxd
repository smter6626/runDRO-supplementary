from cpython.ref cimport PyObject
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

cdef class CircuitNode:
    cdef readonly size_t id
    cdef unordered_set[int] scope

    cdef void _propagate_scope_impl(self, unordered_set[size_t]& visited) except *
    cpdef void propagate_scope(self) except *
    cpdef list scope_as_list(self)
    cpdef void set_scope_from_iterable(self, object indices) except *

cdef CircuitNode node_from_ptr(PyObject* obj)

cdef class CategoricalInputNode(CircuitNode):
    cdef vector[double] probabilities

    cdef int scope_var_c(self) except *
    cdef double probability_at(self, size_t index) except *
    cdef size_t num_outcomes(self)

    cpdef list probabilities_list(self)
    cpdef void set_probabilities_list(self, object probabilities) except *
    cpdef Py_ssize_t cardinality(self)

cdef class ProductNode(CircuitNode):
    cdef vector[PyObject*] _children
    cdef list _child_refs

    cdef size_t num_children(self)
    cdef CircuitNode child_at(self, size_t index) except *

    cpdef list children(self)

cdef class SumNode(CircuitNode):
    cdef vector[PyObject*] _children
    cdef list _child_refs
    cdef vector[double] parameters

    cdef size_t num_children(self)
    cdef CircuitNode child_at(self, size_t index) except *
    cdef double parameter_at(self, size_t index) except *

    cpdef list children(self)
    cpdef list parameters_list(self)
    cpdef void set_parameters_list(self, object parameters) except *

cdef class Evidence:
    cdef unordered_map[int, int] _values

    cdef int get(self, int var) except *
    cdef void require_vars(self, unordered_set[int]& scope_vars) except *
    cdef void validate_value(self, int var, int value, Py_ssize_t cardinality) except *

    cpdef void update_from_mapping(self, object assignment) except *

cdef enum QueryKind:
    LIKELIHOOD = 0
    LOG_LIKELIHOOD = 1

cdef class QueryContext:
    cdef Evidence evidence
    cdef unordered_map[size_t, double] memo
    cdef QueryKind query_kind

    cdef void clear_memo(self)
    cdef bint memo_get(self, size_t node_id, double* out) except *
    cdef void memo_put(self, size_t node_id, double value) except *

cpdef double likelihood(CircuitNode root, object assignment) except *
cpdef double log_likelihood(CircuitNode root, object assignment) except *
cpdef list sample(CircuitNode root, Py_ssize_t n_samples, object seed=*) except *

cdef class LogLikelihoodGradients:
    cdef public double value
    cdef public dict sum_grads
    cdef public dict cat_grads

cpdef tuple mean_log_likelihood_and_grad(CircuitNode root, object dataset)
