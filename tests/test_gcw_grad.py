"""Tests for the differentiable GCW cross-term query (``gcw_crossterm_and_grad``).

Each test compares the analytic subgradient returned by
``gcw_crossterm_and_grad`` to a symmetric finite-difference reference. To avoid
non-smooth points of the underlying piecewise-linear/quadratic structure
(LP degeneracy at sum-sum nodes, argmax / Hungarian ties, NW mass coincidences)
the test fixtures use random non-symmetric weights and probability vectors.

All tests require a working Gurobi license and are marked ``gurobi``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("scipy")
gp = pytest.importorskip("gurobipy", reason="gurobipy not installed")

from fastcircuits import (
    CategoricalInputNode,
    ProductNode,
    SumNode,
    gcw_crossterm,
    gcw_crossterm_and_grad,
)


@pytest.fixture
def gurobi_env():
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
    except gp.GurobiError:
        pytest.skip("No Gurobi license available")
    yield env
    env.dispose()


def _fd_gradient_simplex(
    f, params, *, eps=1e-5
):
    """Symmetric finite-difference gradient on the simplex tangent.

    Given a callable ``f(p)`` and a probability vector ``params``, returns
    a vector ``g`` of the same length such that ``g[i] - g[j]`` approximates
    ``df/d(params[i]) - df/d(params[j])`` (i.e. the directional derivative
    along ``e_i - e_j``). The level of the returned ``g`` is set so that
    ``sum(g) == 0`` (consistent with the simplex tangent).
    """
    n = len(params)
    dirs = np.zeros(n)
    # Perturb each component against component 0.
    for i in range(1, n):
        p_plus = np.array(params, dtype=np.float64)
        p_minus = np.array(params, dtype=np.float64)
        p_plus[i] += eps
        p_plus[0] -= eps
        p_minus[i] -= eps
        p_minus[0] += eps
        dirs[i] = (f(p_plus) - f(p_minus)) / (2.0 * eps)
    # dirs[i] approximates df/dx_i - df/dx_0.
    # Choose g with g[0] = -mean(dirs)/.., or simpler: g - mean(g) e ~ dirs - mean(dirs).
    # Use g_i = dirs[i] with g_0 = 0, then re-center to sum=0.
    g = dirs.copy()
    g[0] = 0.0
    g -= g.mean()
    return g


def _project_to_simplex_tangent(g):
    g = np.asarray(g, dtype=np.float64)
    return g - g.mean()


def _set_cat_probs(leaf: CategoricalInputNode, probs):
    """Build a new categorical leaf with the same id/scope but new probs.

    ``CategoricalInputNode`` validates probabilities at construction, so we
    can't mutate in place; we just build a fresh node when needed.
    """
    return CategoricalInputNode(
        id=int(leaf.id),
        scope_var=leaf.scope_as_list()[0],
        probabilities=list(probs),
    )


@pytest.mark.gurobi
class TestForwardValueMatch:
    """The differentiable solver must return exactly the same value as ``gcw_crossterm``."""

    def test_leaf_value_match(self, gurobi_env):
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.3, 0.7])
        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.55, 0.45])
        v_ref = gcw_crossterm(leaf1, leaf2, gurobi_env=gurobi_env)
        v_grad, grads = gcw_crossterm_and_grad(leaf1, leaf2, gurobi_env=gurobi_env)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-12)
        assert_allclose(grads.value, v_ref, rtol=0, atol=1e-12)

    def test_sum_sum_value_match(self, gurobi_env):
        c1a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.8, 0.15, 0.05])
        c1b = CategoricalInputNode(id=1, scope_var=0, probabilities=[0.2, 0.3, 0.5])
        circ1 = SumNode(id=2, children=[c1a, c1b], parameters=[0.35, 0.65])
        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=[0.1, 0.6, 0.3])
        c2b = CategoricalInputNode(id=4, scope_var=0, probabilities=[0.7, 0.2, 0.1])
        circ2 = SumNode(id=5, children=[c2a, c2b], parameters=[0.4, 0.6])

        v_ref = gcw_crossterm(circ1, circ2, gurobi_env=gurobi_env)
        v_grad, _ = gcw_crossterm_and_grad(circ1, circ2, gurobi_env=gurobi_env)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-10)

    def test_product_product_value_match(self, gurobi_env):
        c1a = CategoricalInputNode(id=0, scope_var=0, probabilities=[0.4, 0.6])
        c1b = CategoricalInputNode(id=1, scope_var=1, probabilities=[0.7, 0.3])
        circ1 = ProductNode(id=2, children=[c1a, c1b])
        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=[0.2, 0.8])
        c2b = CategoricalInputNode(id=4, scope_var=1, probabilities=[0.55, 0.45])
        circ2 = ProductNode(id=5, children=[c2a, c2b])

        v_ref = gcw_crossterm(circ1, circ2, gurobi_env=gurobi_env)
        v_grad, _ = gcw_crossterm_and_grad(circ1, circ2, gurobi_env=gurobi_env)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-10)

    def test_mixed_value_match(self, gurobi_env):
        # circuit1: SumNode whose children are products and a leaf
        rng = np.random.default_rng(0)
        leaves1 = [
            CategoricalInputNode(
                id=10 + i, scope_var=i % 2, probabilities=list(rng.dirichlet([1.0] * 3))
            )
            for i in range(4)
        ]
        prod1 = ProductNode(id=20, children=[leaves1[0], leaves1[1]])
        prod2 = ProductNode(id=21, children=[leaves1[2], leaves1[3]])
        circ1 = SumNode(id=22, children=[prod1, prod2], parameters=[0.45, 0.55])

        leaves2 = [
            CategoricalInputNode(
                id=30 + i, scope_var=i % 2, probabilities=list(rng.dirichlet([1.0] * 3))
            )
            for i in range(4)
        ]
        prod3 = ProductNode(id=40, children=[leaves2[0], leaves2[1]])
        prod4 = ProductNode(id=41, children=[leaves2[2], leaves2[3]])
        circ2 = SumNode(id=42, children=[prod3, prod4], parameters=[0.3, 0.7])

        v_ref = gcw_crossterm(circ1, circ2, gurobi_env=gurobi_env)
        v_grad, _ = gcw_crossterm_and_grad(circ1, circ2, gurobi_env=gurobi_env)
        assert_allclose(v_grad, v_ref, rtol=0, atol=1e-10)


@pytest.mark.gurobi
class TestLeafGradients:
    """Subgradient at a categorical x categorical pair via NW backward."""

    def test_leaf_cat_grad_matches_fd(self, gurobi_env):
        rng = np.random.default_rng(42)
        p_fixed = rng.dirichlet([1.5, 1.0, 0.7])
        q_init = rng.dirichlet([1.0, 1.2, 0.9])
        # circuit1 is fixed; circuit2 is the leaf we differentiate.
        leaf1 = CategoricalInputNode(id=0, scope_var=0, probabilities=list(p_fixed))

        def f_value(probs):
            leaf = CategoricalInputNode(
                id=1, scope_var=0, probabilities=list(probs)
            )
            return gcw_crossterm(leaf1, leaf, gurobi_env=gurobi_env)

        leaf2 = CategoricalInputNode(id=1, scope_var=0, probabilities=list(q_init))
        _, grads = gcw_crossterm_and_grad(leaf1, leaf2, gurobi_env=gurobi_env)
        assert 1 in grads.cat_grads, "circuit2 leaf should receive a gradient"
        g = _project_to_simplex_tangent(grads.cat_grads[1])
        g_fd = _fd_gradient_simplex(f_value, q_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)


@pytest.mark.gurobi
class TestSumSumGradients:
    """Sum-sum LP gradients use the row/col constraint duals."""

    def test_sum_sum_phi_grad_matches_fd(self, gurobi_env):
        rng = np.random.default_rng(7)
        p1 = rng.dirichlet([1.0, 1.0, 1.0])
        p2 = rng.dirichlet([1.0, 1.0, 1.0])
        c1a = CategoricalInputNode(id=0, scope_var=0, probabilities=list(p1))
        c1b = CategoricalInputNode(id=1, scope_var=0, probabilities=list(p2))
        circ1 = SumNode(id=2, children=[c1a, c1b], parameters=[0.3, 0.7])

        q1 = rng.dirichlet([1.0, 1.0, 1.0])
        q2 = rng.dirichlet([1.0, 1.0, 1.0])
        phi_init = np.array([0.42, 0.58])

        def f_value(phi):
            c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=list(q1))
            c2b = CategoricalInputNode(id=4, scope_var=0, probabilities=list(q2))
            return gcw_crossterm(
                circ1,
                SumNode(id=5, children=[c2a, c2b], parameters=list(phi)),
                gurobi_env=gurobi_env,
            )

        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=list(q1))
        c2b = CategoricalInputNode(id=4, scope_var=0, probabilities=list(q2))
        circ2 = SumNode(id=5, children=[c2a, c2b], parameters=list(phi_init))
        _, grads = gcw_crossterm_and_grad(circ1, circ2, gurobi_env=gurobi_env)
        assert 5 in grads.sum_grads
        g = _project_to_simplex_tangent(grads.sum_grads[5])
        g_fd = _fd_gradient_simplex(f_value, phi_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)


@pytest.mark.gurobi
class TestProductGradients:
    """Product-product Hungarian gradients propagate through ed and matched couples."""

    def test_product_leaf_grad_matches_fd(self, gurobi_env):
        rng = np.random.default_rng(99)
        c1a = CategoricalInputNode(
            id=0, scope_var=0, probabilities=list(rng.dirichlet([1.0, 1.0]))
        )
        c1b = CategoricalInputNode(
            id=1, scope_var=1, probabilities=list(rng.dirichlet([1.0, 1.0]))
        )
        circ1 = ProductNode(id=2, children=[c1a, c1b])

        q_init = rng.dirichlet([1.0, 1.0])
        other_q = list(rng.dirichlet([1.0, 1.0]))

        def f_value(probs):
            c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=list(probs))
            c2b = CategoricalInputNode(id=4, scope_var=1, probabilities=other_q)
            return gcw_crossterm(
                circ1, ProductNode(id=5, children=[c2a, c2b]), gurobi_env=gurobi_env
            )

        c2a = CategoricalInputNode(id=3, scope_var=0, probabilities=list(q_init))
        c2b = CategoricalInputNode(id=4, scope_var=1, probabilities=other_q)
        circ2 = ProductNode(id=5, children=[c2a, c2b])
        _, grads = gcw_crossterm_and_grad(circ1, circ2, gurobi_env=gurobi_env)
        assert 3 in grads.cat_grads
        g = _project_to_simplex_tangent(grads.cat_grads[3])
        g_fd = _fd_gradient_simplex(f_value, q_init, eps=1e-6)
        assert_allclose(g, g_fd, atol=1e-5)


@pytest.mark.gurobi
class TestMixedGradients:
    """End-to-end finite-difference check on a small sum-of-products circuit."""

    def _build_circuits(self, rng):
        # circuit1 = SumNode(prod, prod) with shared input scopes 0, 1
        l1 = CategoricalInputNode(
            id=0, scope_var=0, probabilities=list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        l2 = CategoricalInputNode(
            id=1, scope_var=1, probabilities=list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        l3 = CategoricalInputNode(
            id=2, scope_var=0, probabilities=list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        l4 = CategoricalInputNode(
            id=3, scope_var=1, probabilities=list(rng.dirichlet([1.0, 1.0, 1.0]))
        )
        prodA = ProductNode(id=4, children=[l1, l2])
        prodB = ProductNode(id=5, children=[l3, l4])
        circ1 = SumNode(id=6, children=[prodA, prodB], parameters=[0.4, 0.6])
        return circ1

    def test_sum_grad_in_sum_of_products(self, gurobi_env):
        rng = np.random.default_rng(13)
        circ1 = self._build_circuits(rng)

        # circuit2 = SumNode of two products with new leaves
        def build_circ2(theta):
            l1 = CategoricalInputNode(
                id=10, scope_var=0, probabilities=[0.55, 0.25, 0.2]
            )
            l2 = CategoricalInputNode(
                id=11, scope_var=1, probabilities=[0.3, 0.5, 0.2]
            )
            l3 = CategoricalInputNode(
                id=12, scope_var=0, probabilities=[0.15, 0.6, 0.25]
            )
            l4 = CategoricalInputNode(
                id=13, scope_var=1, probabilities=[0.7, 0.1, 0.2]
            )
            prodA = ProductNode(id=14, children=[l1, l2])
            prodB = ProductNode(id=15, children=[l3, l4])
            return SumNode(id=16, children=[prodA, prodB], parameters=list(theta))

        theta_init = np.array([0.45, 0.55])
        circ2 = build_circ2(theta_init)
        _, grads = gcw_crossterm_and_grad(circ1, circ2, gurobi_env=gurobi_env)
        assert 16 in grads.sum_grads
        g_an = _project_to_simplex_tangent(grads.sum_grads[16])
        g_fd = _fd_gradient_simplex(
            lambda th: gcw_crossterm(circ1, build_circ2(th), gurobi_env=gurobi_env),
            theta_init,
            eps=1e-6,
        )
        assert_allclose(g_an, g_fd, atol=1e-5)


@pytest.mark.gurobi
class TestGradientAscentSmoke:
    """A few projected-gradient steps should improve the crossterm monotonically."""

    def test_ascent_increases_crossterm(self, gurobi_env):
        rng = np.random.default_rng(2026)
        n_cats = 3
        p_fixed = rng.dirichlet([1.0] * n_cats)
        leaf_ref = CategoricalInputNode(
            id=0, scope_var=0, probabilities=list(p_fixed)
        )

        q = rng.dirichlet([1.0] * n_cats)
        lr = 5e-2
        history = []
        for _ in range(8):
            leaf_learn = CategoricalInputNode(
                id=1, scope_var=0, probabilities=list(q)
            )
            value, grads = gcw_crossterm_and_grad(
                leaf_ref, leaf_learn, gurobi_env=gurobi_env
            )
            history.append(value)
            g = _project_to_simplex_tangent(grads.cat_grads[1])
            q = q + lr * g
            # Project back to simplex (clip and renormalize).
            q = np.clip(q, 1e-6, None)
            q = q / q.sum()

        # Non-strict monotone non-decrease over the sequence.
        diffs = np.diff(history)
        # Allow tiny noise at later steps but require the trajectory to
        # increase overall.
        assert history[-1] >= history[0] - 1e-12
        assert (diffs >= -1e-8).all(), f"saw decrease in ascent: {history}"
