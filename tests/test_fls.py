# -*- coding: utf-8 -*-
"""
Tests for the Finite Line Source (FLS) module.

Covers: FiniteLineSolution (constructor, response_matrix shape and
        physical sanity properties, _compute_delta_t).

These are unit/property-based checks (shape, monotonicity, reciprocity,
self- vs cross-response) that do not require an external reference.
A numeric comparison against pygfunction (as ground truth) belongs to the
"results" test suite, not covered here.
"""
import numpy as np
import pytest

from carm import (
    GroundGeometry,
    GroundMesh,
    BoreholeGeometry,
    BoreholeMesh,
    BoreholeThermalProperties,
    SingleUtube,
    Fluid,
    FieldInput,
    PhysicalModel,
)
from carm.thermal_interference import FiniteLineSolution


# ============================================================
# FIXTURES
# ============================================================

H = 20.0
D0 = 0.15
n_steps = 10
time_hist = np.arange(3600, 3600 * (n_steps + 1), 3600, dtype=np.float64)


@pytest.fixture
def fluid():
    return Fluid(k_w=0.568, rho_w=1000.0, cp_w=4207.0, ni_w=1.496e-6)


@pytest.fixture
def two_borehole_model(fluid):
    ground_geom = GroundGeometry(D0=D0, L=H, L_sup=1.0, L_inf=5.0, rn=None)
    ground_mesh = GroundMesh(n_mesh=4, m_mesh=6, m_mesh_sup=2, m_mesh_inf=2)

    borehole = SingleUtube(
        geom=BoreholeGeometry(Lbore=H, D0=D0),
        mesh=BoreholeMesh(m_mesh=6),
        thermalprops=BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8),
        fluid=fluid,
        pipe_thick=0.003,
        pipe_spacing=0.05,
        Dpi=0.032,
        n_pipes=2,
        Rp0=0.1,
        RppB=0.1,
    )

    fi = FieldInput(n_bhes=2, xmin=-2.5, ymin=-2.5, xmax=7.5, ymax=2.5, rb=D0 / 2)
    fi.from_array(np.array([0.0, 5.0]), np.array([0.0, 0.0]))

    stratification = [(1.8, 947.37, 1900.0, H + 1.0 + 5.0)]

    return PhysicalModel(
        ground_geom=ground_geom,
        ground_mesh=ground_mesh,
        borehole=borehole,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
        fieldinput=fi,
    )


@pytest.fixture
def fls(two_borehole_model):
    return FiniteLineSolution(
        physicalmodel=two_borehole_model,
        n_steps=n_steps,
        time_hist=time_hist,
    )


# ============================================================
# FiniteLineSolution — response_matrix, shape and physics
# ============================================================

def test_response_matrix_shape(fls):
    assert fls.response_matrix.shape == (n_steps + 1, 2, 2)


def test_response_matrix_zero_at_t0(fls):
    np.testing.assert_array_equal(fls.response_matrix[0], np.zeros((2, 2)))


def test_response_matrix_monotonic_in_time(fls):
    """The G thermal response must grow (or stay constant) over time."""
    diffs = np.diff(fls.response_matrix, axis=0)
    assert np.all(diffs >= -1e-9)


def test_response_matrix_self_response_positive(fls):
    assert np.all(fls.response_matrix[1:, 0, 0] > 0)
    assert np.all(fls.response_matrix[1:, 1, 1] > 0)


def test_response_matrix_self_greater_than_cross(fls):
    """A borehole 'sees' itself more than it sees a neighbor."""
    assert np.all(fls.response_matrix[1:, 0, 0] > fls.response_matrix[1:, 0, 1])
    assert np.all(fls.response_matrix[1:, 1, 1] > fls.response_matrix[1:, 1, 0])


def test_response_matrix_reciprocity(fls):
    """Identical boreholes and a symmetric field: cross response is reciprocal and self-response is equal."""
    np.testing.assert_allclose(
        fls.response_matrix[:, 0, 1], fls.response_matrix[:, 1, 0], rtol=1e-10
    )
    np.testing.assert_allclose(
        fls.response_matrix[:, 0, 0], fls.response_matrix[:, 1, 1], rtol=1e-10
    )


def test_continuous_mode_shape(two_borehole_model):
    fls_cont = FiniteLineSolution(
        physicalmodel=two_borehole_model,
        n_steps=n_steps,
        time_hist=time_hist,
        fls_mode="continuous",
    )
    assert fls_cont.response_matrix.shape == (n_steps + 1, 2, 2)
    assert np.all(fls_cont.response_matrix[1:, 0, 0] > 0)


# ============================================================
# FiniteLineSolution — _compute_delta_t
# ============================================================

def test_compute_delta_t_zero_at_step0(fls, two_borehole_model):
    q_nbhes = np.zeros((n_steps, 2), dtype=np.float64)
    dT = fls._compute_delta_t(q_nbhes=q_nbhes, step=0)
    expected_shape = (2, two_borehole_model.ground_mesh.m_mesh)
    assert dT.shape == expected_shape
    np.testing.assert_array_equal(dT, np.zeros(expected_shape))


def test_compute_delta_t_positive_load_gives_positive_penalty(fls):
    """A positive thermal load (injection) must give a positive thermal penalty."""
    q_nbhes = np.full((n_steps, 2), 1000.0, dtype=np.float64)
    dT = fls._compute_delta_t(q_nbhes=q_nbhes, step=3)
    assert np.all(dT > 0)
