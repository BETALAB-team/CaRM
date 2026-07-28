# -*- coding: utf-8 -*-
"""
Results test: FLS response vs pygfunction (external reference).

Cross-checks CaRM's FiniteLineSolution against pygfunction's
finite_line_source (Cimmino, 2018) — fls.py's own docstring states that
"the numerical implementation of the integral follows the approach used
in pygfunction". This is a numeric correctness check of the integral
itself against a validated external tool, as opposed to
tests/test_fls.py, which only checks physical properties (monotonicity,
reciprocity, self > cross) without an external reference.

Two things are easy to get wrong here, both worth spelling out because an
earlier attempt at this test (later deleted, commit c6882808) got both
wrong:

1. ``response_matrix[k+1]`` is already the *cumulative* FLS response at
   ``time_hist[k]`` (the "sqrt" mode telescopes incremental shell
   integrals via ``cumsum`` internally so that ``response_matrix[k+1]``
   equals a single integral from ``a(time_hist[k])`` to infinity — exactly
   pygfunction's ``h(t)``). It must be compared directly against
   ``finite_line_source(time_hist[k], ...)``, *not* against a further
   ``np.cumsum`` of per-step pygfunction calls (that computes a different,
   much larger quantity — a sum of step responses, not a single one).

2. CaRM's distance metric (``Field._distance_matrix``) is not the plain
   borehole-to-borehole distance: every entry is corrected by the
   equivalent (Voronoi) radius ``req`` of the source borehole, including
   on the diagonal, where the "self" distance is ``req`` — not the
   physical borehole radius ``r_b`` that pygfunction's
   ``Borehole.distance()`` would use for a self-comparison. This is a
   deliberate CaRM modeling choice (the FLS penalty accounts for each
   borehole's Voronoi cell size, not just its physical radius), not
   something for this test to second-guess. To compare like with like,
   this test builds synthetic pygfunction boreholes placed at exactly the
   distances CaRM itself computed (read from
   ``model.field.distance_matrix``), rather than relying on the real
   field coordinates and pygfunction's own distance convention.
"""
import numpy as np
import pytest
import pygfunction as gt

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

H = 20.0
D0 = 0.15
k = 1.8
rho = 1900.0
cp = 947.37
alpha = k / (rho * cp)
n_steps = 30
time_hist = np.arange(3600, 3600 * (n_steps + 1), 3600, dtype=np.float64)


@pytest.fixture
def two_borehole_model():
    ground_geom = GroundGeometry(D0=D0, L=H, L_sup=1.0, L_inf=5.0, rn=None)
    ground_mesh = GroundMesh(n_mesh=4, m_mesh=6, m_mesh_sup=2, m_mesh_inf=2)
    fluid = Fluid(k_w=0.568, rho_w=1000.0, cp_w=4207.0, ni_w=1.496e-6)

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

    stratification = [(k, cp, rho, H + 1.0 + 5.0)]

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
        physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=time_hist
    )


def _pygfunction_response(distance):
    """h(t) from pygfunction for two synthetic boreholes separated by
    exactly ``distance`` meters (same H, same depth D=1.0 as CaRM/D0)."""
    b1 = gt.boreholes.Borehole(H=H, D=1.0, r_b=1e-6, x=0.0, y=0.0)
    b2 = gt.boreholes.Borehole(H=H, D=1.0, r_b=1e-6, x=distance, y=0.0)
    return np.array(
        [
            gt.heat_transfer.finite_line_source(
                t, alpha, b1, b2, reaSource=True, imgSource=False
            )
            for t in time_hist
        ]
    )


def test_fls_self_response_matches_pygfunction(fls, two_borehole_model):
    d_self = two_borehole_model.field.distance_matrix[0, 0]
    G_pyg = _pygfunction_response(d_self)
    assert fls.response_matrix[1:, 0, 0] == pytest.approx(G_pyg, rel=1e-2)


def test_fls_cross_response_matches_pygfunction(fls, two_borehole_model):
    d_cross = two_borehole_model.field.distance_matrix[0, 1]
    G_pyg = _pygfunction_response(d_cross)
    assert fls.response_matrix[1:, 0, 1] == pytest.approx(G_pyg, rel=1e-2)
