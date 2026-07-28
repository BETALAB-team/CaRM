# -*- coding: utf-8 -*-
"""
Tests for physical model module.

Covers: PhysicalModel (constructor, single/multi-borehole assembly,
        _get_temperatures slicing).

Mesh sizes are kept small on purpose (n_mesh/m_mesh/m_mesh_sup/m_mesh_inf)
to keep GroundProperties/Field construction fast; the numeric values are
otherwise consistent with the SingleUtube.py example.
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
    Field,
    PhysicalModel,
)
from carm.properties import GroundProperties
from carm.state import State


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def fluid():
    return Fluid(
        k_w=0.568709114496803,
        rho_w=1000.1435933169,
        cp_w=4207.40834247225,
        ni_w=1.49626063208248e-6,
    )


@pytest.fixture
def ground_mesh():
    return GroundMesh(n_mesh=4, m_mesh=6, m_mesh_sup=2, m_mesh_inf=2)


@pytest.fixture
def stratification():
    # must cover L + L_sup + L_inf = 20 + 1 + 5 = 26 m
    return [(1.8, 947.37, 1900.0, 26.0)]


@pytest.fixture
def single_utube(fluid):
    return SingleUtube(
        geom=BoreholeGeometry(Lbore=20.0, D0=0.15),
        mesh=BoreholeMesh(m_mesh=6),
        thermalprops=BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8),
        fluid=fluid,
        pipe_thick=0.003,
        pipe_spacing=0.0823,
        Dpi=0.026,
        n_pipes=2,
        Rp0=0.25,
        RppB=0.72,
    )


@pytest.fixture
def two_bhe_field():
    fi = FieldInput(n_bhes=2, xmin=-5.0, ymin=-5.0, xmax=15.0, ymax=5.0, rb=0.075)
    fi.from_array(np.array([0.0, 5.0]), np.array([0.0, 0.0]))
    return fi


# ============================================================
# PhysicalModel — single borehole
# ============================================================

def test_single_borehole_builds_one_ground(ground_mesh, stratification, single_utube, fluid):
    model = PhysicalModel(
        ground_geom=GroundGeometry(rn=5.0, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=single_utube,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
    )
    assert model.fieldinput is None
    assert len(model.ground) == 1
    assert isinstance(model.ground[0], GroundProperties)
    assert model.ground[0].rn == 5.0


def test_single_borehole_requires_rn(ground_mesh, stratification, single_utube, fluid):
    with pytest.raises(ValueError):
        PhysicalModel(
            ground_geom=GroundGeometry(rn=None, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
            ground_mesh=ground_mesh,
            borehole=single_utube,
            fluid=fluid,
            Tg=13.0,
            stratification=stratification,
        )


def test_fieldinput_with_single_bhe_takes_single_borehole_path(
    ground_mesh, stratification, single_utube, fluid
):
    """fieldinput.n_bhes == 1 must be treated as the single-borehole case."""
    fi = FieldInput(n_bhes=1, xmin=-5.0, ymin=-5.0, xmax=5.0, ymax=5.0, rb=0.075)
    fi.from_array(np.array([0.0]), np.array([0.0]))

    model = PhysicalModel(
        ground_geom=GroundGeometry(rn=5.0, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=single_utube,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
        fieldinput=fi,
    )
    assert len(model.ground) == 1
    assert not hasattr(model, "field")


# ============================================================
# PhysicalModel — multi-borehole
# ============================================================

def test_multi_borehole_builds_one_ground_per_bhe(
    ground_mesh, stratification, single_utube, fluid, two_bhe_field
):
    model = PhysicalModel(
        ground_geom=GroundGeometry(rn=None, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=single_utube,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
        fieldinput=two_bhe_field,
    )
    assert len(model.ground) == 2
    assert isinstance(model.field, Field)


def test_multi_borehole_rn_matches_field_req(
    ground_mesh, stratification, single_utube, fluid, two_bhe_field
):
    model = PhysicalModel(
        ground_geom=GroundGeometry(rn=None, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=single_utube,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
        fieldinput=two_bhe_field,
    )
    for j in range(2):
        assert model.ground[j].rn == pytest.approx(model.field.field_dict[j]["req"])


def test_multi_borehole_rejects_explicit_rn(
    ground_mesh, stratification, single_utube, fluid, two_bhe_field
):
    with pytest.raises(ValueError):
        PhysicalModel(
            ground_geom=GroundGeometry(rn=5.0, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
            ground_mesh=ground_mesh,
            borehole=single_utube,
            fluid=fluid,
            Tg=13.0,
            stratification=stratification,
            fieldinput=two_bhe_field,
        )


# ============================================================
# PhysicalModel — _get_temperatures
# ============================================================

def test_get_temperatures_slicing(ground_mesh, stratification, single_utube, fluid):
    model = PhysicalModel(
        ground_geom=GroundGeometry(rn=5.0, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=single_utube,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
    )

    ngs = model.ground[0].m_mesh_sup + 1  # 3
    ng = model.ground[0].n_mesh * model.ground[0].m_mesh  # 24
    nb = model.borehole.n_equations * model.borehole.m_mesh  # 36
    total = ngs + ng + nb + ground_mesh.m_mesh_inf  # + 2 = 65

    T0 = np.arange(total, dtype=np.float64)[None, :]
    state = State(T0)
    state.update(T0 + 1000.0)  # T_state != T_old, to distinguish use_old

    T_borehole, T_ground, T_ground_sup, T_ground_inf, Ts = model._get_temperatures(
        state, j=0, use_old=False
    )
    np.testing.assert_array_equal(T_borehole, T0[0, (ngs + ng):(ngs + ng + nb)] + 1000.0)
    np.testing.assert_array_equal(T_ground, T0[0, ngs:(ngs + ng)] + 1000.0)
    np.testing.assert_array_equal(T_ground_sup, T0[0, 1:ngs] + 1000.0)
    np.testing.assert_array_equal(T_ground_inf, T0[0, (ngs + ng + nb):] + 1000.0)
    assert Ts == pytest.approx(1000.0)

    T_borehole_old, T_ground_old, T_ground_sup_old, T_ground_inf_old, Ts_old = (
        model._get_temperatures(state, j=0, use_old=True)
    )
    np.testing.assert_array_equal(T_borehole_old, T0[0, (ngs + ng):(ngs + ng + nb)])
    np.testing.assert_array_equal(T_ground_old, T0[0, ngs:(ngs + ng)])
    np.testing.assert_array_equal(T_ground_sup_old, T0[0, 1:ngs])
    np.testing.assert_array_equal(T_ground_inf_old, T0[0, (ngs + ng + nb):])
    assert Ts_old == pytest.approx(0.0)
