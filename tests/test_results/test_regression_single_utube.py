# -*- coding: utf-8 -*-
"""
Results test: golden-master regression for a single U-tube simulation.

Runs a fixed, small-scale single-borehole configuration (loosely modeled
on examples/SingleUtube.py, with a coarser mesh for speed) for 48 hourly
steps under constant extraction (Tf1 = Tg - 5 degC) and checks a handful
of checkpoint values against numbers captured from the current
implementation on 2026-07-28.

This is a characterization test, not a proof of physical correctness: it
exists to catch *unintended* changes in the solver's output while you
work on solver.py / soil_moisture.py, not to assert the numbers are
"right" in an absolute sense (test_results/test_fls_vs_pygfunction.py and
test_results/test_energy_flow_direction.py cover correctness/physical
sanity instead).

If you deliberately change the physics/assembly and this test starts
failing, regenerate the reference values by running this exact fixture
through ``Simulation.run()`` and printing the checkpoints below, then
update REF_* accordingly — do not just loosen the tolerance.

``Simulation.run()`` unconditionally writes a ``results/*.npz`` file
(``Simulation._save_results``); ``monkeypatch.chdir(tmp_path)`` keeps that
side effect out of the repository working directory.
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
    PhysicalModel,
    EnvironmentalProperties,
    EnvironmentalTimeSeries,
    Simulation,
)

N_STEPS = 48

# Reference values captured on 2026-07-28 from the configuration defined
# in this file (see the `t_history` fixture below).
REF_TS_STEP1 = 10.815172036112143
REF_TS_STEP24 = 10.444300512621506
REF_TS_STEP48 = 10.71112945491945

REF_TFOUT_STEP1 = 8.833015359568776
REF_TFOUT_STEP24 = 8.327697130035515
REF_TFOUT_STEP48 = 8.316122603340155

REF_TGROUND_FIRST_CELL_STEP1 = 11.036372444248954
REF_TGROUND_FIRST_CELL_STEP24 = 10.87443901544809
REF_TGROUND_FIRST_CELL_STEP48 = 10.742181960681517

REF_QFLUID_STEP1 = -700.967154651438
REF_QFLUID_STEP12 = -281.5937893076968
REF_QFLUID_STEP24 = -275.7511277431283
REF_QFLUID_STEP48 = -266.0113757074826


@pytest.fixture
def fluid():
    return Fluid(
        k_w=0.568709114496803,
        rho_w=1000.1435933169,
        cp_w=4207.40834247225,
        ni_w=1.49626063208248e-6,
    )


@pytest.fixture
def model(fluid):
    ground_mesh = GroundMesh(n_mesh=4, m_mesh=6, m_mesh_sup=2, m_mesh_inf=2)
    stratification = [(1.8, 947.37, 1900.0, 26.0)]
    borehole = SingleUtube(
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
    return PhysicalModel(
        ground_geom=GroundGeometry(rn=5.0, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=borehole,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
    )


@pytest.fixture
def mw_tot():
    return np.full((1, N_STEPS), 0.2, dtype=np.float64)


@pytest.fixture
def tf1():
    return np.full((1, N_STEPS), 8.0, dtype=np.float64)  # Tg - 5, extraction


@pytest.fixture
def t_history(model, fluid, mw_tot, tf1, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    env_props = EnvironmentalProperties(
        R_ext=0.04, absorptance=0.6, eps=0.9, At=10.0,
        tau=3600.0, tau_y=31_536_000.0, tau_shift=1_296_000.0,
    )
    T_ext = np.full(N_STEPS, 10.0, dtype=np.float64)
    solar_rad = np.full(N_STEPS, 200.0, dtype=np.float64)
    env_series = EnvironmentalTimeSeries.from_array(13.0, T_ext, solar_rad)

    sim = Simulation(
        model=model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=tf1,
    )
    return sim.run()


def test_surface_temperature_checkpoints(t_history):
    assert t_history[1, 0, 0] == pytest.approx(REF_TS_STEP1, rel=1e-6)
    assert t_history[24, 0, 0] == pytest.approx(REF_TS_STEP24, rel=1e-6)
    assert t_history[48, 0, 0] == pytest.approx(REF_TS_STEP48, rel=1e-6)


def test_fluid_outlet_checkpoints(t_history, model):
    ns = model.ground[0].m_mesh_sup + 1
    nm = model.ground[0].n_mesh * model.ground[0].m_mesh
    id_outlet = model.borehole.id_outlet

    assert t_history[1, 0, ns + nm + id_outlet] == pytest.approx(REF_TFOUT_STEP1, rel=1e-6)
    assert t_history[24, 0, ns + nm + id_outlet] == pytest.approx(REF_TFOUT_STEP24, rel=1e-6)
    assert t_history[48, 0, ns + nm + id_outlet] == pytest.approx(REF_TFOUT_STEP48, rel=1e-6)


def test_ground_first_cell_checkpoints(t_history, model):
    ns = model.ground[0].m_mesh_sup + 1
    assert t_history[1, 0, ns] == pytest.approx(REF_TGROUND_FIRST_CELL_STEP1, rel=1e-6)
    assert t_history[24, 0, ns] == pytest.approx(REF_TGROUND_FIRST_CELL_STEP24, rel=1e-6)
    assert t_history[48, 0, ns] == pytest.approx(REF_TGROUND_FIRST_CELL_STEP48, rel=1e-6)


def test_qfluid_checkpoints(t_history, model, fluid, mw_tot, tf1):
    ns = model.ground[0].m_mesh_sup + 1
    nm = model.ground[0].n_mesh * model.ground[0].m_mesh
    id_outlet = model.borehole.id_outlet

    Tfout = t_history[1:, 0, ns + nm + id_outlet]
    q_fluid = mw_tot[0] * fluid.cp_w * (tf1[0] - Tfout)

    assert q_fluid[0] == pytest.approx(REF_QFLUID_STEP1, rel=1e-6)
    assert q_fluid[11] == pytest.approx(REF_QFLUID_STEP12, rel=1e-6)
    assert q_fluid[23] == pytest.approx(REF_QFLUID_STEP24, rel=1e-6)
    assert q_fluid[47] == pytest.approx(REF_QFLUID_STEP48, rel=1e-6)
