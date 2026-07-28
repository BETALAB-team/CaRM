# -*- coding: utf-8 -*-
"""
Results test: heat-flow direction and thermal response trend.

Runs short, full simulations (Simulation.run()) and checks that the
physical direction of the response is correct:

  - a cold inlet (Tf1 < Tg) must extract heat from the ground
    (q_fluid = mw * cp_w * (Tf1 - Tfout) < 0, per the sign convention
    documented in soil_moisture.py: "Negative in heat extraction,
    positive in heat injection");
  - a warm inlet (Tf1 > Tg) must inject heat into the ground (q_fluid > 0);
  - under a constant Tf1 forcing, the near-borehole ground progressively
    responds, so the magnitude of the exchanged heat must trend downward
    over time (declining extraction/injection rate) rather than diverge
    or reverse sign.

This is deliberately not a strict energy-conservation ledger (that would
require re-deriving the surface/far-field/bottom boundary fluxes
independently, at high risk of encoding a wrong "expected" formula) — it
checks sign and trend, which is enough to catch coupling/sign-convention
regressions in the assembled system without duplicating solver internals.

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

N_STEPS = 48  # 2 days, hourly steps
TG = 13.0


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
        Tg=TG,
        stratification=stratification,
    )


@pytest.fixture
def env_props():
    return EnvironmentalProperties(
        R_ext=0.04, absorptance=0.6, eps=0.9, At=10.0,
        tau=3600.0, tau_y=31_536_000.0, tau_shift=1_296_000.0,
    )


@pytest.fixture
def env_series():
    T_ext = np.full(N_STEPS, 10.0, dtype=np.float64)
    solar_rad = np.full(N_STEPS, 200.0, dtype=np.float64)
    return EnvironmentalTimeSeries.from_array(13.0, T_ext, solar_rad)


def _run(model, env_props, env_series, fluid, Tf1_value, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), Tf1_value, dtype=np.float64)

    sim = Simulation(
        model=model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    T_history = sim.run()

    ns = model.ground[0].m_mesh_sup + 1
    nm = model.ground[0].n_mesh * model.ground[0].m_mesh
    id_outlet = model.borehole.id_outlet

    Tfout = T_history[1:, 0, ns + nm + id_outlet]
    q_fluid = mw_tot[0] * fluid.cp_w * (Tf1[0] - Tfout)
    return q_fluid


def test_cold_inlet_extracts_heat(model, env_props, env_series, fluid, tmp_path, monkeypatch):
    q_fluid = _run(model, env_props, env_series, fluid, TG - 5.0, tmp_path, monkeypatch)
    assert np.all(q_fluid < 0.0)


def test_warm_inlet_injects_heat(model, env_props, env_series, fluid, tmp_path, monkeypatch):
    q_fluid = _run(model, env_props, env_series, fluid, TG + 5.0, tmp_path, monkeypatch)
    assert np.all(q_fluid > 0.0)


def test_extraction_rate_declines_under_constant_forcing(
    model, env_props, env_series, fluid, tmp_path, monkeypatch
):
    """With Tf1 held constant, the progressive cooling of the near-borehole
    ground must reduce the heat-exchange intensity over time."""
    q_fluid = _run(model, env_props, env_series, fluid, TG - 5.0, tmp_path, monkeypatch)

    first_quarter = np.mean(np.abs(q_fluid[: N_STEPS // 4]))
    last_quarter = np.mean(np.abs(q_fluid[-N_STEPS // 4 :]))
    assert last_quarter < first_quarter
