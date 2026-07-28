# -*- coding: utf-8 -*-
"""
Tests for soil moisture module.

Covers: SoilMoisture (constructor, validation, _properties_calculation).

Reference formulas (see carm/properties/soil_moisture.py):
    k  = b1 + b2 * wr + b3 * sqrt(wr)                          (Chung-Horton, 1987)
    cp = (1.92e6 * xs + 2.51e6 * x0 + 4.18e6 * wr) / rho        (de Vries, 1963)
    rho = wr * rho_water + (1 - wr) * rho_dry
"""
import numpy as np
import pytest

from carm.properties import SoilMoisture


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def water_input():
    return np.full(10, 1e-7, dtype=np.float64)  # m/s


@pytest.fixture
def sand(water_input):
    return SoilMoisture(water_input=water_input, rho_dry=1500.0, soil_type="sand")


# ============================================================
# SoilMoisture — constructor
# ============================================================

def test_soil_moisture_stores_values(sand, water_input):
    np.testing.assert_array_equal(sand.water_input, water_input)
    assert sand.rho_dry == 1500.0


def test_soil_moisture_invalid_soil_type(water_input):
    with pytest.raises(ValueError):
        SoilMoisture(water_input=water_input, rho_dry=1500.0, soil_type="silt")


@pytest.mark.parametrize(
    "soil_type,b1,b2,b3,theta_s,theta_r,x0",
    [
        ("sand", 0.228, -2.406, 4.909, 0.417, 0.020, 0.012),
        ("loam", 0.243, 0.393, 1.534, 0.434, 0.027, 0.018),
        ("clay", -0.197, -0.962, 2.521, 0.385, 0.090, 0.024),
    ],
)
def test_soil_moisture_params_by_type(
    water_input, soil_type, b1, b2, b3, theta_s, theta_r, x0
):
    sm = SoilMoisture(water_input=water_input, rho_dry=1500.0, soil_type=soil_type)
    assert sm.b1_loc == pytest.approx(b1)
    assert sm.b2_loc == pytest.approx(b2)
    assert sm.b3_loc == pytest.approx(b3)
    assert sm.theta_s_loc == pytest.approx(theta_s)
    assert sm.theta_r_loc == pytest.approx(theta_r)
    assert sm.x0_loc == pytest.approx(x0)
    assert sm.xs_loc == pytest.approx(1 - theta_s - x0)


def test_soil_moisture_initial_state_is_zero(sand):
    assert sand.Wvol_prev == 0.0
    assert sand.Wvol_r == 0.0
    assert sand.Wvol_loss == 0.0
    assert sand.Wvol_evap == 0.0


# ============================================================
# SoilMoisture — _properties_calculation, water balance
# ============================================================

def test_first_step_no_loss_no_evap(sand):
    """At the first step Wvol_prev=0: no drainage loss nor evaporation when q<=0."""
    V = 10.0
    A_irr = 0.5
    timesteps = 3600.0
    sand._properties_calculation(step=0, timesteps=timesteps, V=V, A_irr=A_irr, q=0.0)

    expected_raw = sand.water_input[0] * A_irr * timesteps
    expected = np.clip(expected_raw, sand.theta_r_loc * V, sand.theta_s_loc * V)
    assert sand.Wvol_r == pytest.approx(expected)
    assert sand.Wvol_loss == pytest.approx(0.0)
    assert sand.Wvol_evap == pytest.approx(0.0)


def test_evaporation_only_when_q_positive(sand):
    """q negative (heat extraction) must not generate evaporation."""
    V = 10.0
    sand._properties_calculation(step=0, timesteps=3600.0, V=V, A_irr=0.5, q=-500.0)
    assert sand.Wvol_evap == pytest.approx(0.0)


def test_evaporation_formula_when_q_positive(sand):
    """q positive (heat injection) generates evaporation according to the formula."""
    V = 10.0
    q = 500.0
    timesteps = 3600.0
    sand._properties_calculation(step=0, timesteps=timesteps, V=V, A_irr=0.5, q=q)
    expected_evap = (q * timesteps) / sand.w_latent / 1000.0
    assert sand.Wvol_evap == pytest.approx(expected_evap)


def test_dry_step_floors_to_residual_content(sand):
    """Without water input, the volume must still be floored to the residual value theta_r*V."""
    V = 10.0
    sand.water_input = np.zeros(10, dtype=np.float64)
    sand._properties_calculation(step=0, timesteps=3600.0, V=V, A_irr=0.5, q=0.0)
    assert sand.Wvol_r == pytest.approx(sand.theta_r_loc * V)


def test_excess_water_caps_to_saturation(sand):
    """A very high water input must be capped to the saturation value theta_s*V."""
    V = 10.0
    sand.water_input = np.full(10, 1e3, dtype=np.float64)
    sand._properties_calculation(step=0, timesteps=3600.0, V=V, A_irr=0.5, q=0.0)
    assert sand.Wvol_r == pytest.approx(sand.theta_s_loc * V)


def test_wvol_r_never_negative(sand):
    """Invariant: Wvol_r >= 0 always (guaranteed by the lower clip theta_r*V > 0)."""
    V = 10.0
    sand.water_input = np.zeros(10, dtype=np.float64)
    for step in range(3):
        sand._properties_calculation(step=step, timesteps=3600.0, V=V, A_irr=0.5, q=1e9)
        assert sand.Wvol_r >= 0.0


def test_state_carries_over_between_steps(sand):
    """The second step must apply the drainage loss on the first step's Wvol_prev."""
    V = 10.0
    A_irr = 0.5
    timesteps = 3600.0

    sand._properties_calculation(step=0, timesteps=timesteps, V=V, A_irr=A_irr, q=0.0)
    wvol_after_step0 = sand.Wvol_r

    sand._properties_calculation(step=1, timesteps=timesteps, V=V, A_irr=A_irr, q=0.0)

    expected_loss = wvol_after_step0 * sand.loss_factor
    expected_raw = wvol_after_step0 + sand.water_input[1] * A_irr * timesteps - expected_loss
    expected = np.clip(expected_raw, sand.theta_r_loc * V, sand.theta_s_loc * V)

    assert sand.Wvol_loss == pytest.approx(expected_loss)
    assert sand.Wvol_r == pytest.approx(expected)


# ============================================================
# SoilMoisture — thermophysical formulas
# ============================================================

def test_k_formula(sand):
    V = 10.0
    k, cp, rho = sand._properties_calculation(
        step=0, timesteps=3600.0, V=V, A_irr=0.5, q=0.0
    )
    wr = sand.W_content
    expected_k = sand.b1_loc + sand.b2_loc * wr + sand.b3_loc * wr**0.5
    assert k == pytest.approx(expected_k)


def test_rho_formula(sand):
    V = 10.0
    k, cp, rho = sand._properties_calculation(
        step=0, timesteps=3600.0, V=V, A_irr=0.5, q=0.0
    )
    wr = sand.W_content
    expected_rho = wr * sand.w_rho + (1 - wr) * sand.rho_dry
    assert rho == pytest.approx(expected_rho)


def test_cp_formula(sand):
    V = 10.0
    k, cp, rho = sand._properties_calculation(
        step=0, timesteps=3600.0, V=V, A_irr=0.5, q=0.0
    )
    wr = sand.W_content
    expected_cp_vol = 1.92e6 * sand.xs_loc + 2.51e6 * sand.x0_loc + 4.18e6 * wr
    expected_cp = expected_cp_vol / rho
    assert cp == pytest.approx(expected_cp)


def test_returns_match_stored_attributes(sand):
    k, cp, rho = sand._properties_calculation(
        step=0, timesteps=3600.0, V=10.0, A_irr=0.5, q=0.0
    )
    assert k == pytest.approx(sand.k)
    assert cp == pytest.approx(sand.cp)
    assert rho == pytest.approx(sand.rho)
