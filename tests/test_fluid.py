# -*- coding: utf-8 -*-
"""
Tests for fluid properties module.

Covers: Fluid (constructor, validation, immutability).
"""
import pytest

from carm import Fluid


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


# ============================================================
# Fluid — storage
# ============================================================

def test_fluid_stores_values(fluid):
    assert fluid.k_w == 0.568709114496803
    assert fluid.rho_w == 1000.1435933169
    assert fluid.cp_w == 4207.40834247225
    assert fluid.ni_w == 1.49626063208248e-6


def test_fluid_is_frozen(fluid):
    with pytest.raises(Exception):
        fluid.k_w = 99.0


# ============================================================
# Fluid — validation
# ============================================================

def test_fluid_invalid_k_w():
    with pytest.raises(ValueError):
        Fluid(k_w=0.0, rho_w=1000.0, cp_w=4200.0, ni_w=1.5e-6)


def test_fluid_invalid_rho_w():
    with pytest.raises(ValueError):
        Fluid(k_w=0.57, rho_w=0.0, cp_w=4200.0, ni_w=1.5e-6)


def test_fluid_invalid_cp_w():
    with pytest.raises(ValueError):
        Fluid(k_w=0.57, rho_w=1000.0, cp_w=0.0, ni_w=1.5e-6)


def test_fluid_invalid_ni_w():
    with pytest.raises(ValueError):
        Fluid(k_w=0.57, rho_w=1000.0, cp_w=4200.0, ni_w=0.0)


def test_fluid_negative_k_w():
    with pytest.raises(ValueError):
        Fluid(k_w=-0.1, rho_w=1000.0, cp_w=4200.0, ni_w=1.5e-6)
