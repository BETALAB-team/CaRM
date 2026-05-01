# -*- coding: utf-8 -*-
"""
Tests for simulation state module.

Covers: State (constructor, save_old, update).
"""
import numpy as np
import pytest

from carm.state import State


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def T0():
    return np.array([10.0, 11.0, 12.0, 13.0])


@pytest.fixture
def state(T0):
    return State(T0)


# ============================================================
# State — constructor
# ============================================================

def test_state_stores_t_state(state, T0):
    np.testing.assert_array_equal(state.T_state, T0)


def test_state_stores_t_old(state, T0):
    np.testing.assert_array_equal(state.T_old, T0)


def test_state_constructor_copies_array(T0):
    """T_state deve essere una copia, non un riferimento a T0."""
    state = State(T0)
    T0[0] = 999.0
    assert state.T_state[0] != 999.0


# ============================================================
# State — update
# ============================================================

def test_update_changes_t_state(state):
    T_new = np.array([20.0, 21.0, 22.0, 23.0])
    state.update(T_new)
    np.testing.assert_array_equal(state.T_state, T_new)


def test_update_copies_array(state):
    """update deve copiare T_new, non tenersi il riferimento."""
    T_new = np.array([20.0, 21.0, 22.0, 23.0])
    state.update(T_new)
    T_new[0] = 999.0
    assert state.T_state[0] != 999.0


def test_update_does_not_change_t_old(state, T0):
    state.update(np.array([20.0, 21.0, 22.0, 23.0]))
    np.testing.assert_array_equal(state.T_old, T0)


# ============================================================
# State — save_old
# ============================================================

def test_save_old_copies_t_state_to_t_old(state):
    T_new = np.array([20.0, 21.0, 22.0, 23.0])
    state.update(T_new)
    state.save_old()
    np.testing.assert_array_equal(state.T_old, T_new)


def test_save_old_copies_array(state):
    """save_old deve copiare T_state, non tenersi il riferimento."""
    T_new = np.array([20.0, 21.0, 22.0, 23.0])
    state.update(T_new)
    state.save_old()
    state.T_state[0] = 999.0
    assert state.T_old[0] != 999.0


def test_save_old_then_update_preserves_old(state, T0):
    """Sequenza tipica del time-stepping: save_old → update → T_old invariato."""
    state.save_old()
    state.update(np.array([20.0, 21.0, 22.0, 23.0]))
    np.testing.assert_array_equal(state.T_old, T0)