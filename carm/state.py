# -*- coding: utf-8 -*-
"""
Simulation state module.

Manages the temperature state vector during time-stepping,
tracking both the current and previous time step values.
"""
from numpy.typing import NDArray

class State:
    """
    Temperature state vector for the BHE simulation.

    Holds the current and previous time step temperature arrays,
    used by the time-stepping loop to advance and roll back the solution.

    Attributes
    ----------
    T_state : NDArray
        Current temperature state vector.
    T_old : NDArray
        Temperature state vector at the previous time step.
    """

    def __init__(self, T0: NDArray) -> None:
        self.T_state = T0.copy()
        self.T_old = self.T_state.copy()

    def save_old(self) -> None:
        self.T_old = self.T_state.copy()

    def update(self, T_new) -> None:
        self.T_state = T_new.copy()
