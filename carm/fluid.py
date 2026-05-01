"""
Fluid properties module.

Defines the thermophysical properties of the heat carrier fluid
circulating in the borehole heat exchanger.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fluid:
    """
    Thermophysical properties of the heat carrier fluid.

    Attributes
    ----------
    k_w : float
        Thermal conductivity [W / (m K)].
    rho_w : float
        Density [kg/m³].
    cp_w : float
        Specific heat capacity [J / (kg K)].
    ni_w : float
        Kinematic viscosity [m²/s].
    """
    k_w: float  # W/mK
    rho_w: float  # kg/m3
    cp_w: float  # J/kgK
    ni_w: float  # m2/s

    def __post_init__(self) -> None:
        if self.k_w <= 0:
            raise ValueError("k_w must be > 0")
        if self.rho_w <= 0:
            raise ValueError("rho_w must be > 0")
        if self.cp_w <= 0:
            raise ValueError("cp_w must be > 0")
        if self.ni_w <= 0:
            raise ValueError("ni_w must be > 0")
