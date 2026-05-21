"""
Soil mositure module.

Empirical equations are here implemented to account for water content
in porous means. This module provides insights on ground and borehole
proprieties variability against water content.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class SoilMoisture:
    """
    Soil moisture and thermophysical properties module.

    Computes ground and borehole thermophysical properties (thermal conductivity,
    specific heat capacity, density) as a function of volumetric water content,
    using empirical correlations.

    Attributes
    ----------
    k_conv : float
        Conversion factor from BTU/(h ft °F) to W/(m K).
    cp_conv : float
        Conversion factor from BTU/(lb °F) to J/(kg K).
    rho_conv : float
        Conversion factor from lb/ft³ to kg/m³.
    loss_factor : float
        Fraction of water volume lost by drainage at each timestep [-].
    w_latent : float
        Latent heat of vaporization of water [J/kg].
    water_input : NDArray
        Water flux time series [m/s].
    k_dry : float
        Thermal conductivity of dry soil [BTU/(h ft °F)].
    cp_dry : float
        Specific heat capacity of dry soil [BTU/(lb °F)].
    rho_dry : float
        Density of dry soil [lb/ft³].
    porosity : float
        Soil porosity [-].
    Wvol_prev : float
        Water volume at the previous timestep [m³].
    Wvol_r : float
        Residual water volume at the current timestep [m³].
    Wvol_loss : float
        Water volume lost by drainage at the current timestep [m³].
    Wvol_evap : float
        Water volume lost by evaporation at the current timestep [m³].
    """

    k_conv = 1.730735  # BTU / (h * ft * °F) -> W / (m * K)
    cp_conv = 4186.8  # BTU / (lb * °F) -> J / (kg * K)
    rho_conv = 16.01846  # lb / ft**3 -> kg / m**3
    loss_factor = 0.1
    w_latent = 2250000.0

    def __init__(
        self,
        *,
        water_input: NDArray,
        k_dry: float,
        cp_dry: float,
        rho_dry: float,
        porosity: float,
    ) -> None:

        self.water_input = water_input
        self.k_dry = k_dry
        self.cp_dry = cp_dry
        self.rho_dry = rho_dry
        self.porosity = porosity

        self.Wvol_prev = 0.0
        self.Wvol_r = 0.0
        self.Wvol_loss = 0.0
        self.Wvol_evap = 0.0

    def _properties_calculation(
        self,
        step: int,
        timesteps: int,
        V: float,
        A: float,
        q: float,
    ) -> Tuple[float, float, float]:
        """
        Compute thermophysical properties from water content at a given timestep.

        Parameters
        ----------
        step : int
            Current simulation timestep index.
        timesteps : float
            Duration of each timestep [s].
        q : float
            Thermal power exchanged by the system [W].
        V : float
            Reference soil volume [m³].
        A : float
            Surface area over which water input is applied [m²].

        Returns
        -------
        k : float
            Thermal conductivity [W/(m K)].
        cp : float
            Specific heat capacity [J/(kg K)].
        rho : float
            Density [kg/m³].

        Notes
        -----
        If the computed water content falls below 7%, dry soil properties
        are returned directly.
        """

        self.Wvol_loss = self.Wvol_prev * self.loss_factor
        self.Wvol_evap = ((q * timesteps) / self.w_latent) / 1000.0
        self.Wvol_r = max(
            self.Wvol_prev
            + self.water_input[step] * A * timesteps
            - self.Wvol_loss
            + self.Wvol_evap,
            0.0,
        )

        assert self.Wvol_r >= 0

        self.Wvol_prev = self.Wvol_r

        self.W_content = (
            100.0 * self.Wvol_r / ((1 - self.porosity) * V) * 1000.0 / (110 * 16.01846)
        )

        if self.W_content < 7.0:
            return self.k_dry, self.cp_dry, self.rho_dry

        else:
            self.k = (
                (
                    (0.7 * np.log10(self.W_content) + 0.4)
                    * 10 ** (0.01 * self.rho_dry / self.rho_conv)
                )
                / 12.0
            ) * self.k_conv
            self.cp = (
                (
                    self.W_content * 1.0
                    + (100.0 - self.W_content) * self.cp_dry / self.cp_conv
                )
                / 100.0
            ) * self.cp_conv
            self.rho = (
                (
                    self.W_content * 62.4
                    + (100.0 - self.W_content) * self.rho_dry / self.rho_conv
                )
                / 100.0
            ) * self.rho_conv

        return self.k, self.cp, self.rho
