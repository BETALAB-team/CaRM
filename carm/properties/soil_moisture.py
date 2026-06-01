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
    using Chung-Horton (1987) for thermal conductivity, de Vries (1963) for
    volumetric heat capacity, and a mass-weighted average for density.

    Attributes
    ----------
    loss_factor : float
        Fraction of water volume lost by drainage at each timestep [-].
    w_rho : float
        Density of water [kg/m³].
    w_latent : float
        Latent heat of vaporization of water [J/kg].
    SOIL_PARAMS : dict
        Tabulated parameters for sand, loam, and clay soil types.
        Each entry contains: b1, b2, b3 (Chung-Horton), theta_s, theta_r
        (Rawls et al.), xs (solid volume fraction), x0 (organic matter fraction).
    water_input : NDArray
        Water flux time series [m/s].
    rho_dry : float
        Dry soil density [kg/m³].
    b1_loc : float
        Chung-Horton parameter b1 for the selected soil type [W/(m K)].
    b2_loc : float
        Chung-Horton parameter b2 for the selected soil type [W/(m K)].
    b3_loc : float
        Chung-Horton parameter b3 for the selected soil type [W/(m K)].
    theta_s_loc : float
        Saturated volumetric water content for the selected soil type [-].
    theta_r_loc : float
        Residual volumetric water content for the selected soil type [-].
    xs_loc : float
        Solid volume fraction for the selected soil type [-].
    x0_loc : float
        Organic matter volume fraction for the selected soil type [-].
    Wvol_prev : float
        Water volume at the previous timestep [m³].
    Wvol_r : float
        Residual water volume at the current timestep [m³].
    Wvol_loss : float
        Water volume lost by drainage at the current timestep [m³].
    Wvol_evap : float
        Water volume lost by evaporation at the current timestep [m³].
    W_content : float
        Volumetric water content at the current timestep [-].
    """

    loss_factor = 0.1
    w_rho = 1000.0
    w_latent = 2250000.0
    SOIL_PARAMS = {
    "sand": {"b1": 0.228, "b2": 2.406, "b3": 4.909, "theta_s": 0.417, "theta_r": 0.020, "xs": 1-0.417, "x0": 0.012},
    "loam": {"b1": 0.310, "b2": 1.534, "b3": 3.222, "theta_s": 0.434, "theta_r": 0.027, "xs": 1-0.434, "x0": 0.018},
    "clay": {"b1": 0.197, "b2": 0.962, "b3": 2.521, "theta_s": 0.385, "theta_r": 0.090, "xs": 1-0.385, "x0": 0.024},
}

    
    def __init__(
        self,
        *,
        water_input: NDArray,
        rho_dry: float,
        soil_type: str,
    ) -> None:

        self.water_input = water_input
        self.rho_dry = rho_dry

        self.Wvol_prev = 0.0
        self.Wvol_r = 0.0
        self.Wvol_loss = 0.0
        self.Wvol_evap = 0.0

        if soil_type not in list(self.SOIL_PARAMS.keys()):
            raise ValueError("Soil  type must match one between Clay, Loam, and Sand soil types.")
        
        self.b1_loc = self.SOIL_PARAMS[soil_type]["b1"]
        self.b2_loc = self.SOIL_PARAMS[soil_type]["b2"]
        self.b3_loc = self.SOIL_PARAMS[soil_type]["b3"]
        self.theta_s_loc = self.SOIL_PARAMS[soil_type]["theta_s"]
        self.theta_r_loc = self.SOIL_PARAMS[soil_type]["theta_r"]
        self.xs_loc = self.SOIL_PARAMS[soil_type]["xs"]
        self.x0_loc = self.SOIL_PARAMS[soil_type]["x0"]

    def _properties_calculation(
        self,
        step: int,
        timesteps: int,
        V: float,
        A: float,
        q: float,
    ) -> Tuple[float, float, float]:
        """
        Compute thermophysical properties from volumetric water content at a given timestep.

        Updates the water volume balance and computes thermal conductivity via
        Chung-Horton (1987), volumetric heat capacity via de Vries (1963), and
        density via mass-weighted average.

        Parameters
        ----------
        step : int
            Current simulation timestep index.
        timesteps : float
            Duration of each timestep [s].
        V : float
            Reference soil volume [m³].
        A : float
            Surface area over which water input is applied [m²].
        q : float
            Thermal power exchanged by the system [W]. Negative in heat extraction,
            positive in heat injection.

        Returns
        -------
        k : float
            Thermal conductivity [W/(m K)].
        cp : float
            Volumetric heat capacity [J/(kg K)].
        rho : float
            Density [kg/m³].
        """

        f_k = lambda b1, b2, b3, wr: b1 + b2 * wr + b3 * wr ** 0.5
        f_cp = lambda xs, x0, wr: 1.92 * 10**6 * xs + 2.51 * 10**6 * x0 + 4.18 * 10**6 * wr
        f_rho = lambda wr, rho_water, rho_dry: wr * rho_water + (1 - wr) * rho_dry

        self.Wvol_loss = self.Wvol_prev * self.loss_factor
        self.Wvol_evap = ((q * timesteps) / self.w_latent) / 1000.0
        self.Wvol_r = np.clip(
            self.Wvol_prev
            + self.water_input[step] * A * timesteps
            - self.Wvol_loss
            - self.Wvol_evap,
            self.theta_r_loc* V,
            self.theta_s_loc * V,
        )

        assert self.Wvol_r >= 0

        self.Wvol_prev = self.Wvol_r

        self.W_content = self.Wvol_r / V

        self.rho = f_rho(self.W_content, self.w_rho, self.rho_dry)
        self.k = f_k(self.b1_loc, self.b2_loc, self.b3_loc, self.W_content)
        self.cp = f_cp(self.xs_loc, self.x0_loc, self.W_content) / self.rho
        

        return self.k, self.cp, self.rho
