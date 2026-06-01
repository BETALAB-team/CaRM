# -*- coding: utf-8 -*-
"""
Time-stepping orchestrator for the BHE simulation.

Assembles all physical inputs, initializes the FLS model and boundary
conditions, and runs the simulation via ``run()``. Supports parallel
(independent boreholes) and series (fluid outlet of one borehole feeds
the next) configurations.
"""

from dataclasses import dataclass, field

from numpy.typing import NDArray
from typing import Sequence, Dict

from ..matrix import build_global_matrix, build_global_rhs
from ..external_environment import (
    ExternalEnvironment,
    EnvironmentalProperties,
    EnvironmentalTimeSeries,
)
from ..model import PhysicalModel
from ..state import State
from ..thermal_interference import FiniteLineSolution
from ..initial_conditions import kusuda_achenbach
from ..properties import SoilMoisture

from scipy.sparse.linalg import splu

from pathlib import Path
from datetime import datetime

import time
import copy
import pickle
import numpy as np


@dataclass
class Simulation:
    """
    Time-stepping orchestrator for the BHE simulation.

    Assembles all physical inputs, initializes the FLS model and boundary
    conditions, and runs the simulation via ``run()``. Supports parallel
    (independent boreholes) and series (fluid outlet of one borehole feeds
    the next) configurations.

    Attributes
    ----------
    model : PhysicalModel
        Full physical model of the BHE system.
    envprops : EnvironmentalProperties
        Static radiative and thermal properties of the external environment.
    envinput : EnvironmentalTimeSeries
        Time series of external air temperature and solar irradiance.
    timesteps : float
        Duration of each time step [s].
    n_steps : int
        Total number of simulation time steps.
    mw_tot : NDArray[np.float64]
        Mass flow rate time series, shape (n_bhes, n_steps) or
        (n_groups, n_steps) in series mode [kg/s].
    Tf1 : NDArray[np.float64]
        Inlet fluid temperature time series, shape (n_bhes, n_steps) or
        (n_groups, n_steps) [°C].
        When mw is 0 this value must be set as NaN.

        Example
        -------
        >>> Tf1_5 = np.full((1, 100), 4, dtype = np.float64)
        >>> Tf1_null = np.full((1, 200), np.nan, dtype = np.float64)
        >>> Tf1 = np.concatenate((Tf1_5, Tf1_null), axis = 1)

    fls_mode : str
        FLS simulation mode. Accepts ``'sqrt'`` or ``'continuous'``.
        The first is less expensive; the latter is more accurate.
    groups : dict or None
        Series groups mapping group index to ordered list of borehole indices.
        Required for series mode, ``None`` in parallel mode.
    env : ExternalEnvironment
        Assembled external environment, built at construction time.
    T_sup_kusuda : NDArray[np.float64]
        Kusuda-Achenbach temperature profile for upper ground layers,
        shape (n_steps, m_mesh_sup + 1) [°C].
    T_middle_kusuda : NDArray[np.float64]
        Kusuda-Achenbach temperature profile for middle ground and borehole layers,
        shape (n_steps, m_mesh * (n_mesh + n_equations)) [°C].
    T_inf_kusuda : NDArray[np.float64]
        Kusuda-Achenbach temperature profile for lower ground layers,
        shape (n_steps, m_mesh_inf) [°C].
    T_bc : NDArray[np.float64]
        Far-field boundary condition array, shape (n_steps, n_bhes, m_mesh) [°C].
    T_history : NDArray[np.float64]
        Output temperature history, shape (n_steps + 1, n_bhes, n_dof) [°C].
    fls : FiniteLineSolution or None
        FLS thermal interference model. ``None`` in single-borehole mode.
    bh_p_varprops : SoilMoisture or None
        Soil moisture module for borehole thermophysical properties.
        Instantiated only if ``envinput.water_input`` is not ``None``.
    k_borehole_history : NDArray[np.float64]
        Thermal conductivity history for the borehole, shape (n_steps, n_bhes) [W/(m K)].
        Available only if ``envinput.water_input`` is not ``None``.
    cp_borehole_history : NDArray[np.float64]
        Volumetric heat capacity history for the borehole, shape (n_steps, n_bhes) [J/(m³ K)].
        Available only if ``envinput.water_input`` is not ``None``.
    rho_borehole_history : NDArray[np.float64]
        Density history for the borehole, shape (n_steps, n_bhes) [kg/m³].
        Available only if ``envinput.water_input`` is not ``None``.
    wc_history_borehole : NDArray[np.float64]
        Residual water volume history for the borehole, shape (n_steps, n_bhes) [m³].
        Available only if ``envinput.water_input`` is not ``None``.
    """

    model: PhysicalModel
    envprops: EnvironmentalProperties
    envinput: EnvironmentalTimeSeries
    timesteps: float
    n_steps: int
    mw_tot: NDArray[np.float64]  # shape: (n_bhes, n_steps) | (n_groups, n_steps)
    Tf1: NDArray[np.float64] | None = (
        None  # shape: (n_bhes, n_steps) | (n_groups, n_steps))
    )
    heat_flux: bool = False
    Q_buildings: NDArray[np.float64] | None = None
    T_supply: NDArray[np.float64] | None = None
    fls_mode: str = "sqrt"  # "sqrt" | "continuous"
    groups: Dict | None = None
    T_sup_kusuda: NDArray[np.float64] = field(init=False)
    T_middle_kusuda: NDArray[np.float64] = field(init=False)
    T_inf_kusuda: NDArray[np.float64] = field(init=False)
    env: ExternalEnvironment = field(init=False)
    T_bc: NDArray[np.float64] = field(init=False)  # boundary condition array
    T_history: NDArray[np.float64] = field(init=False)  # output array
    fls: FiniteLineSolution | None = field(default=None, init=False)

    def __post_init__(self):
        # Tf1 calculation mode and field input check
        if self.heat_flux:
            if (self.T_supply is None) or (self.Q_buildings is None):
                raise ValueError(
                    "When heat_flux is set as 'True', Q_building and T_supply must be given as input."
                )
            if self.Tf1 is not None:
                raise ValueError(
                    "When heat_flux is set as 'True', Tf1 must be set as None."
                )
            if self.model.fieldinput is None:
                if (len(self.Q_buildings) != self.n_steps) or (
                    len(self.T_supply) != self.n_steps
                    or (self.mw_tot.shape[1] != self.n_steps)
                ):
                    raise ValueError(
                        "Q_buildings, T_supply, and mw time series must have n_steps"
                    )
            else:
                if self.groups is not None:
                    if self.mw_tot.shape != (len(self.groups), self.n_steps):
                        raise ValueError(
                            "mw_tot time series must have n_bhes rows and n_steps columns"
                        )

                else:
                    if self.mw_tot.shape != (
                        self.model.fieldinput.n_bhes,
                        self.n_steps,
                    ):
                        raise ValueError(
                            "mw_tot time series must have n_bhes rows and n_steps columns"
                        )

        else:
            if (self.T_supply is not None) or (self.Q_buildings is not None):
                raise ValueError(
                    "When heat_flux is set as 'False', Q_building and T_supply must be set as None."
                )
            if self.Tf1 is None:
                raise ValueError(
                    "When heat_flux is set as 'False', Tf1 must be given as input."
                )

            if self.model.fieldinput is None:
                if (self.mw_tot.shape[1] != self.n_steps) or (
                    self.Tf1.shape[1] != self.n_steps
                ):
                    raise ValueError("mw_tot and Tf1 time series must have n_steps")
            else:
                if self.groups is not None:
                    if self.mw_tot.shape != (
                        len(self.groups),
                        self.n_steps,
                    ) or self.Tf1.shape != (len(self.groups), self.n_steps):
                        raise ValueError(
                            "mw_tot and Tf1 time series must have n_bhes rows and n_steps columns"
                        )

                else:
                    if self.mw_tot.shape != (
                        self.model.fieldinput.n_bhes,
                        self.n_steps,
                    ) or self.Tf1.shape != (self.model.fieldinput.n_bhes, self.n_steps):
                        raise ValueError(
                            "mw_tot and Tf1 time series must have n_bhes rows and n_steps columns"
                        )

        # environmental input check
        if (
            len(self.envinput.T_ext) < self.n_steps
            or len(self.envinput.SolarRad) < self.n_steps
        ):
            raise ValueError(
                "T_ext and SolarRad time series have less than n_steps values"
            )
        if (
            len(self.envinput.T_ext) > self.n_steps
            or len(self.envinput.SolarRad) > self.n_steps
        ):
            print(
                "T_ext and SolarRad lengths > n_steps, the first n_steps values are used"
            )

        if self.envinput.water_input is not None:
            water_input = self.envinput.water_input
            if len(water_input) < self.n_steps:
                raise ValueError("water_input time series has less than n_steps values")

            self.bh_p_varprops = SoilMoisture(
                water_input=water_input,
                rho_dry=np.mean(self.model.borehole.rho_0),
                soil_type=self.model.borehole.soil_type,
            )

            self.k_borehole_history = np.empty(
                (self.n_steps, len(self.model.ground)), dtype=np.float64
            )
            self.cp_borehole_history = copy.deepcopy(self.k_borehole_history)
            self.rho_borehole_history = copy.deepcopy(self.k_borehole_history)

            self.wc_history_borehole = copy.deepcopy(self.k_borehole_history)

        # to properly build ground rhs term
        self.adiabatic = False

        if self.model.fieldinput is not None:
            if self.model.fieldinput.layout == "regular":
                self.adiabatic = True

        self.env = ExternalEnvironment(envprops=self.envprops, envinput=self.envinput)
        self.T_sup_kusuda, self.T_middle_kusuda, self.T_inf_kusuda = kusuda_achenbach(
            self.model.ground[0],
            self.model.borehole,
            self.envprops,
            self.envinput.Tm,
            self.timesteps,
            self.n_steps,
        )
        self._init_fls()
        self._boundary_condition()

    def _boundary_condition(self) -> None:
        self.T_bc = self.T_middle_kusuda[
            :,
            [
                j * self.model.ground[0].n_mesh
                for j in range(self.model.ground[0].m_mesh)
            ],
        ]
        self.T_bc = np.repeat(self.T_bc[:, None, :], len(self.model.ground), axis=1)

    def _init_fls(self) -> None:
        fls_modes = {"sqrt", "continuous"}
        if self.fls_mode not in fls_modes:
            raise ValueError(
                "fls_mode must be sqrt or continuous, see documentations to choose the one you prefer"
            )

        time_hist = np.arange(
            self.timesteps,
            self.timesteps * (self.n_steps + 1),
            self.timesteps,
            dtype=np.float64,
        )  # dt array: (dt, 2 dt, 3 dt, 4 dt, ...., (n_steps + 1) * dt)

        if len(self.model.ground) > 1 and self.model.fieldinput.layout == "irregular":
            self.fls = FiniteLineSolution(
                physicalmodel=self.model,
                n_steps=self.n_steps,
                time_hist=time_hist,
                fls_mode=self.fls_mode,
            )

    def run(
        self, parallel: bool | None = None, series: bool | None = None
    ) -> NDArray[np.float64]:
        """
        Run the simulation in parallel or series mode.

        Dispatches to ``_run_parallel()`` or ``_run_series()`` based on the
        provided flags. For single-borehole configurations, ``parallel`` and
        ``series`` must both be ``None``.

        Parameters
        ----------
        parallel : bool or None
            Set to ``True`` to run in parallel mode (independent boreholes).
        series : bool or None
            Set to ``True`` to run in series mode (fluid outlet chaining).

        Returns
        -------
        NDArray[np.float64]
            Temperature history array of shape (n_steps + 1, n_bhes, n_dof).

        Raises
        ------
        ValueError
            If both or neither flags are set, or if series groups are not defined.

        Examples
        --------
        >>> T_hist = sim.run(parallel=True)
        >>> T_hist.shape
        (n_steps + 1, n_bhes, n_dof)
        """

        if len(self.model.ground) > 1:

            if parallel is True and series is None:
                return self._run_parallel()

            elif parallel is None and series is True:
                nodes = sorted(list(self.model.field._borehole_graph))
                neighbors = {
                    b: list(self.model.field._borehole_graph.neighbors(b))
                    for b in nodes
                }

                if self.groups is None:
                    raise ValueError("Define series groups to put in parallel")

                for group in self.groups.values():
                    if not all(b in neighbors[a] for a, b in zip(group, group[1:])):
                        raise ValueError("Some borehole can't be connected in series")

                return self._run_series()

            else:
                raise ValueError(
                    "Invalid configuration: set exactly one of parallel or series to True"
                )
        else:
            if parallel is not None and series is not None:
                raise ValueError(
                    "Series and Parallel must be set as None when running single borehole configuration"
                )
            else:
                return self._run_parallel()

    def _run_parallel(self) -> NDArray[np.float64]:
        tic = time.time()  # start simulation

        model = self.model
        ground = model.ground[0]  # alias for ground
        borehole = model.borehole  # alias for borehole
        n = len(model.ground)  # number of boreholes
        ns = ground.m_mesh_sup + 1  # upper ground layers
        nm = ground.m_mesh * ground.n_mesh  # middle ground layers
        nb = borehole.m_mesh * borehole.n_equations  # middle borehole layers
        ninf = ground.m_mesh_inf  # lower ground layers

        T0_matrix = np.zeros(
            (n, (ns + nm + nb + ninf)), dtype=np.float64
        )  # Starting condition initialization

        self.T_history = np.empty(
            (self.n_steps + 1, n, (ns + nm + nb + ninf)),
            dtype=np.float64,
        )  # T_history initialization array, n_steps x n_bhes x total mesh

        self.q_nbhes = np.zeros(
            (self.n_steps, n), dtype=np.float64
        )  # Borehole heat flux array, n_steps x n_bhes

        Tstartsup = self.T_sup_kusuda[0, :].ravel()
        Tstartmiddle = self.T_middle_kusuda[0, :].ravel()
        Tstartinf = self.T_inf_kusuda[0, :].ravel()

        T0 = np.concatenate((Tstartsup, Tstartmiddle, Tstartinf))
        T0_matrix[:] = T0  # broadcasting on number of borehole axis

        currstate = State(T0_matrix)
        self.T_history[0] = T0_matrix.copy()

        self.A = [0] * n  # type: ignore
        self.A_inv = copy.deepcopy(self.A)  # type: ignore

        for step in range(self.n_steps):
            # external environment aliasing
            T_ext = self.envinput.T_ext[step]
            T_sky = self.env.T_sky[step]
            SolarRad = self.env.envinput.SolarRad[step]

            currstate.save_old()

            idx_null = np.where(self.mw_tot[:, step] == 0)[0]
            if len(idx_null) > 0:
                self.Tf1[idx_null, step] = currstate.T_old[
                    idx_null, ns + nm + (borehole.id_inlet)
                ]

            Tfout = currstate.T_old[:, ns + nm + borehole.id_outlet]

            T_new_step = np.zeros((n, (ns + nm + nb + ninf)))

            if step != 0:
                qfluid = (
                    self.mw_tot[:, step]
                    * model.fluid.cp_w
                    * (self.Tf1[:, step] - Tfout)
                )
                self.q_nbhes[step] = qfluid

            if self.fls is not None:
                self.T_bc[step] = self.T_bc[step] + self.fls._compute_delta_t(
                    q_nbhes=self.q_nbhes, step=step
                )

            for j, gr_p in enumerate(model.ground):
                properties_changed = False

                if self.envinput.water_input is not None:
                    tol = 1e-3
                    k_bh, cp_bh, rho_bh = self._props_calculation(
                        step=step, borehole=borehole, j=j
                    )

                    if (
                        (abs(np.mean(borehole.k0) - k_bh) > tol)
                        or (abs(np.mean(borehole.cp_0) - cp_bh) > tol)
                        or (abs(np.mean(borehole.rho_0) - rho_bh) > tol)
                    ):

                        properties_changed = True
                        borehole._update_properties(k_bh, cp_bh, rho_bh)

                (
                    T_borehole_old,
                    T_ground_old,
                    T_ground_sup_old,
                    T_ground_inf_old,
                    Ts_old,
                ) = self.model._get_temperatures(currstate, j)

                if (
                    step == 0
                    or (self.mw_tot[j, step] != self.mw_tot[j, step - 1])
                    or (properties_changed == True)
                ):
                    self.A[j] = build_global_matrix(
                        self.model,
                        gr_p,
                        self.env,
                        self.timesteps,
                        self.mw_tot[j, step],
                        self.adiabatic,
                    )
                    self.A_inv[j] = splu(self.A[j])

                self.b = build_global_rhs(
                    self.model,
                    gr_p,
                    self.env,
                    self.timesteps,
                    T_ground_old,
                    T_borehole_old,
                    T_ground_sup_old,
                    T_ground_inf_old,
                    Ts_old,
                    T_ext,
                    T_sky,
                    self.T_bc[step, j],
                    SolarRad,
                    self.mw_tot[j, step],
                    self.Tf1[j, step],
                    self.adiabatic,
                )

                assert np.all(np.isfinite(self.A[j].data)), "A contains NaN or Inf"  # type: ignore
                assert np.all(np.isfinite(self.b)), "b contains NaN or Inf"  # type: ignore

                T_new_j = self.A_inv[j].solve(self.b)  # type: ignore

                assert np.all(np.isfinite(T_new_j)), "T_new is not finite"

                r = self.A[j] @ T_new_j - self.b
                assert np.max(np.abs(r)) < 1e-6

                T_new_step[j, :] = T_new_j

            currstate.update(T_new_step)
            self.T_history[step + 1] = currstate.T_state.copy()

        toc = time.time()

        print(f"Simulation loop running time: {(toc - tic)} seconds")

        self._save_results()

        return self.T_history

    def _run_series(self) -> NDArray[np.float64]:
        tic = time.time()  # start simulation

        model = self.model
        ground = model.ground[0]  # alias for ground
        borehole = model.borehole  # alias for borehole
        n = len(model.ground)  # number of boreholes
        ns = ground.m_mesh_sup + 1  # upper ground layers
        nm = ground.m_mesh * ground.n_mesh  # middle ground layers
        nb = borehole.m_mesh * borehole.n_equations  # middle borehole layers
        ninf = ground.m_mesh_inf  # lower ground layers

        T0_matrix = np.zeros(
            (n, (ns + nm + nb + ninf)), dtype=np.float64
        )  # Starting condition initialization

        self.T_history = np.empty(
            (self.n_steps + 1, n, (ns + nm + nb + ninf)),
            dtype=np.float64,
        )  # T_history initialization array, n_steps x n_bhes x total mesh

        qfluid = np.zeros((self.n_steps, n), dtype=np.float64)  # fluid heat flux array

        self.q_nbhes = copy.deepcopy(
            qfluid
        )  # Borehole heat flux array, n_steps x n_bhes

        Tstartsup = self.T_sup_kusuda[0, :].ravel()
        Tstartmiddle = self.T_middle_kusuda[0, :].ravel()
        Tstartinf = self.T_inf_kusuda[0, :].ravel()

        T0 = np.concatenate((Tstartsup, Tstartmiddle, Tstartinf))
        T0_matrix[:] = T0  # broadcasting on number of borehole axis

        currstate = State(T0_matrix)
        self.T_history[0] = T0_matrix.copy()

        self.A = [0] * n  # type: ignore
        self.A_inv = copy.deepcopy(self.A)  # type: ignore

        for step in range(self.n_steps):
            # external environment aliasing
            T_ext = self.envinput.T_ext[step]
            T_sky = self.env.T_sky[step]
            SolarRad = self.env.envinput.SolarRad[step]

            currstate.save_old()

            T_new_step = np.zeros((n, (ns + nm + nb + ninf)))

            if step != 0:
                self.q_nbhes[step] = qfluid[step - 1]

            if self.fls is not None:
                self.T_bc[step] = self.T_bc[step] + self.fls._compute_delta_t(
                    q_nbhes=self.q_nbhes, step=step
                )

            idx_null = np.where(self.mw_tot[:, step] == 0)[0]
            if len(idx_null) > 0:
                self.Tf1[idx_null, step] = currstate.T_old[
                    idx_null, ns + nm + (borehole.id_inlet)
                ]

            for i, group in enumerate(self.groups.values()):

                mw_loc = self.mw_tot[i, step]
                Tf1_loc = self.Tf1[i, step]
                for j in group:
                    gr_p = self.model.ground[j]

                    properties_changed = False

                    if self.envinput.water_input is not None:
                        tol = 1e-3
                        k_bh, cp_bh, rho_bh = (
                            self._props_calculation(
                                step=step, borehole=borehole, j=j
                            )
                        )

                        if (
                            (abs(np.mean(borehole.k0) - k_bh) > tol)
                            or (abs(np.mean(borehole.cp_0) - cp_bh) > tol)
                            or (abs(np.mean(borehole.rho_0) - rho_bh) > tol)
                        ):

                            properties_changed = True
                            borehole._update_properties(k_bh, cp_bh, rho_bh)

                    (
                        T_borehole_old,
                        T_ground_old,
                        T_ground_sup_old,
                        T_ground_inf_old,
                        Ts_old,
                    ) = self.model._get_temperatures(currstate, j)

                    if (
                        (step == 0)
                        or (mw_loc != self.mw_tot[i, step - 1])
                        or (properties_changed == True)
                    ):
                        self.A[j] = build_global_matrix(
                            self.model, gr_p, self.env, self.timesteps, mw_loc
                        )
                        self.A_inv[j] = splu(self.A[j])

                    b = build_global_rhs(
                        self.model,
                        gr_p,
                        self.env,
                        self.timesteps,
                        T_ground_old,
                        T_borehole_old,
                        T_ground_sup_old,
                        T_ground_inf_old,
                        Ts_old,
                        T_ext,
                        T_sky,
                        self.T_bc[step, j],
                        SolarRad,
                        mw_loc,
                        Tf1_loc,
                        self.adiabatic,
                    )

                    assert np.all(np.isfinite(self.A[j].data)), "A contains NaN or Inf"  # type: ignore
                    assert np.all(np.isfinite(b)), "b contains NaN or Inf"  # type: ignore

                    T_new_i_j = self.A_inv[j].solve(b)  # type: ignore

                    assert np.all(np.isfinite(T_new_i_j)), "T_new is not finite"

                    r = self.A[j] @ T_new_i_j - b
                    assert np.max(np.abs(r)) < 1e-6

                    Tfout = T_new_i_j[ns + nm + borehole.id_outlet]
                    qfluid[step, j] = mw_loc * model.fluid.cp_w * (Tf1_loc - Tfout)

                    Tf1_loc = T_new_i_j[ns + nm + (borehole.n_equations - 1)]

                    T_new_step[j, :] = T_new_i_j

            currstate.update(T_new_step)
            self.T_history[step + 1] = currstate.T_state.copy()

        toc = time.time()

        print(f"Simulation loop running time: {(toc - tic)} seconds")

        self._save_results()

        return self.T_history

    def _props_calculation(self, step: int, borehole, j):

        k_bh, cp_bh, rho_bh = self.bh_p_varprops._properties_calculation(
            step=step,
            timesteps=self.timesteps,
            V=np.pi * (borehole.D0**2) * borehole.Lbore,
            A=np.pi * (borehole.D0**2) / 4.0,
            q=self.q_nbhes[step, j],
        )

        self.k_borehole_history[step, j] = k_bh
        self.cp_borehole_history[step, j] = cp_bh
        self.rho_borehole_history[step, j] = rho_bh

        self.wc_history_borehole[step, j] = self.bh_p_varprops.Wvol_r

        return k_bh, cp_bh, rho_bh

    def _save_results(self) -> None:
        to_save = {
            "T_history": self.T_history,
            "T_f1": self.Tf1,
            "T_bc": self.T_bc,
            "rn_list": [self.model.ground[i].rn for i in range(len(self.model.ground))],
            "n_steps": self.n_steps,
            "timesteps": self.timesteps,
        }

        if self.envinput.water_input is not None:
            to_save["k_borehole"] = self.k_borehole_history
            to_save["cp_borehole"] = self.cp_borehole_history
            to_save["rho_borehole"] = self.rho_borehole_history
            to_save["water_content_borehole"] = self.wc_history_borehole

        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"CaRM_results_{timestamp}.pkl"

        with open(output_path, "wb") as f:
            pickle.dump(to_save, f)

        print(f"Results saved to {output_path}")
