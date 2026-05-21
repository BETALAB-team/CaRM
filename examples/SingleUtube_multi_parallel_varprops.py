# -*- coding: utf-8 -*-
"""
Example: Parallel configuration with single U-tube boreholes.

Annual simulation with seasonal operation profile and variable soil moisture.
"""

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from carm import (
    BoreholeGeometry,
    BoreholeMesh,
    BoreholeThermalProperties,
    SingleUtube,
)
from carm import EnvironmentalProperties, EnvironmentalTimeSeries
from carm import FieldInput
from carm import Fluid
from carm import GroundGeometry, GroundMesh
from carm import PhysicalModel
from carm import Simulation


def main():

    # -------------------------------------------------------------------------
    # Input parameters
    # -------------------------------------------------------------------------

    BASE_DIR = Path(__file__).parent

    field_path = BASE_DIR / "spacing.xlsx"
    path       = BASE_DIR / "input_env.xlsx"

    n_bhes = 9
    x_min, y_min = -2.5, -2.5
    x_max, y_max = 12.5, 12.5

    stratification = [(1.8, 947.37, 1762.0, 111)]  # rho = 110 * 16.01846
    n_mesh     = 20
    m_mesh     = 40
    Tg         = 13
    L          = 100
    m_mesh_sup = 4
    m_mesh_inf = 40
    L_sup      = 1
    L_inf      = 10

    k_w   = 0.568709114496803
    rho_w = 1000.1435933169
    cp_w  = 4207.40834247225
    ni_w  = 1.49626063208248e-6

    Dpi          = 0.026
    Lbore        = 100
    D0           = 0.15
    Rp0          = 0.25
    RppB         = 0.72
    n_pipes      = 2
    pipe_thick   = 0.003
    pipe_spacing = 0.0823
    cp_0  = 921.0   # 0.22 * 4186.8
    rho_0 = 1762.0  # 110 * 16.01846
    k0    = 1.8
    porosity     = 0.3

    absorptance = 0.7
    eps         = 0.95
    At          = 10
    tau         = 0
    tau_y       = 365 * 24 * 3600
    tau_shift   = 210 * 24 * 3600
    R_ext       = 0.04

    Tm     = 13
    dt     = 3600
    n_steps = 8760

    # -------------------------------------------------------------------------
    # Operational profile
    # -------------------------------------------------------------------------

    hours       = np.arange(n_steps)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    winter = (day_of_year < 90) | (day_of_year >= 274)   # oct-mar
    summer = (day_of_year >= 152) & (day_of_year < 244)   # jun-aug
    # may (day 120-151) and shoulder: off

    winter_active = winter & (hour_of_day >= 6)  & (hour_of_day < 22)
    summer_active = summer & (hour_of_day >= 10) & (hour_of_day < 18)
    active        = winter_active | summer_active

    Tf1    = np.full((n_bhes, n_steps), np.nan,  dtype=np.float64)
    mw_tot = np.zeros((n_bhes, n_steps),          dtype=np.float64)

    Tf1[:, winter_active] = 2.0
    Tf1[:, summer_active] = 35.0
    mw_tot[:, active]     = 0.1657

    # -------------------------------------------------------------------------
    # Water input signal: zero -> fixed value for 3000 h -> zero
    # -------------------------------------------------------------------------

    water_input = np.zeros(n_steps, dtype=np.float64)
    water_input[2000:5000] = 1e-3

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------

    myfield = FieldInput(
        n_bhes=n_bhes, xmin=x_min, ymin=y_min,
        xmax=x_max, ymax=y_max, rb=D0 / 2.0,
    )
    myfield.from_excel(field_path)

    fluid = Fluid(k_w=k_w, rho_w=rho_w, cp_w=cp_w, ni_w=ni_w)

    bore_geom     = BoreholeGeometry(Lbore=Lbore, D0=D0)
    bore_mesh     = BoreholeMesh(m_mesh=m_mesh)
    bore_th_props = BoreholeThermalProperties(
        porosity=porosity, cp_0=cp_0, rho_0=rho_0, k0=k0,
    )
    props_b = SingleUtube(
        geom=bore_geom, mesh=bore_mesh, thermalprops=bore_th_props, fluid=fluid,
        Rp0=Rp0, RppB=RppB, pipe_spacing=pipe_spacing,
        pipe_thick=pipe_thick, Dpi=Dpi, n_pipes=n_pipes,
    )

    ground_geom = GroundGeometry(D0=D0, L=L, L_sup=L_sup, L_inf=L_inf, rn=None)
    ground_mesh = GroundMesh(
        n_mesh=n_mesh, m_mesh=m_mesh,
        m_mesh_sup=m_mesh_sup, m_mesh_inf=m_mesh_inf,
    )

    env_input_base = EnvironmentalTimeSeries.from_excel(Tm=Tm, path=path)
    env_input = EnvironmentalTimeSeries.from_array(
        Tm=Tm,
        T_ext=env_input_base.T_ext,
        SolarRad=env_input_base.SolarRad,
        water_input=water_input,
    )
    env_props = EnvironmentalProperties(
        R_ext=R_ext, absorptance=absorptance, eps=eps,
        At=At, tau=tau, tau_y=tau_y, tau_shift=tau_shift,
    )

    model = PhysicalModel(
        ground_geom=ground_geom, ground_mesh=ground_mesh,
        borehole=props_b, fluid=fluid,
        Tg=Tg, porosity=porosity, stratification=stratification,
        fieldinput=myfield,
    )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    simulation = Simulation(
        model=model, envinput=env_input, timesteps=dt, n_steps=n_steps,
        envprops=env_props, mw_tot=mw_tot, Tf1=Tf1,
    )
    T_history = simulation.run(parallel=True)

    # -------------------------------------------------------------------------
    # Post-processing indices
    # -------------------------------------------------------------------------

    nsup    = m_mesh_sup + 1
    nground = n_mesh * m_mesh
    rb      = reference_borehole = 0

    time  = np.arange(dt, dt * (n_steps + 1), dt, dtype=np.float64)
    depth = np.arange(-L_sup, -L_sup - model.ground[0].dz * m_mesh, -model.ground[0].dz)

    slice_shell = [nsup + nground + j * props_b.n_equations for j in range(m_mesh)]
    slice_down  = [nsup + nground + j * props_b.n_equations + (props_b.n_equations - 2) for j in range(m_mesh)]
    slice_up    = [nsup + nground + j * props_b.n_equations + (props_b.n_equations - 1) for j in range(m_mesh)]

    steps = [1000, 4000, 8760]

    # -------------------------------------------------------------------------
    # Plot: operational profile
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    axes[0].plot(time / 3600, mw_tot[rb], color="tab:blue", linewidth=0.5)
    axes[0].set_ylabel("mw [kg/s]")
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    axes[1].plot(time / 3600, np.where(np.isnan(Tf1[rb]), np.nan, Tf1[rb]), color="tab:orange", linewidth=0.5)
    axes[1].set_ylabel("Tf1 [°C]")
    axes[1].set_xlabel("Time [h]")
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    fig.suptitle("Operational profile")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: water input signal
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(time / 3600, water_input * 1e3, color="tab:blue")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Water input [×10⁻³ m/s]")
    ax.set_title("Water input signal")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: water content over time
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    fig.suptitle("Water volume over time")

    axes[0].plot(time / 3600, simulation.wc_history_ground[:, rb], color="tab:blue")
    axes[0].set_xlabel("Time [h]")
    axes[0].set_ylabel("Wvol_r [m³]")
    axes[0].set_title("Ground")
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    axes[1].plot(time / 3600, simulation.wc_history_borehole[:, rb], color="tab:orange")
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel("Wvol_r [m³]")
    axes[1].set_title("Borehole")
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: outlet fluid temperature over time
    # -------------------------------------------------------------------------

    Tfout = T_history[1:, rb, nsup + nground + (props_b.n_equations - 1)]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(time / 3600, Tfout)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"Outlet fluid temperature - Borehole {rb}")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: shell temperature vertical profile
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(4, 4))
    for s in steps:
        ax.plot(T_history[s, rb, slice_shell], depth, label=f'Step = {s}')
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Depth [m]")
    ax.legend(fontsize=7)
    ax.set_title(f"Shell temperature - Borehole {rb}")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: fluid temperature vertical profile
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(4, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for s, c in zip(steps, colors):
        ax.plot(T_history[s, rb, slice_down], depth, color=c, linestyle="-",  label=f"Step = {s}")
        ax.plot(T_history[s, rb, slice_up],   depth, color=c, linestyle="--")
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Depth [m]")
    ax.legend(fontsize=7)
    ax.set_title(f"Fluid temperature - Borehole {rb}")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: ground thermal properties vs water content
    # -------------------------------------------------------------------------

    wc_gr = simulation.wc_history_ground[:, rb]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Ground thermal properties vs water content - Borehole {rb}")

    axes[0].scatter(wc_gr, simulation.k_ground_history[:, rb],   s=1, color="tab:blue")
    axes[0].set_xlabel("Wvol_r [m³]")
    axes[0].set_ylabel("k [W/(m K)]")
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    axes[1].scatter(wc_gr, simulation.cp_ground_history[:, rb],  s=1, color="tab:orange")
    axes[1].set_xlabel("Wvol_r [m³]")
    axes[1].set_ylabel("cp [J/(kg K)]")
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    axes[2].scatter(wc_gr, simulation.rho_ground_history[:, rb], s=1, color="tab:green")
    axes[2].set_xlabel("Wvol_r [m³]")
    axes[2].set_ylabel("ρ [kg/m³]")
    axes[2].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: borehole thermal properties vs water content
    # -------------------------------------------------------------------------

    wc_bh = simulation.wc_history_borehole[:, rb]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Borehole thermal properties vs water content - Borehole {rb}")

    axes[0].scatter(wc_bh, simulation.k_borehole_history[:, rb],   s=1, color="tab:blue")
    axes[0].set_xlabel("Wvol_r [m³]")
    axes[0].set_ylabel("k [W/(m K)]")
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    axes[1].scatter(wc_bh, simulation.cp_borehole_history[:, rb],  s=1, color="tab:orange")
    axes[1].set_xlabel("Wvol_r [m³]")
    axes[1].set_ylabel("cp [J/(kg K)]")
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    axes[2].scatter(wc_bh, simulation.rho_borehole_history[:, rb], s=1, color="tab:green")
    axes[2].set_xlabel("Wvol_r [m³]")
    axes[2].set_ylabel("ρ [kg/m³]")
    axes[2].grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: ground temperature heatmap
    # -------------------------------------------------------------------------

    r0 = model.ground[0].r0
    rn = model.ground[0].rn
    dz = model.ground[0].dz

    T_all   = np.array([T_history[steps, rb, nsup:nsup + nground]]).reshape(len(steps), m_mesh, n_mesh)
    vmin, vmax = T_all.min(), T_all.max()
    x_ticks = np.round(np.linspace(r0, rn, 5), decimals=2)
    radius  = np.linspace(D0 / 2, rn, n_mesh)
    R, D    = np.meshgrid(radius, depth)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for i, (s, ax) in enumerate(zip(steps, axes)):
        pc = ax.pcolormesh(R, D, T_all[i], cmap='RdYlGn_r', shading='gouraud', vmin=vmin, vmax=vmax)
        ax.set_xlabel("Radius [m]")
        ax.set_ylabel("Depth [m]")
        fig.colorbar(pc, ax=ax, label="Temperature [°C]")
        ax.set_title(f"Step {s}")
        ax.set_xlim((r0, rn))
        ax.set_xticks(x_ticks)
    fig.suptitle(f"Ground temperature - BHE {rb}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()