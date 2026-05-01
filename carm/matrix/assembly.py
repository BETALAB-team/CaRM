# -*- coding: utf-8 -*-
"""
Global system matrix assembly module.

Assembles the sparse global coefficient matrix and right-hand side vector
for the coupled ground-borehole thermal system at each time step.
"""
import numpy as np
from scipy.sparse import block_diag, coo_matrix

from .MatrixFunctions import (
    build_coefficient_matrix_ground,
    build_rhs_ground,
    build_coefficient_matrix_ground_sup,
    build_rhs_ground_sup,
    build_coefficient_matrix_ground_inf,
    build_rhs_ground_inf,
)
from .BheMatrix import build_bhe_matrix, build_bhe_rhs


def build_global_matrix(model, gr_p, env_params, timesteps, mw_tot_j):

    bh_p = model.borehole

    A_ground_sup = build_coefficient_matrix_ground_sup(
        gr_p,
        env_params,
        timesteps,
    )
    A_ground = build_coefficient_matrix_ground(gr_p, timesteps)
    A_bhe = build_bhe_matrix(model, gr_p, timesteps, mw_tot_j)
    A_ground_inf = build_coefficient_matrix_ground_inf(gr_p, timesteps)

    ng = A_ground.shape[0]
    ngs = A_ground_sup.shape[0]
    nb = A_bhe.shape[0]

    A0 = block_diag((A_ground_sup, A_ground, A_bhe, A_ground_inf), format="csc")

    rows = []
    columns = []
    data = []

    id_axial = getattr(bh_p, "id_shell_middle", bh_p.id_shell)
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # shell off-diagonale sup
    rows.append(ngs - 1)
    columns.append(ngs + ng + id_axial)
    data.append(1 / (0.5 * gr_p.R_sup[-1] + 0.5 * bh_p.R_axial_shell[0, 0]))

    rows.append(ngs + ng + id_axial)
    columns.append(ngs - 1)
    data.append(1 / (0.5 * gr_p.R_sup[-1] + 0.5 * bh_p.R_axial_shell[0, 0]))

    # shell diag
    rows.append(ngs + ng + id_axial)
    columns.append(ngs + ng + id_axial)
    data.append(-1 / (0.5 * gr_p.R_sup[-1] + 0.5 * bh_p.R_axial_shell[0, 0]))

    rows.append(ngs - 1)
    columns.append(ngs - 1)
    data.append(-1 / (0.5 * gr_p.R_sup[-1] + 0.5 * bh_p.R_axial_shell[0, 0]))

    # eq. collegamento shell borehole con interfaccia del terreno off-diagonale radiale
    for j in range(gr_p.m_mesh):
        rows.append(ngs + ng + j * bh_p.n_equations)
        columns.append(ngs + j * gr_p.n_mesh)
        data.append(1 / gr_p.R_ground[j, 0])

        rows.append(ngs + j * gr_p.n_mesh)
        columns.append(ngs + ng + j * bh_p.n_equations)
        data.append(1 / gr_p.R_ground[j, 0])

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # core diag e off
    if bh_p.id_core is not None:
        rows.append(ngs - 1)
        columns.append(ngs + ng + bh_p.id_core)
        data.append(1 / (0.5 * gr_p.R_sup[-1] + 0.5 * bh_p.R_axial_core[0, 0]))

        rows.append(ngs + ng + bh_p.id_core)
        columns.append(ngs - 1)
        data.append(1 / (0.5 * gr_p.R_sup[-1] + 0.5 * bh_p.R_axial_core[0, 0]))

        rows.append(ngs + ng + bh_p.id_core)
        columns.append(ngs + ng + bh_p.id_core)
        data.append(-1 / (0.5 * gr_p.R_sup[-1] + 0.5 * bh_p.R_axial_core[0, 0]))

        rows.append(ngs - 1)
        columns.append(ngs - 1)
        data.append(-1 / (0.5 * gr_p.R_sup[-1] + 0.5 * bh_p.R_axial_core[0, 0]))

    for i in range(gr_p.n_mesh):
        # ultimo sup vs centrale off
        rows.append(ngs - 1)
        columns.append(ngs + i)
        data.append(1 / (0.5 * gr_p.R_sup[-1] + 0.5 * gr_p.R_axial[0, i]))

        rows.append(ngs + i)
        columns.append(ngs - 1)
        data.append(1 / (0.5 * gr_p.R_sup[-1] + 0.5 * gr_p.R_axial[0, i]))

        # diag ultimo sup vs primo centrale
        rows.append(ngs - 1)
        columns.append(ngs - 1)
        data.append(-1 / (0.5 * gr_p.R_sup[-1] + 0.5 * gr_p.R_axial[0, i]))

        rows.append(ngs + i)
        columns.append(ngs + i)
        data.append(-1 / (0.5 * gr_p.R_sup[-1] + 0.5 * gr_p.R_axial[0, i]))

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # off-diagonale
    rows.append(ngs + ng + nb)
    columns.append(ngs + ng + bh_p.n_equations * (bh_p.m_mesh - 1) + id_axial)
    data.append(1 / (0.5 * gr_p.R_inf[0] + 0.5 * bh_p.R_axial_shell[-1, 0]))

    rows.append(ngs + ng + bh_p.n_equations * (bh_p.m_mesh - 1) + id_axial)
    columns.append(ngs + ng + nb)
    data.append(1 / (0.5 * gr_p.R_inf[0] + 0.5 * bh_p.R_axial_shell[-1, 0]))

    # diagonale primo strato inf + contributo shell

    rows.append(ngs + ng + nb)
    columns.append(ngs + ng + nb)
    data.append(-1 / (0.5 * gr_p.R_inf[0] + 0.5 * bh_p.R_axial_shell[-1, 0]))

    # diagonale shell

    rows.append(ngs + ng + bh_p.n_equations * (bh_p.m_mesh - 1) + id_axial)
    columns.append(ngs + ng + bh_p.n_equations * (bh_p.m_mesh - 1) + id_axial)
    data.append(-1 / (0.5 * gr_p.R_inf[0] + 0.5 * bh_p.R_axial_shell[-1, 0]))

    # off-diagonale core

    if bh_p.id_core is not None:

        rows.append(ngs + ng + nb)
        columns.append(ngs + ng + bh_p.n_equations * (bh_p.m_mesh - 1) + bh_p.id_core)
        data.append(1 / (0.5 * gr_p.R_inf[0] + 0.5 * bh_p.R_axial_core[-1, 0]))

        rows.append(ngs + ng + bh_p.n_equations * (bh_p.m_mesh - 1) + bh_p.id_core)
        columns.append(ngs + ng + nb)
        data.append(1 / (0.5 * gr_p.R_inf[0] + 0.5 * bh_p.R_axial_core[-1, 0]))

        # diagonale primo strato inf + contributo core
        rows.append(ngs + ng + nb)
        columns.append(ngs + ng + nb)
        data.append(-1 / (0.5 * gr_p.R_inf[0] + 0.5 * bh_p.R_axial_core[-1, 0]))

        # diagonale ultimo strato core vs primo strato inf
        rows.append(ngs + ng + bh_p.n_equations * (bh_p.m_mesh - 1) + bh_p.id_core)
        columns.append(ngs + ng + bh_p.n_equations * (bh_p.m_mesh - 1) + bh_p.id_core)
        data.append(-1 / (0.5 * gr_p.R_inf[0] + 0.5 * bh_p.R_axial_core[-1, 0]))

    # off-diagonale terreno centrale vs primo strato inferiore
    for i in range(gr_p.n_mesh):  # eq. collegamento inf con terreno centrale
        rows.append(ngs + ng + nb)
        columns.append(ngs + gr_p.n_mesh * (gr_p.m_mesh - 1) + i)
        data.append(1 / (0.5 * gr_p.R_inf[0] + 0.5 * gr_p.R_axial[-1, i]))

        rows.append(ngs + gr_p.n_mesh * (gr_p.m_mesh - 1) + i)
        columns.append(ngs + ng + nb)
        data.append(1 / (0.5 * gr_p.R_inf[0] + 0.5 * gr_p.R_axial[-1, i]))

    # diagonale primo strato inf vs terreno centrale
    for i in range(gr_p.n_mesh):
        rows.append(ngs + ng + nb)
        columns.append(ngs + ng + nb)
        data.append(-1 / (0.5 * gr_p.R_inf[0] + 0.5 * gr_p.R_axial[-1, i]))

        # diagonale nodi ultimo strato terreno centrale vs primo strato inf
        rows.append(ngs + gr_p.n_mesh * (gr_p.m_mesh - 1) + i)
        columns.append(ngs + gr_p.n_mesh * (gr_p.m_mesh - 1) + i)
        data.append(-1 / (0.5 * gr_p.R_inf[0] + 0.5 * gr_p.R_axial[-1, i]))

    A1 = coo_matrix(
        (data, (rows, columns)), shape=(A0.shape[0], A0.shape[0]), dtype=np.float64 #type: ignore
    )
    A1 = A1.tocsc()
    A_tot = A0 + A1

    return A_tot


def build_global_rhs(
    model,
    gr_p,
    env_params,
    timesteps,
    T_old_ground,
    T_old_borehole,
    T_old_ground_sup,
    T_old_ground_inf,
    Ts_old,
    T_ext,
    T_sky,
    T_bc,
    SolarRad,
    mw_tot_j,
    Tf1_j,
):

    x = env_params.envprops.eps * env_params.boltzmann * np.pi * gr_p.rn**2.0
    y = (np.pi * gr_p.rn**2.0) / env_params.envprops.R_ext
    sky_rad = env_params.envprops.absorptance * SolarRad * np.pi * gr_p.rn**2

    b_ground_sup = build_rhs_ground_sup(gr_p, timesteps, T_old_ground_sup)
    b_ground = build_rhs_ground(gr_p, timesteps, T_old_ground, T_bc)
    b_ground_sup_ground = np.concatenate((b_ground_sup, b_ground))
    b_bhe = build_bhe_rhs(model, gr_p, timesteps, T_old_borehole, mw_tot_j, Tf1_j)
    b_ground_bhe = np.concatenate((b_ground_sup_ground, b_bhe))
    b_ground_inf = build_rhs_ground_inf(gr_p, env_params, timesteps, T_old_ground_inf)
    b_tot = np.concatenate((b_ground_bhe, b_ground_inf))

    b_tot[0] = -y * (T_ext) - sky_rad - (T_sky**4.0 - (Ts_old + 273.15) ** 4.0) * x

    return b_tot
