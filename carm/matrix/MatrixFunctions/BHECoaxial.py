# -*- coding: utf-8 -*-
"""
Coaxial BHE matrix assembly module.

Builds the local coefficient matrix and right-hand side vector for the
coaxial pipe BHE configuration, used internally by the global assembler.
"""
import numpy as np
from scipy.sparse import coo_matrix


def build_coefficient_matrix_coaxial(model, gr_p, timesteps, mw_tot_j):

    bh_p = model.borehole
    n = bh_p.n_equations
    m = bh_p.m_mesh
    fluid = model.fluid
    id_shell = bh_p.id_shell
    R_axial_shell = bh_p.R_axial_shell
    R_ground = gr_p.R_ground

    if mw_tot_j == 0:
        R_fluid1 = bh_p.R_pipes1
        R_fluid2i = bh_p.R_pipes2
        R_fluid2e = bh_p.R_pipes2
    else:
        R_fluid1, R_fluid2i, R_fluid2e = bh_p._borehole_resistances(mw_tot_j)

    rows = []
    columns = []
    data = []

    for j in range(m):
        # ----------------------------------------------------------------------
        # HEAT EXCHANGER GROUP OF EQUATIONS
        # ----------------------------------------------------------------------
        rows.extend([j * n + id_shell] * 2)  # shell
        columns.append(j * n + id_shell)
        columns.append(j * n + 2)
        data.append(-1 / bh_p.R_shell[j, 0] - bh_p.C_shell[j, 0] / timesteps - 1 / R_ground[j, 0])
        data.append(1 / bh_p.R_shell[j, 0])

        rows.extend([j * n + 1] * 3)  # pipe 1
        columns.append(j * n + 1)
        columns.append(j * n + 3)
        columns.append(j * n + 4)
        data.append(-1 / (R_fluid1 + bh_p.R_cond1) - 1 / R_fluid2i)
        data.append(1 / (R_fluid1 + bh_p.R_cond1))
        data.append(1 / R_fluid2i)

        rows.extend([j * n + 2] * 3)  # pipe 2
        columns.append(j * n + 0)
        columns.append(j * n + 2)
        columns.append(j * n + 4)
        data.append(1 / bh_p.R_shell[j, 0])
        data.append(-1 / (R_fluid2e + bh_p.R_cond2) - 1 / bh_p.R_shell[j, 0])
        data.append(1 / (R_fluid2e + bh_p.R_cond2))

        rows.extend([j * n + 3] * 2)  # fluid 1
        columns.append(j * n + 1)
        columns.append(j * n + 3)
        data.append(1 / (R_fluid1 + bh_p.R_cond1))
        data.append(
            -mw_tot_j * fluid.cp_w
            - 1 / (R_fluid1 + bh_p.R_cond1)
            - bh_p.C_fluid1 / timesteps
        )

        rows.extend([j * n + 4] * 3)  # fluid 2
        columns.append(j * n + 1)
        columns.append(j * n + 2)
        columns.append(j * n + 4)
        data.append(1 / R_fluid2i)
        data.append(1 / (R_fluid2e + bh_p.R_cond2))
        data.append(
            -mw_tot_j * fluid.cp_w
            - 1 / R_fluid2i
            - 1 / (R_fluid2e + bh_p.R_cond2)
            - bh_p.C_fluid2 / timesteps
        )

        # ----------------------------------------------------------------------
        # AXIAL COUPLING TERMS FOR SHELL
        # ----------------------------------------------------------------------

        t = j * n + id_shell

        # per shell
        rows.append(t)
        columns.append(t)

        if j == 0:
            data.append(
                -1 / (0.5 * R_axial_shell[j, 0] + 0.5 * R_axial_shell[j + 1, 0])
            )  # shell manca termine prec.

            rows.append(t)
            columns.append(t + n)
            data.append(
                1 / (0.5 * R_axial_shell[j + 1, 0] + 0.5 * R_axial_shell[j, 0])
            )  # diag

        elif j == (m - 1):
            data.append(
                -1 / (0.5 * R_axial_shell[j, 0] + 0.5 * R_axial_shell[j - 1, 0])
            )  # shell manca termine succ.

            rows.append(t)
            columns.append(t - n)
            data.append(
                1 / (0.5 * R_axial_shell[j - 1, 0] + 0.5 * R_axial_shell[j, 0])
            )  # diag

        else:
            data.append(
                -1 / (0.5 * R_axial_shell[j, 0] + 0.5 * R_axial_shell[j - 1, 0])
                - 1 / (0.5 * R_axial_shell[j, 0] + 0.5 * R_axial_shell[j + 1, 0])
            )  # shell

            rows.append(t)
            columns.append(t - n)
            data.append(
                1 / (0.5 * R_axial_shell[j - 1, 0] + 0.5 * R_axial_shell[j, 0])
            )  # diag

            rows.append(t)
            columns.append(t + n)
            data.append(
                1 / (0.5 * R_axial_shell[j + 1, 0] + 0.5 * R_axial_shell[j, 0])
            )  # diag

    N = n * m

    # aggiungo termini di portata
    if bh_p.supply_and_return == "1_2":
        for j in range(bh_p.m_mesh):
            if j != 0:
                rows.append(j * n + 3)
                columns.append((j - 1) * n + 3)
                data.append(mw_tot_j * fluid.cp_w)

            if j == bh_p.m_mesh - 1:
                rows.append(j * n + 4)
                columns.append(j * n + 3)
                data.append(mw_tot_j * fluid.cp_w)

            else:
                rows.append(j * n + 4)
                columns.append((j + 1) * n + 4)
                data.append(mw_tot_j * fluid.cp_w)

    elif bh_p.supply_and_return == "2_1":
        for j in range(bh_p.m_mesh):
            if j != 0:
                rows.append(j * n + 4)
                columns.append((j - 1) * n + 4)
                data.append(mw_tot_j * fluid.cp_w)

            if j == bh_p.m_mesh - 1:
                rows.append(j * n + 3)
                columns.append(j * n + 4)
                data.append(mw_tot_j * fluid.cp_w)

            else:
                rows.append(j * n + 3)
                columns.append((j + 1) * n + 3)
                data.append(mw_tot_j * fluid.cp_w)

    A_bhe_coaxial = coo_matrix((data, (rows, columns)), shape=(N, N), dtype=np.float64)
    A_bhe_coaxial = A_bhe_coaxial.tocsc()

    return A_bhe_coaxial


def build_rhs_coaxial(model, gr_p, timesteps, T_old_borehole, mw_tot_j, Tf1_j):

    bh_p = model.borehole
    fluid = model.fluid
    n = bh_p.n_equations

    b_bhe_coaxial = np.zeros((n * bh_p.m_mesh))

    for j in range(bh_p.m_mesh):
        b_bhe_coaxial[j * n + bh_p.id_shell] = (
            -bh_p.C_shell[j, 0] * T_old_borehole[j * n + bh_p.id_shell] / (timesteps)
        )
        b_bhe_coaxial[j * n + 3] = (
            -bh_p.C_fluid1 * T_old_borehole[j * n + 3] / timesteps
        )
        b_bhe_coaxial[j * n + 4] = (
            -bh_p.C_fluid2 * T_old_borehole[j * n + 4] / timesteps
        )

    if mw_tot_j != 0:
        if bh_p.supply_and_return == "1_2":
            b_bhe_coaxial[3] += -mw_tot_j * fluid.cp_w * Tf1_j

        elif bh_p.supply_and_return == "2_1":
            b_bhe_coaxial[4] += -mw_tot_j * fluid.cp_w * Tf1_j

    return b_bhe_coaxial