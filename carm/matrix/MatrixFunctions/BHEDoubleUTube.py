# -*- coding: utf-8 -*-
"""
Double U-tube BHE matrix assembly module.

Builds the local coefficient matrix and right-hand side vector for the
double U-tube BHE configuration, used internally by the global assembler.
"""
import numpy as np
from scipy.sparse import coo_matrix


def build_coefficient_matrix_bhe_double_u_tube(model, gr_p, timesteps, mw_tot_j):

    bh_p = model.borehole
    fluid = model.fluid
    m = bh_p.m_mesh
    n = bh_p.n_equations
    id_core = bh_p.id_core
    id_shell = bh_p.id_shell
    R_axial_core = bh_p.R_axial_core
    R_axial_shell = bh_p.R_axial_shell
    R_ground = gr_p.R_ground

    if mw_tot_j == 0:
        R_fluid = bh_p.R_pipes
    else:
        R_fluid = bh_p._borehole_resistances(mw_tot_j)

    if model.borehole.connection in {"P", "p"}:
        mw_tot_j = mw_tot_j / 2.0

    rows = []
    columns = []
    data = []

    for j in range(m):
        # ----------------------------------------------------------------------
        # HEAT EXCHANGER GROUP OF EQUATIONS
        # ----------------------------------------------------------------------
        rows.extend([j * n + id_shell] * 5)  # shell
        columns.append(j * n + id_shell)
        columns.append(j * n + 2)
        columns.append(j * n + 3)
        columns.append(j * n + 4)
        columns.append(j * n + 5)
        data.append(
            -4 / bh_p.Rp0_dz - bh_p.C_shell[j, 0] / timesteps - 1 / R_ground[j, 0]
        )
        data.append(1 / (bh_p.Rp0_dz))
        data.append(1 / (bh_p.Rp0_dz))
        data.append(1 / (bh_p.Rp0_dz))
        data.append(1 / (bh_p.Rp0_dz))

        rows.extend([j * n + id_core] * 5)  # core
        columns.append(j * n + id_core)
        columns.append(j * n + 2)
        columns.append(j * n + 3)
        columns.append(j * n + 4)
        columns.append(j * n + 5)
        data.append(-4 / (0.5 * bh_p.RppB_dz) - bh_p.C_core[j, 0] / timesteps)
        data.append(1 / (0.5 * bh_p.RppB_dz))
        data.append(1 / (0.5 * bh_p.RppB_dz))
        data.append(1 / (0.5 * bh_p.RppB_dz))
        data.append(1 / (0.5 * bh_p.RppB_dz))

        rows.extend([j * n + 2] * 6)  # pipe 1
        columns.append(j * n + 0)
        columns.append(j * n + 3)
        columns.append(j * n + 5)
        columns.append(j * n + 1)
        columns.append(j * n + 2)
        columns.append(j * n + 6)
        data.append(1 / bh_p.Rp0_dz)
        data.append(1 / bh_p.RppA_dz)
        data.append(1 / bh_p.RppA_dz)
        data.append(1 / (0.5 * bh_p.RppB_dz))
        data.append(
            -1 / bh_p.Rp0_dz - 2 / bh_p.RppA_dz - 1 / (0.5 * bh_p.RppB_dz) - 1 / R_fluid
        )
        data.append(1 / R_fluid)

        rows.extend([j * n + 3] * 6)  # pipe 2
        columns.append(j * n + 0)
        columns.append(j * n + 2)
        columns.append(j * n + 4)
        columns.append(j * n + 1)
        columns.append(j * n + 3)
        columns.append(j * n + 7)
        data.append(1 / bh_p.Rp0_dz)
        data.append(1 / bh_p.RppA_dz)
        data.append(1 / bh_p.RppA_dz)
        data.append(1 / (0.5 * bh_p.RppB_dz))
        data.append(
            -1 / bh_p.Rp0_dz - 2 / bh_p.RppA_dz - 1 / (0.5 * bh_p.RppB_dz) - 1 / R_fluid
        )
        data.append(1 / R_fluid)

        rows.extend([j * n + 4] * 6)  # pipe 3
        columns.append(j * n + 0)
        columns.append(j * n + 3)
        columns.append(j * n + 5)
        columns.append(j * n + 1)
        columns.append(j * n + 4)
        columns.append(j * n + 8)
        data.append(1 / bh_p.Rp0_dz)
        data.append(1 / bh_p.RppA_dz)
        data.append(1 / bh_p.RppA_dz)
        data.append(1 / (0.5 * bh_p.RppB_dz))
        data.append(
            -1 / bh_p.Rp0_dz - 2 / bh_p.RppA_dz - 1 / (0.5 * bh_p.RppB_dz) - 1 / R_fluid
        )
        data.append(1 / R_fluid)

        rows.extend([j * n + 5] * 6)  # pipe 4
        columns.append(j * n + 0)
        columns.append(j * n + 4)
        columns.append(j * n + 2)
        columns.append(j * n + 1)
        columns.append(j * n + 5)
        columns.append(j * n + 9)
        data.append(1 / bh_p.Rp0_dz)
        data.append(1 / bh_p.RppA_dz)
        data.append(1 / bh_p.RppA_dz)
        data.append(1 / (0.5 * bh_p.RppB_dz))
        data.append(
            -1 / bh_p.Rp0_dz - 2 / bh_p.RppA_dz - 1 / (0.5 * bh_p.RppB_dz) - 1 / R_fluid
        )
        data.append(1 / R_fluid)

        rows.extend([j * n + 6] * 2)
        columns.append(j * n + 2)
        columns.append(j * n + 6)
        data.append(1 / R_fluid)
        data.append(-mw_tot_j * fluid.cp_w - 1 / R_fluid - bh_p.C_fluid / timesteps)

        rows.extend([j * n + 7] * 2)
        columns.append(j * n + 3)
        columns.append(j * n + 7)
        data.append(1 / R_fluid)
        data.append(-mw_tot_j * fluid.cp_w - 1 / R_fluid - bh_p.C_fluid / timesteps)

        rows.extend([j * n + 8] * 2)
        columns.append(j * n + 4)
        columns.append(j * n + 8)
        data.append(1 / R_fluid)
        data.append(-mw_tot_j * fluid.cp_w - 1 / R_fluid - bh_p.C_fluid / timesteps)

        rows.extend([j * n + 9] * 2)
        columns.append(j * n + 5)
        columns.append(j * n + 9)
        data.append(1 / R_fluid)
        data.append(-mw_tot_j * fluid.cp_w - 1 / R_fluid - bh_p.C_fluid / timesteps)

        # ----------------------------------------------------------------------
        # AXIAL COUPLING TERMS FOR SHELL AND CORE
        # ----------------------------------------------------------------------

        t = j * n + id_shell

        if id_core is not None:
            k = j * n + id_core
            rows.append(k)
            columns.append(k)

            if j == 0:
                data.append(
                    -1 / (0.5 * R_axial_core[j, 0] + 0.5 * R_axial_core[j + 1, 0])
                )  # core manca termine precedente

                rows.append(k)
                columns.append(k + n)
                data.append(
                    1 / (0.5 * R_axial_core[j + 1, 0] + 0.5 * R_axial_core[j, 0])
                )  # diag

            elif j == (m - 1):
                data.append(
                    -1 / (0.5 * R_axial_core[j, 0] + 0.5 * R_axial_core[j - 1, 0])
                )  # core manca termine successivo

                rows.append(k)
                columns.append(k - n)
                data.append(
                    1 / (0.5 * R_axial_core[j - 1, 0] + 0.5 * R_axial_core[j, 0])
                )  # diag

            else:
                data.append(
                    -1 / (0.5 * R_axial_core[j, 0] + 0.5 * R_axial_core[j - 1, 0])
                    - 1 / (0.5 * R_axial_core[j, 0] + 0.5 * R_axial_core[j + 1, 0])
                )  # core

                rows.append(k)
                columns.append(k - n)
                data.append(
                    1 / (0.5 * R_axial_core[j - 1, 0] + 0.5 * R_axial_core[j, 0])
                )  # diag

                rows.append(k)
                columns.append(k + n)
                data.append(
                    1 / (0.5 * R_axial_core[j + 1, 0] + 0.5 * R_axial_core[j, 0])
                )  # diag

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
    for j in range(bh_p.m_mesh):
        if j != 0:
            rows.append(j * n + 6)
            columns.append((j - 1) * n + 6)
            data.append(mw_tot_j * fluid.cp_w)

            rows.append(j * n + 7)
            columns.append((j - 1) * n + 7)
            data.append(mw_tot_j * fluid.cp_w)

        if j == bh_p.m_mesh - 1:
            rows.append(j * n + 8)
            columns.append(j * n + 6)
            data.append(mw_tot_j * fluid.cp_w)

            rows.append(j * n + 9)
            columns.append(j * n + 7)
            data.append(mw_tot_j * fluid.cp_w)

        else:
            rows.append(j * n + 8)
            columns.append((j + 1) * n + 8)
            data.append(mw_tot_j * fluid.cp_w)

            rows.append(j * n + 9)
            columns.append((j + 1) * n + 9)
            data.append(mw_tot_j * fluid.cp_w)

    if (bh_p.connection == "S") or (bh_p.connection == "s"):
        rows.append(7)
        columns.append(8)
        data.append(mw_tot_j * fluid.cp_w)

    A_bhe_double = coo_matrix((data, (rows, columns)), shape=(N, N), dtype=np.float64)
    A_bhe_double = A_bhe_double.tocsc()

    return A_bhe_double


def build_rhs_double_u_tube(model, gr_p, timesteps, T_old_borehole, mw_tot_j, Tf1_j):

    bh_p = model.borehole
    fluid = model.fluid
    n = bh_p.n_equations

    b_bhe_double = np.zeros((n * bh_p.m_mesh))

    if model.borehole.connection in {"P", "p"}:
        mw_tot_j = mw_tot_j / 2.0

    for j in range(bh_p.m_mesh):

        b_bhe_double[j * n + bh_p.id_shell] = (
            -bh_p.C_shell[j, 0] * T_old_borehole[j * n + bh_p.id_shell] / (timesteps)
        )
        b_bhe_double[j * n + bh_p.id_core] = (
            -bh_p.C_core[j, 0] * T_old_borehole[j * n + bh_p.id_core] / (timesteps)
        )
        b_bhe_double[j * n + 6] = -bh_p.C_fluid * T_old_borehole[j * n + 6] / timesteps
        b_bhe_double[j * n + 7] = -bh_p.C_fluid * T_old_borehole[j * n + 7] / timesteps
        b_bhe_double[j * n + 8] = -bh_p.C_fluid * T_old_borehole[j * n + 8] / timesteps
        b_bhe_double[j * n + 9] = -bh_p.C_fluid * T_old_borehole[j * n + 9] / timesteps

    if mw_tot_j != 0:
        b_bhe_double[6] += -mw_tot_j * fluid.cp_w * Tf1_j
        if bh_p.connection not in ("S", "s"):
            b_bhe_double[7] += -mw_tot_j * fluid.cp_w * Tf1_j

    return b_bhe_double
