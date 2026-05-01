# -*- coding: utf-8 -*-
"""
Ground matrix assembly module.

Builds the local coefficient matrices and right-hand side vectors for
the upper, middle, and lower ground regions, used internally by the
global assembler.
"""
import numpy as np
from scipy.sparse import coo_matrix


def build_coefficient_matrix_ground(gr_p, timesteps):
    m = gr_p.m_mesh
    n = gr_p.n_mesh
    N = m * n
    rows = []
    columns = []
    data = []

    for j in range(m):
        for i in range(n):
            k = j * n + i
            if j == 0:
                rows.append(k + n)
                columns.append(k)
                data.append(
                    1 / (0.5 * gr_p.R_axial[j + 1, i] + 0.5 * gr_p.R_axial[j, i])
                )
                if i == 0:
                    rows.append(k)
                    columns.append(k)
                    data.append(
                        -1 / (gr_p.R_ground[j, i])
                        - 1 / (gr_p.R_ground[j, i + 1])
                        - gr_p.C_ground[j, i] / timesteps
                        - 1 / (0.5 * gr_p.R_axial[j + 1, i] + 0.5 * gr_p.R_axial[j, i])
                    )

                    rows.append(k)
                    columns.append(k + 1)
                    data.append(1 / (gr_p.R_ground[j, i + 1]))
                elif i == (n - 1):
                    rows.append(k)
                    columns.append(k)
                    data.append(
                        -1 / (gr_p.R_ground[j, i + 1])
                        - 1 / (gr_p.R_ground[j, i])
                        - gr_p.C_ground[j, i] / timesteps
                        - 1 / (0.5 * gr_p.R_axial[j + 1, i] + 0.5 * gr_p.R_axial[j, i])
                    )

                    rows.append(k)
                    columns.append(k - 1)
                    data.append(1 / (gr_p.R_ground[j, i]))
                else:
                    rows.append(k)
                    columns.append(k - 1)
                    data.append(1 / (gr_p.R_ground[j, i]))

                    rows.append(k)
                    columns.append(k)
                    data.append(
                        -1 / (gr_p.R_ground[j, i])
                        - 1 / (gr_p.R_ground[j, i + 1])
                        - gr_p.C_ground[j, i] / timesteps
                        - 1 / (0.5 * gr_p.R_axial[j + 1, i] + 0.5 * gr_p.R_axial[j, i])
                    )

                    rows.append(k)
                    columns.append(k + 1)
                    data.append(1 / (gr_p.R_ground[j, i + 1]))

            elif j == (m - 1):
                rows.append(k - n)
                columns.append(k)
                data.append(
                    1 / (0.5 * gr_p.R_axial[j - 1, i] + 0.5 * gr_p.R_axial[j, i])
                )
                if i == 0:
                    rows.append(k)
                    columns.append(k)
                    data.append(
                        -1 / (gr_p.R_ground[j, i])
                        - 1 / (gr_p.R_ground[j, i + 1])
                        - gr_p.C_ground[j, i] / timesteps
                        - 1 / (0.5 * gr_p.R_axial[j - 1, i] + 0.5 * gr_p.R_axial[j, i])
                    )

                    rows.append(k)
                    columns.append(k + 1)
                    data.append(1 / (gr_p.R_ground[j, i + 1]))
                elif i == (gr_p.n_mesh - 1):
                    rows.append(k)
                    columns.append(k)
                    data.append(
                        -1 / (gr_p.R_ground[j, i + 1])
                        - 1 / (gr_p.R_ground[j, i])
                        - gr_p.C_ground[j, i] / timesteps
                        - 1 / (0.5 * gr_p.R_axial[j - 1, i] + 0.5 * gr_p.R_axial[j, i])
                    )

                    rows.append(k)
                    columns.append(k - 1)
                    data.append(1 / (gr_p.R_ground[j, i]))
                else:
                    rows.append(k)
                    columns.append(k)
                    data.append(
                        -1 / (gr_p.R_ground[j, i])
                        - 1 / (gr_p.R_ground[j, i + 1])
                        - gr_p.C_ground[j, i] / timesteps
                        - 1 / (0.5 * gr_p.R_axial[j - 1, i] + 0.5 * gr_p.R_axial[j, i])
                    )

                    rows.append(k)
                    columns.append(k + 1)
                    data.append(1 / (gr_p.R_ground[j, i + 1]))

                    rows.append(k)
                    columns.append(k - 1)
                    data.append(1 / (gr_p.R_ground[j, i]))

            elif (j != 0) and (j != m - 1):
                rows.append(k - n)
                columns.append(k)
                data.append(
                    1 / (0.5 * gr_p.R_axial[j - 1, i] + 0.5 * gr_p.R_axial[j, i])
                )

                rows.append(k + n)
                columns.append(k)
                data.append(
                    1 / (0.5 * gr_p.R_axial[j + 1, i] + 0.5 * gr_p.R_axial[j, i])
                )
                if i == 0:
                    rows.append(k)
                    columns.append(k)
                    data.append(
                        -1 / (gr_p.R_ground[j, i])
                        - 1 / (gr_p.R_ground[j, i + 1])
                        - gr_p.C_ground[j, i] / timesteps
                        - 1 / (0.5 * gr_p.R_axial[j, i] + 0.5 * gr_p.R_axial[j - 1, i])
                        - 1 / (0.5 * gr_p.R_axial[j, i] + 0.5 * gr_p.R_axial[j + 1, i])
                    )

                    rows.append(k)
                    columns.append(k + 1)
                    data.append(1 / (gr_p.R_ground[j, i + 1]))
                elif i == (gr_p.n_mesh - 1):
                    rows.append(k)
                    columns.append(k)
                    data.append(
                        -1 / (gr_p.R_ground[j, i + 1])
                        - 1 / (gr_p.R_ground[j, i])
                        - gr_p.C_ground[j, i] / timesteps
                        - 1 / (0.5 * gr_p.R_axial[j, i] + 0.5 * gr_p.R_axial[j - 1, i])
                        - 1 / (0.5 * gr_p.R_axial[j, i] + 0.5 * gr_p.R_axial[j + 1, i])
                    )

                    rows.append(k)
                    columns.append(k - 1)
                    data.append(1 / (gr_p.R_ground[j, i]))
                else:
                    rows.append(k)
                    columns.append(k)
                    data.append(
                        -1 / (gr_p.R_ground[j, i])
                        - 1 / (gr_p.R_ground[j, i + 1])
                        - gr_p.C_ground[j, i] / timesteps
                        - 1 / (0.5 * gr_p.R_axial[j, i] + 0.5 * gr_p.R_axial[j - 1, i])
                        - 1 / (0.5 * gr_p.R_axial[j, i] + 0.5 * gr_p.R_axial[j + 1, i])
                    )

                    rows.append(k)
                    columns.append(k + 1)
                    data.append(1 / (gr_p.R_ground[j, i + 1]))

                    rows.append(k)
                    columns.append(k - 1)
                    data.append(1 / (gr_p.R_ground[j, i]))

    A_ground = coo_matrix((data, (rows, columns)), shape=(N, N), dtype=np.float64)
    A_ground = A_ground.tocsc()

    return A_ground


def build_rhs_ground(gr_p, timesteps, T_old_ground, T_bc):

    b_ground = np.zeros((gr_p.n_mesh * gr_p.m_mesh))

    for j in range(gr_p.m_mesh):
        b_ground_j = np.zeros((gr_p.n_mesh))

        for i in range(gr_p.n_mesh):
            if i == 0:
                b_ground_j[i] = (
                    -(gr_p.C_ground[j, i] / timesteps)
                    * T_old_ground[j * gr_p.n_mesh + i]
                )

            elif i == (gr_p.n_mesh - 1):
                b_ground_j[i] = (
                    -(gr_p.C_ground[j, i] / timesteps)
                    * T_old_ground[j * gr_p.n_mesh + i]
                    - T_bc[j] / gr_p.R_ground[j, -1]
                )

            else:
                b_ground_j[i] = (
                    -(gr_p.C_ground[j, i] / timesteps)
                    * T_old_ground[j * gr_p.n_mesh + i]
                )

        b_ground[j * gr_p.n_mesh : (j + 1) * gr_p.n_mesh] = b_ground_j

    return b_ground


def build_coefficient_matrix_ground_sup(gr_p, env_params, timesteps):
    m = gr_p.m_mesh_sup
    R_sup = gr_p.R_sup
    C_sup = gr_p.C_sup

    y = (np.pi * gr_p.rn**2.0) / env_params.envprops.R_ext

    rows = []
    columns = []
    data = []

    for j in range(
        1, m + 1
    ):  # per le resistenze e capacità: j rappresneta le successive, j-1 le attuali, j-2 le precedneti
        if j == 1:
            rows.append(j - 1)
            columns.append(j - 1)
            data.append(-y - 1 / (0.5 * gr_p.R_sup[j - 1]))  # nodo esterno

            rows.append(j - 1)
            columns.append(j)
            data.append(1 / (0.5 * R_sup[j - 1]))

            rows.append(j)
            columns.append(j - 1)
            data.append(1 / (0.5 * R_sup[j - 1]))  # nodo 1

            rows.append(j)
            columns.append(j)
            data.append(
                -1 / (0.5 * R_sup[j - 1])
                - 1 / (0.5 * R_sup[j - 1] + 0.5 * R_sup[j])
                - C_sup[j - 1] / timesteps
            )

            rows.append(j)
            columns.append(j + 1)
            data.append(1 / (0.5 * R_sup[j - 1] + 0.5 * R_sup[j]))

        elif j == gr_p.m_mesh_sup:
            rows.append(j)
            columns.append(j)
            data.append(
                -1 / (0.5 * R_sup[-1] + 0.5 * R_sup[-2]) - C_sup[-1] / timesteps
            )  # nodo vs matrice centrale

            rows.append(j)
            columns.append(j - 1)
            data.append(1 / (0.5 * R_sup[-1] + 0.5 * R_sup[-2]))

        else:
            rows.append(j)
            columns.append(j - 1)
            data.append(1 / (0.5 * R_sup[j - 2] + 0.5 * R_sup[j - 1]))

            rows.append(j)
            columns.append(j)
            data.append(
                -1 / (0.5 * R_sup[j - 2] + 0.5 * R_sup[j - 1])
                - 1 / (0.5 * R_sup[j - 1] + 0.5 * R_sup[j])
                - C_sup[j - 1] / timesteps
            )

            rows.append(j)
            columns.append(j + 1)
            data.append(1 / (0.5 * R_sup[j - 1] + 0.5 * R_sup[j]))

    A_sup = coo_matrix((data, (rows, columns)), shape=(m + 1, m + 1), dtype=np.float64)
    A_sup = A_sup.tocsc()

    return A_sup


def build_rhs_ground_sup(gr_p, timesteps, T_old_ground_sup):

    b_sup = np.zeros((gr_p.m_mesh_sup + 1))

    for j in range(1, gr_p.m_mesh_sup + 1):
        b_sup[j] = -(gr_p.C_sup[j - 1] / timesteps) * T_old_ground_sup[j - 1]

    return b_sup


def build_coefficient_matrix_ground_inf(gr_p, timesteps):
    R_inf = gr_p.R_inf
    C_inf = gr_p.C_inf
    m = gr_p.m_mesh_inf

    rows = []
    columns = []
    data = []

    for j in range(m):
        if j == 0:
            rows.append(j)
            columns.append(j)
            data.append(
                -1 / (0.5 * R_inf[j] + 0.5 * R_inf[j + 1]) - C_inf[j] / timesteps
            )

            rows.append(j)
            columns.append(j + 1)
            data.append(1 / (0.5 * gr_p.R_inf[j] + 0.5 * gr_p.R_inf[j + 1]))

        elif j == m - 1:

            rows.append(j)
            columns.append(j - 1)
            data.append(1 / (0.5 * R_inf[j - 1] + 0.5 * R_inf[j]))

            rows.append(j)
            columns.append(j)
            data.append(
                -1 / (0.5 * R_inf[-1])
                - 1 / (0.5 * R_inf[j] + 0.5 * R_inf[j - 1])
                - C_inf[j] / timesteps
            )

        else:
            rows.append(j)
            columns.append(j - 1)
            data.append(1 / (0.5 * R_inf[j - 1] + 0.5 * R_inf[j]))

            rows.append(j)
            columns.append(j)
            data.append(
                -1 / (0.5 * R_inf[j - 1] + 0.5 * R_inf[j])
                - 1 / (0.5 * R_inf[j] + 0.5 * R_inf[j + 1])
                - C_inf[j] / timesteps
            )

            rows.append(j)
            columns.append(j + 1)
            data.append(1 / (0.5 * R_inf[j] + 0.5 * R_inf[j + 1]))

    A_inf = coo_matrix((data, (rows, columns)), shape=(m, m), dtype=np.float64)
    A_inf = A_inf.tocsc()

    return A_inf


def build_rhs_ground_inf(gr_p, env_params, timesteps, T_old_ground_inf):

    b_inf = np.zeros((gr_p.m_mesh_inf))

    for j in range(gr_p.m_mesh_inf):
        b_inf[j] = -(gr_p.C_inf[j] / timesteps) * T_old_ground_inf[j]

    b_inf[-1] = b_inf[-1] - env_params.envinput.Tm / (0.5 * gr_p.R_inf[-1])

    return b_inf
