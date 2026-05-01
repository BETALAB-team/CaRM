# -*- coding: utf-8 -*-
"""
BHE matrix dispatcher module.

Routes the assembly of the local BHE coefficient matrix and right-hand
side vector to the correct implementation based on the borehole type
(SingleUtube, DoubleUtube, Coaxial, Helical).
"""
from ..properties import SingleUtube, DoubleUtube, Coaxial, Helical

from .MatrixFunctions import (
    build_coefficient_matrix_bhe_single_u_tube,
    build_rhs_single_u_tube,
)
from .MatrixFunctions import (
    build_coefficient_matrix_bhe_double_u_tube,
    build_rhs_double_u_tube,
)
from .MatrixFunctions import build_coefficient_matrix_coaxial, build_rhs_coaxial
from .MatrixFunctions import build_coefficient_matrix_helical, build_rhs_helical


def build_bhe_matrix(model, gr_p, timesteps, mw_tot_j):

    conf = {
        SingleUtube: build_coefficient_matrix_bhe_single_u_tube,
        DoubleUtube: build_coefficient_matrix_bhe_double_u_tube,
        Coaxial: build_coefficient_matrix_coaxial,
        Helical: build_coefficient_matrix_helical,
    }

    bh_p = model.borehole

    A_bhe = conf[type(bh_p)](model, gr_p, timesteps, mw_tot_j)

    return A_bhe


def build_bhe_rhs(model, gr_p, timesteps, T_old_borehole, mw_tot_j, Tf1_j):

    conf = {
        SingleUtube: build_rhs_single_u_tube,
        DoubleUtube: build_rhs_double_u_tube,
        Coaxial: build_rhs_coaxial,
        Helical: build_rhs_helical,
    }

    bh_p = model.borehole

    b_bhe = conf[type(bh_p)](model, gr_p, timesteps, T_old_borehole, mw_tot_j, Tf1_j)

    return b_bhe
