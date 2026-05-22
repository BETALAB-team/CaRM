# -*- coding: utf-8 -*-
"""
Ground properties module.

Defines the geometry, mesh, and thermophysical properties of the ground
domain surrounding the borehole. Supports heterogeneous stratification
via a layer-by-layer property averaging scheme.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class GroundGeometry:
    """
    Geometric parameters of the ground domain.

    Attributes
    ----------
    D0 : float
        Borehole diameter [m].
    L : float
        Active borehole length (middle ground region) [m].
    L_sup : float
        Length of the upper ground region [m].
    L_inf : float
        Length of the lower ground region [m].
    rn : float or None
        Outer radius of the radial discretization [m].
        Required for single-borehole mode; ``None`` in multi-borehole mode
        (where ``r_eq`` from the Voronoi decomposition is used instead).
    r0 : float
        Borehole radius, derived as ``D0 / 2`` [m].
    """

    D0: float  # m
    L: float  # m
    L_sup: float  # m
    L_inf: float  # m
    rn: float | None = None  # m, default is None

    @property
    def r0(self) -> float:
        return self.D0 / 2.0

    def __post_init__(self) -> None:

        if self.D0 <= 0:
            raise ValueError("D0 must be > 0")

        if self.rn is not None:
            if self.rn <= self.r0:
                raise ValueError("rn must be > D0/2")

        if self.L <= 0 or self.L_sup <= 0 or self.L_inf <= 0:
            raise ValueError("L, L_sup, L_inf must be > 0")


@dataclass(frozen=True, slots=True)
class GroundMesh:
    """
    Discretization parameters for the ground domain.

    Attributes
    ----------
    n_mesh : int
        Number of radial mesh elements.
    m_mesh : int
        Number of axial mesh elements in the middle (active) region.
    m_mesh_sup : int
        Number of axial mesh elements in the upper region.
    m_mesh_inf : int
        Number of axial mesh elements in the lower region.
    f : float
        Radial expansion factor for the mesh (default 1.2). Controls how
        rapidly cell thickness increases moving outward from the borehole.
    """

    n_mesh: int  # radial mesh
    m_mesh: int  # axial mesh middle part
    m_mesh_sup: int  # axial mesh upper part
    m_mesh_inf: int  # axial mesh lower part
    f: float = 1.2  # expansion factor, default is 1.2

    def __post_init__(self) -> None:
        if self.n_mesh < 2:
            raise ValueError("n_mesh must be >= 2")
        if self.m_mesh <= 0 or self.m_mesh_sup <= 0 or self.m_mesh_inf <= 0:
            raise ValueError("m_mesh, m_mesh_sup, m_mesh_inf must be > 0")
        if self.f <= 0:
            raise ValueError("f must be > 0")


class GroundProperties:
    """
    Thermophysical and discretization properties of the ground domain.

    Computes layer-averaged thermal properties from the stratigraphic input,
    then derives all radial/axial resistances and capacitances used in the
    global system matrix.

    Attributes
    ----------
    geom : GroundGeometry
        Geometric parameters of the ground domain.
    mesh : GroundMesh
        Discretization settings.
    Tg : float
        Undisturbed ground temperature [°C].
    stratification : Sequence[tuple[float, float, float, float]]
        Ground layering as a sequence of ``(k, cp, rho, thickness)`` tuples.
        The sum of layer thicknesses must equal the total discretized length.
    soil_type: str
        Soil type string. This is set as None by default. If accounting for
        time variable properties, it must be set as 'sand', 'loam', or 'clay' 
        and the correct properties must be given as input.
    k : NDArray
        Layer-averaged thermal conductivity, shape (n_cells, 1) [W / (m K)].
    cp : NDArray
        Layer-averaged specific heat capacity, shape (n_cells, 1) [J / (kg K)].
    rho : NDArray
        Layer-averaged density, shape (n_cells, 1) [kg/m³].
    k_mean : float
        Mean thermal conductivity over the active (middle) region [W / (m K)].
    cp_mean : float
        Mean specific heat capacity over the active region [J / (kg K)].
    rho_mean : float
        Mean density over the active region [kg/m³].
    radius : NDArray
        Radial cell boundary positions, shape (1, n_mesh + 1) [m].
    rm : NDArray
        Barycentric radii for resistance calculations, shape (1, n_mesh + 2) [m].
    C_ground : NDArray
        Radial thermal capacitances, shape (m_mesh, n_mesh) [J/K].
    R_ground : NDArray
        Radial thermal resistances, shape (m_mesh, n_mesh + 1) [K/W].
    R_axial : NDArray
        Axial thermal resistances in the middle region, shape (m_mesh, n_mesh) [K/W].
    R_sup : NDArray
        Axial thermal resistances in the upper region, shape (m_mesh_sup,) [K/W].
    C_sup : NDArray
        Axial thermal capacitances in the upper region, shape (m_mesh_sup,) [J/K].
    R_inf : NDArray
        Axial thermal resistances in the lower region, shape (m_mesh_inf,) [K/W].
    C_inf : NDArray
        Axial thermal capacitances in the lower region, shape (m_mesh_inf,) [J/K].
    """

    def __init__(
        self,
        *,
        geom: GroundGeometry,
        mesh: GroundMesh,
        Tg: float,
        stratification: Sequence[tuple[float, float, float, float]],
        soil_type: str | None = None,
    ) -> None:

        self.geom = geom
        self.mesh = mesh

        # alias
        self.n_mesh = self.mesh.n_mesh
        self.m_mesh = self.mesh.m_mesh
        self.m_mesh_sup = self.mesh.m_mesh_sup
        self.m_mesh_inf = self.mesh.m_mesh_inf
        self.f = self.mesh.f

        self.r0 = self.geom.r0
        self.rn = self.geom.rn
        self.L = self.geom.L
        self.L_sup = self.geom.L_sup
        self.L_inf = self.geom.L_inf

        # input
        self.stratification = stratification
        self.Tg = Tg

        if soil_type is not None:
            soil_type = soil_type.strip().lower()
            if soil_type not in {"sand", "loam", "clay"}:
                raise ValueError("If soil_type is not None it must be set as 'sand', 'loam', or 'clay'.")
        self.soil_type = soil_type

        # dz calculation
        self.dz = self.L / self.m_mesh
        self.dz_sup = self.L_sup / self.m_mesh_sup
        self.dz_inf = self.L_inf / self.m_mesh_inf

        # props calulcation
        self.k, self.cp, self.rho = self._variable_properties()  # shape: (n_cells, 1)

        # fls props calculation
        self._mean_properties()

        # resistances and capacitances calculations
        self._compute_radius()
        self._compute_baricentric_radius()

        self._capacitance()
        self._resistance()
        self._resistance_axial()
        self._res_cap_sup()
        self._res_cap_inf()

    def _mean_properties(self):
        nstart = self.m_mesh_sup
        nend = self.m_mesh_sup + self.m_mesh

        k_middle = self.k[nstart:nend]
        cp_middle = self.cp[nstart:nend]
        rho_middle = self.rho[nstart:nend]

        self.k_mean = np.mean(k_middle)
        self.cp_mean = np.mean(cp_middle)
        self.rho_mean = np.mean(rho_middle)

    def _variable_properties(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dz_tot = np.asarray(
            [self.dz_sup] * self.m_mesh_sup
            + [self.dz] * self.m_mesh
            + [self.dz_inf] * self.m_mesh_inf,
            dtype=np.float64,
        )

        tol = 1e-9
        if abs((sum(Ls for _, _, _, Ls in self.stratification) - np.sum(dz_tot))) > tol:
            raise ValueError(
                "Length of stratifications must match the total length of discretization"
            )

        n_cells = self.m_mesh_sup + self.m_mesh + self.m_mesh_inf

        k = np.zeros(n_cells, dtype=np.float64)
        cp = np.zeros(n_cells, dtype=np.float64)
        rho = np.zeros(n_cells, dtype=np.float64)

        i = 0
        j = 0
        dz_res = dz_tot[i]
        k_m, cp_m, rho_m, L_res = self.stratification[j]

        k_acc = cp_acc = rho_acc = 0.0

        while i < n_cells and j < len(self.stratification):
            delta = min(dz_res, L_res)

            k_acc = k_acc + k_m * delta
            cp_acc = cp_acc + cp_m * delta
            rho_acc = rho_acc + rho_m * delta

            dz_res = dz_res - delta
            L_res = L_res - delta

            if dz_res <= tol:
                k[i] = k_acc / dz_tot[i]
                cp[i] = cp_acc / dz_tot[i]
                rho[i] = rho_acc / dz_tot[i]

                i += 1
                if i >= n_cells:
                    break

                dz_res = dz_tot[i]
                k_acc = cp_acc = rho_acc = 0

            if L_res <= tol:
                j += 1
                if j >= len(self.stratification):
                    break

                k_m, cp_m, rho_m, L_res = self.stratification[j]

        if np.any(k <= 0) or np.any(rho <= 0) or np.any(cp <= 0):
            bad = [
                (idx, k[idx], cp[idx], rho[idx])
                for idx in range(n_cells)
                if k[idx] <= 0 or cp[idx] <= 0 or rho[idx] <= 0
            ]
            raise ValueError(
                f"Unfilled/invalid cell properties at indices (idx,k,cp,rho): {bad[:10]} ..."
            )

        k = np.asarray(k)[:, None]
        cp = np.asarray(cp)[:, None]
        rho = np.asarray(rho)[:, None]

        return k, cp, rho

    def _update_properties(self, k: float, cp: float, rho: float) -> None:
        self.k[:, 0] = k
        self.cp[:, 0] = cp
        self.rho[:, 0] = rho

        self._capacitance()
        self._resistance()
        self._resistance_axial()
        self._res_cap_sup()
        self._res_cap_inf()

    def _compute_radius(self) -> None:
        self.radius = np.full(self.n_mesh + 1, self.r0, dtype=np.float64)

        i = np.arange(self.n_mesh, dtype=np.float64)
        weight = self.f**i
        corona_min_thick = (self.rn - self.r0) / np.sum(weight)

        dr = weight * corona_min_thick
        self.radius[1:] += np.cumsum(dr)

        self.radius = self.radius[None, :]  # shape (1, n_mesh + 1)

    def _compute_baricentric_radius(self) -> None:
        self.rm = np.empty(self.n_mesh + 2, dtype=np.float64)
        self.rm[0] = self.r0
        self.rm[1:-1] = np.sqrt(
            (self.radius[0, 1:] ** 2 + self.radius[0, :-1] ** 2) / 2
        )
        self.rm[-1] = self.rn
        self.rm = self.rm[None, :]  # shape (1, n_mesh + 2)

    def _capacitance(self) -> None:
        area = np.pi * (
            self.radius[0, 1:] ** 2 - self.radius[0, :-1] ** 2
        )  # shape (n_mesh,)
        volume = area * self.dz
        volume = volume[None, :]  # shape (1, n_mesh)

        cp = self.cp[self.m_mesh_sup : self.m_mesh_sup + self.m_mesh]
        rho = self.rho[self.m_mesh_sup : self.m_mesh_sup + self.m_mesh]

        self.C_ground = rho * cp * volume  # shape (m_mesh, n_mesh)

    def _resistance(self) -> None:
        k = self.k[self.m_mesh_sup : self.m_mesh_sup + self.m_mesh]
        self.R_ground = (
            1 / (2 * np.pi * k * self.dz) * np.log(self.rm[0, 1:] / self.rm[0, :-1])
        )  # shape (m_mesh, n_mesh + 1)

    def _resistance_axial(self) -> None:
        k = self.k[self.m_mesh_sup : self.m_mesh_sup + self.m_mesh]
        self.R_axial = self.dz / (
            k * np.pi * (self.radius[0, 1:] ** 2 - self.radius[0, :-1] ** 2)
        )  # shape (m_mesh, n_mesh)

    def _res_cap_sup(self) -> None:
        k = self.k[: self.m_mesh_sup]
        rho = self.rho[: self.m_mesh_sup]
        cp = self.cp[: self.m_mesh_sup]
        Area = np.pi * self.rn**2

        self.R_sup = (self.dz_sup / (k * Area)).ravel()
        self.C_sup = (rho * cp * self.dz_sup * Area).ravel()

    def _res_cap_inf(self) -> None:
        k = self.k[self.m_mesh_sup + self.m_mesh :]
        rho = self.rho[self.m_mesh_sup + self.m_mesh :]
        cp = self.cp[self.m_mesh_sup + self.m_mesh :]
        Area = np.pi * self.rn**2

        self.R_inf = (self.dz_inf / (k * Area)).ravel()
        self.C_inf = (rho * cp * self.dz_inf * Area).ravel()
