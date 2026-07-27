"""Axisymmetric magnetic-field primitives.

The circular-loop field is the exact magnetostatic solution expressed through
complete elliptic integrals.  A finite rectangular winding pack is integrated
with tensor-product Gauss--Legendre quadrature; the quadrature order is an
explicit numerical parameter and can therefore be convergence-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import mu_0
from scipy.integrate import solve_ivp
from scipy.special import ellipe, ellipk

FloatArray = NDArray[np.float64]


class AxisymmetricMagneticField(Protocol):
    """Protocol for a source invariant under rotations about the z axis."""

    name: str

    def field_cylindrical(
        self, rho_m: ArrayLike, z_m: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``(B_rho, B_z)`` in tesla."""


def _broadcast_coordinates(
    rho_m: ArrayLike, z_m: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    rho, z = np.broadcast_arrays(
        np.asarray(rho_m, dtype=float), np.asarray(z_m, dtype=float)
    )
    if np.any(rho < 0.0):
        raise ValueError("The cylindrical radius rho_m must be non-negative.")
    return rho, z


@dataclass(frozen=True)
class CircularLoop:
    """Infinitesimally thin circular current loop.

    ``ampere_turns`` is the product ``N I``.  It may be negative to reverse
    the field direction.
    """

    name: str
    radius_m: float
    center_z_m: float
    ampere_turns: float

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")

    def field_cylindrical(
        self, rho_m: ArrayLike, z_m: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Evaluate the exact field using complete elliptic integrals."""

        rho, z = _broadcast_coordinates(rho_m, z_m)
        dz = z - self.center_z_m
        a = self.radius_m

        alpha_sq = (a - rho) ** 2 + dz**2
        beta_sq = (a + rho) ** 2 + dz**2
        if np.any(alpha_sq == 0.0):
            raise ValueError("The field of an ideal thin loop diverges on its wire.")

        b_rho = np.zeros_like(rho)
        b_z = np.empty_like(rho)

        # The general expression contains a removable 1/rho singularity.
        # On the axis we use its analytic limit.
        axis = rho <= 32.0 * np.finfo(float).eps * max(1.0, a)
        b_z[axis] = (
            mu_0
            * self.ampere_turns
            * a**2
            / (2.0 * (a**2 + dz[axis] ** 2) ** 1.5)
        )

        off_axis = ~axis
        if np.any(off_axis):
            r = rho[off_axis]
            zz = dz[off_axis]
            alpha2 = alpha_sq[off_axis]
            beta2 = beta_sq[off_axis]
            root_beta = np.sqrt(beta2)
            parameter = np.clip(4.0 * a * r / beta2, 0.0, 1.0)
            k_complete = ellipk(parameter)
            e_complete = ellipe(parameter)
            common = mu_0 * self.ampere_turns / (2.0 * np.pi * root_beta)

            b_rho[off_axis] = (
                common
                * zz
                / r
                * (
                    -k_complete
                    + (a**2 + r**2 + zz**2) * e_complete / alpha2
                )
            )
            b_z[off_axis] = common * (
                k_complete
                + (a**2 - r**2 - zz**2) * e_complete / alpha2
            )

        return b_rho, b_z

    def axis_field(self, z_m: ArrayLike) -> FloatArray:
        """Analytic axial component on the symmetry axis."""

        z = np.asarray(z_m, dtype=float)
        dz = z - self.center_z_m
        return (
            mu_0
            * self.ampere_turns
            * self.radius_m**2
            / (2.0 * (self.radius_m**2 + dz**2) ** 1.5)
        )


@dataclass(frozen=True)
class RectangularCoilPack:
    """Finite winding pack with uniform turn density in its cross-section."""

    name: str
    inner_radius_m: float
    radial_thickness_m: float
    center_z_m: float
    axial_length_m: float
    turns: int
    current_a: float
    radial_order: int = 6
    axial_order: int = 10

    def __post_init__(self) -> None:
        if self.inner_radius_m <= 0.0:
            raise ValueError("inner_radius_m must be positive")
        if self.radial_thickness_m <= 0.0 or self.axial_length_m <= 0.0:
            raise ValueError("The winding-pack dimensions must be positive")
        if self.turns <= 0:
            raise ValueError("turns must be positive")
        if self.radial_order <= 0 or self.axial_order <= 0:
            raise ValueError("Quadrature orders must be positive")

    @property
    def ampere_turns(self) -> float:
        return self.turns * self.current_a

    @cached_property
    def _quadrature_arrays(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        radial_nodes, radial_weights = np.polynomial.legendre.leggauss(
            self.radial_order
        )
        axial_nodes, axial_weights = np.polynomial.legendre.leggauss(
            self.axial_order
        )
        radius_mid = self.inner_radius_m + 0.5 * self.radial_thickness_m
        radius_half_width = 0.5 * self.radial_thickness_m
        axial_half_width = 0.5 * self.axial_length_m

        radius_grid, center_grid = np.meshgrid(
            radius_mid + radius_half_width * radial_nodes,
            self.center_z_m + axial_half_width * axial_nodes,
            indexing="ij",
        )
        # The tensor-product weights sum to four. Dividing by four makes the
        # effective loop currents sum exactly to N I.
        weight_grid = 0.25 * np.outer(radial_weights, axial_weights)
        return (
            radius_grid.ravel(),
            center_grid.ravel(),
            self.ampere_turns * weight_grid.ravel(),
        )

    def field_cylindrical(
        self, rho_m: ArrayLike, z_m: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        rho, z = _broadcast_coordinates(rho_m, z_m)
        b_rho = np.zeros_like(rho)
        b_z = np.empty_like(rho)
        radii, centers, ampere_turns = self._quadrature_arrays

        axis = rho <= 32.0 * np.finfo(float).eps * max(1.0, float(radii.max()))
        if np.any(axis):
            dz = z[axis, None] - centers
            b_z[axis] = np.sum(
                mu_0
                * ampere_turns
                * radii**2
                / (2.0 * (radii**2 + dz**2) ** 1.5),
                axis=-1,
            )

        off_axis = ~axis
        if np.any(off_axis):
            r = rho[off_axis, None]
            dz = z[off_axis, None] - centers
            alpha_sq = (radii - r) ** 2 + dz**2
            if np.any(alpha_sq == 0.0):
                raise ValueError(
                    "The quadrature filament field diverges on its wire."
                )
            beta_sq = (radii + r) ** 2 + dz**2
            root_beta = np.sqrt(beta_sq)
            parameter = np.clip(4.0 * radii * r / beta_sq, 0.0, 1.0)
            k_complete = ellipk(parameter)
            e_complete = ellipe(parameter)
            common = mu_0 * ampere_turns / (2.0 * np.pi * root_beta)

            b_rho[off_axis] = np.sum(
                common
                * dz
                / r
                * (
                    -k_complete
                    + (radii**2 + r**2 + dz**2) * e_complete / alpha_sq
                ),
                axis=-1,
            )
            b_z[off_axis] = np.sum(
                common
                * (
                    k_complete
                    + (radii**2 - r**2 - dz**2) * e_complete / alpha_sq
                ),
                axis=-1,
            )
        return b_rho, b_z

    def axis_field(self, z_m: ArrayLike) -> FloatArray:
        z = np.asarray(z_m, dtype=float)
        radii, centers, ampere_turns = self._quadrature_arrays
        dz = z[..., None] - centers
        return np.sum(
            mu_0
            * ampere_turns
            * radii**2
            / (2.0 * (radii**2 + dz**2) ** 1.5),
            axis=-1,
        )


@dataclass(frozen=True)
class UniformAxialField:
    """Spatially uniform field parallel to the z axis."""

    name: str
    field_z_t: float

    def field_cylindrical(
        self, rho_m: ArrayLike, z_m: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        rho, _ = _broadcast_coordinates(rho_m, z_m)
        return np.zeros_like(rho), np.full_like(rho, self.field_z_t)

    def axis_field(self, z_m: ArrayLike) -> FloatArray:
        z = np.asarray(z_m, dtype=float)
        return np.full_like(z, self.field_z_t)


@dataclass(frozen=True)
class CompositeMagneticField:
    """Linear superposition of axisymmetric magnetostatic sources."""

    name: str
    components: tuple[AxisymmetricMagneticField, ...]

    def field_cylindrical(
        self, rho_m: ArrayLike, z_m: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        rho, z = _broadcast_coordinates(rho_m, z_m)
        b_rho = np.zeros_like(rho)
        b_z = np.zeros_like(rho)
        for component in self.components:
            component_b_rho, component_b_z = component.field_cylindrical(rho, z)
            b_rho += component_b_rho
            b_z += component_b_z
        return b_rho, b_z

    def axis_field(self, z_m: ArrayLike) -> FloatArray:
        z = np.asarray(z_m, dtype=float)
        result = np.zeros_like(z)
        for component in self.components:
            result += component.axis_field(z)
        return result

    def field_cartesian(
        self, x_m: ArrayLike, y_m: ArrayLike, z_m: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return ``(B_x, B_y, B_z)`` for trajectory integration."""

        x, y, z = np.broadcast_arrays(
            np.asarray(x_m, dtype=float),
            np.asarray(y_m, dtype=float),
            np.asarray(z_m, dtype=float),
        )
        rho = np.hypot(x, y)
        b_rho, b_z = self.field_cylindrical(rho, z)
        b_x = np.zeros_like(rho)
        b_y = np.zeros_like(rho)
        off_axis = rho > 0.0
        b_x[off_axis] = b_rho[off_axis] * x[off_axis] / rho[off_axis]
        b_y[off_axis] = b_rho[off_axis] * y[off_axis] / rho[off_axis]
        return b_x, b_y, b_z


def integrate_axisymmetric_field_line_radius(
    field: AxisymmetricMagneticField,
    sampled_z_m: ArrayLike,
    initial_radius_m: float,
    *,
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance_m: float = 1.0e-11,
) -> FloatArray:
    """Integrate one axisymmetric magnetic field line as ``rho(z)``.

    A field line in a meridional plane satisfies

    ``d rho / dz = B_rho / B_z``.

    The azimuth is constant by axisymmetry.  The returned radius uses the
    caller's exact ``sampled_z_m`` grid.
    """

    sampled_z = np.asarray(sampled_z_m, dtype=float)
    if sampled_z.ndim != 1 or sampled_z.size < 2:
        raise ValueError("sampled_z_m must be a one-dimensional array")
    differences = np.diff(sampled_z)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError("sampled_z_m must be strictly monotonic")
    if initial_radius_m < 0.0:
        raise ValueError("initial_radius_m must be non-negative")

    def derivative(z_m: float, radius: FloatArray) -> FloatArray:
        b_rho, b_z = field.field_cylindrical(float(radius[0]), z_m)
        b_z_scalar = float(b_z)
        if abs(b_z_scalar) <= np.finfo(float).tiny:
            raise ValueError("B_z vanishes along the requested field line")
        return np.array([float(b_rho) / b_z_scalar])

    solution = solve_ivp(
        derivative,
        (float(sampled_z[0]), float(sampled_z[-1])),
        np.array([initial_radius_m], dtype=float),
        method="DOP853",
        t_eval=sampled_z,
        rtol=relative_tolerance,
        atol=absolute_tolerance_m,
    )
    if not solution.success:
        raise RuntimeError(f"Field-line integration failed: {solution.message}")
    radius = np.asarray(solution.y[0], dtype=float)
    if np.any(radius < 0.0):
        raise RuntimeError("Field-line integration crossed the symmetry axis")
    return radius
