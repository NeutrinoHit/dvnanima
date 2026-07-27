"""Explicit electrostatic models used by the MAC-E animations.

The public KATRIN references constrain the main-spectrometer dimensions and
retarding voltage, but do not provide an as-built three-dimensional potential
map.  The model below is therefore an explicitly idealized, source-free
solution of Laplace's equation, not a hidden fitted axial curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray
from scipy.special import i0e, i1e

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FiniteCylindricalElectrodePotential:
    r"""Axisymmetric Laplace solution inside an infinite cylindrical boundary.

    The cylindrical boundary at ``rho = radius_m`` is held at
    ``boundary_potential_v`` for ``|z| < length_m / 2`` and at zero outside
    that interval.  Inside the cylinder,

    .. math::

       \Phi(\rho,z)=\frac{2V_0}{\pi}\int_0^\infty
       \frac{\sin(kL/2)}{k}\cos(kz)
       \frac{I_0(k\rho)}{I_0(kR)}\,dk .

    Every retained Fourier-Bessel mode satisfies the axisymmetric vacuum
    Laplace equation.  Gauss-Legendre quadrature and a finite upper integration
    limit are explicit numerical approximations and are convergence-tested by
    the exporter.
    """

    radius_m: float
    length_m: float
    boundary_potential_v: float
    quadrature_order: int = 512
    k_max_inv_m: float = 8.0
    chunk_size: int = 512

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        if self.length_m <= 0.0:
            raise ValueError("length_m must be positive")
        if self.quadrature_order < 32:
            raise ValueError("quadrature_order must be at least 32")
        if self.k_max_inv_m <= 0.0:
            raise ValueError("k_max_inv_m must be positive")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")

    @cached_property
    def _quadrature(self) -> tuple[FloatArray, FloatArray]:
        nodes, weights = leggauss(self.quadrature_order)
        wave_number = 0.5 * self.k_max_inv_m * (nodes + 1.0)
        scaled_weights = 0.5 * self.k_max_inv_m * weights
        return wave_number, scaled_weights

    def _prepare_coordinates(
        self,
        rho_m: ArrayLike,
        z_m: ArrayLike,
    ) -> tuple[FloatArray, FloatArray, tuple[int, ...]]:
        rho, z = np.broadcast_arrays(
            np.asarray(rho_m, dtype=float),
            np.asarray(z_m, dtype=float),
        )
        if np.any(rho < 0.0):
            raise ValueError("rho_m must be non-negative")
        if np.any(rho >= self.radius_m):
            raise ValueError(
                "The cylindrical Laplace series is defined for rho < radius_m"
            )
        return rho.ravel(), z.ravel(), rho.shape

    @staticmethod
    def _bessel_ratio(
        numerator_argument: FloatArray,
        denominator_argument: FloatArray,
        *,
        order: int,
    ) -> FloatArray:
        if order == 0:
            numerator = i0e(numerator_argument)
        elif order == 1:
            numerator = i1e(numerator_argument)
        else:
            raise ValueError("Only Bessel orders zero and one are supported")
        return (
            numerator
            / i0e(denominator_argument)
            * np.exp(numerator_argument - denominator_argument)
        )

    def potential_cylindrical(
        self,
        rho_m: ArrayLike,
        z_m: ArrayLike,
    ) -> FloatArray:
        """Return the scalar potential ``Phi(rho,z)`` in volts."""

        rho, z, output_shape = self._prepare_coordinates(rho_m, z_m)
        k, weights = self._quadrature
        denominator_argument = k * self.radius_m
        half_length = 0.5 * self.length_m
        spectral_boundary = np.sin(k * half_length) / k
        output = np.empty_like(rho)
        prefactor = 2.0 * self.boundary_potential_v / np.pi
        for start in range(0, rho.size, self.chunk_size):
            stop = min(start + self.chunk_size, rho.size)
            local_rho = rho[start:stop, None]
            local_z = z[start:stop, None]
            ratio = self._bessel_ratio(
                k[None, :] * local_rho,
                denominator_argument[None, :],
                order=0,
            )
            integrand = (
                spectral_boundary[None, :]
                * np.cos(k[None, :] * local_z)
                * ratio
            )
            output[start:stop] = prefactor * (integrand @ weights)
        return output.reshape(output_shape)

    def electric_field_cylindrical(
        self,
        rho_m: ArrayLike,
        z_m: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``(E_rho, E_z) = -grad(Phi)`` in V/m."""

        rho, z, output_shape = self._prepare_coordinates(rho_m, z_m)
        k, weights = self._quadrature
        denominator_argument = k * self.radius_m
        half_length = 0.5 * self.length_m
        sine_boundary = np.sin(k * half_length)
        e_rho = np.empty_like(rho)
        e_z = np.empty_like(rho)
        prefactor = 2.0 * self.boundary_potential_v / np.pi
        for start in range(0, rho.size, self.chunk_size):
            stop = min(start + self.chunk_size, rho.size)
            local_rho = rho[start:stop, None]
            local_z = z[start:stop, None]
            argument = k[None, :] * local_rho
            ratio_zero = self._bessel_ratio(
                argument,
                denominator_argument[None, :],
                order=0,
            )
            ratio_one = self._bessel_ratio(
                argument,
                denominator_argument[None, :],
                order=1,
            )
            cosine = np.cos(k[None, :] * local_z)
            sine = np.sin(k[None, :] * local_z)
            e_rho[start:stop] = -prefactor * (
                sine_boundary[None, :] * cosine * ratio_one @ weights
            )
            e_z[start:stop] = prefactor * (
                sine_boundary[None, :] * sine * ratio_zero @ weights
            )
        return e_rho.reshape(output_shape), e_z.reshape(output_shape)

    def with_boundary_potential(
        self,
        boundary_potential_v: float,
    ) -> "FiniteCylindricalElectrodePotential":
        """Return the same linear Laplace model with a different voltage."""

        return FiniteCylindricalElectrodePotential(
            radius_m=self.radius_m,
            length_m=self.length_m,
            boundary_potential_v=boundary_potential_v,
            quadrature_order=self.quadrature_order,
            k_max_inv_m=self.k_max_inv_m,
            chunk_size=self.chunk_size,
        )

