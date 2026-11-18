"""
wavefunctions.py
=================
Analytic hydrogenic wavefunctions psi_nlm(r, theta, phi) = R_nl(r) * Y_l^m(theta, phi),
built on top of the from-scratch special functions in special_functions.py.

r is expressed in Bohr radii throughout (dimensionless).
"""

import numpy as np
from .special_functions import assoc_laguerre, spherical_harmonic, factorial
from .constants import energy_level_eV, energy_level_hartree


def validate_quantum_numbers(n: int, l: int, m: int):
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0 <= l <= n - 1):
        raise ValueError(f"l must satisfy 0 <= l <= n-1 (got l={l}, n={n})")
    if not (-l <= m <= l):
        raise ValueError(f"m must satisfy -l <= m <= l (got m={m}, l={l})")


def radial_wavefunction(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """
    R_nl(r), normalized such that integral_0^inf |R_nl(r)|^2 r^2 dr = 1.
    r in Bohr radii.
    """
    r = np.asarray(r, dtype=float)
    rho = 2.0 * r / n
    laguerre = assoc_laguerre(n - l - 1, 2 * l + 1, rho)

    norm = np.sqrt(
        (2.0 / n) ** 3 * factorial(n - l - 1) / (2.0 * n * factorial(n + l))
    )
    return norm * np.exp(-rho / 2.0) * rho**l * laguerre


def angular_wavefunction(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return spherical_harmonic(l, m, theta, phi)


def full_wavefunction(n: int, l: int, m: int, r, theta, phi):
    """Complex-valued psi_nlm(r, theta, phi)."""
    return radial_wavefunction(n, l, r) * angular_wavefunction(l, m, theta, phi)


def probability_density(n: int, l: int, m: int, r, theta, phi):
    """|psi_nlm(r, theta, phi)|^2."""
    psi = full_wavefunction(n, l, m, r, theta, phi)
    return np.abs(psi) ** 2


def radial_probability_density(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """P(r) = r^2 |R_nl(r)|^2 -- probability per unit radius."""
    R = radial_wavefunction(n, l, r)
    return r**2 * R**2


def expectation_r(n: int, l: int) -> float:
    """
    <r> for state (n,l), in Bohr radii, via the known closed form:
        <r>_nl = (n^2 / 2) * [ 3 - l(l+1)/n^2 ]
    (A standard result derivable from the Kramers relation; used here as
    an analytic cross-check for the numerical/Monte-Carlo results.)
    """
    return 0.5 * n * n * (3.0 - l * (l + 1) / (n * n))
