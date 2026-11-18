"""
special_functions.py
=====================
The mathematical building blocks of the hydrogenic wavefunctions,
implemented directly from their recurrence relations rather than pulled
pre-built from a library. Only `math.factorial` and `numpy` array
machinery are used as low-level utilities; the actual polynomial /
harmonic recursions are ours.

Contents:
    factorial(n)                        -- exact integer factorial
    double_factorial(n)                 -- n!! (odd double factorial, used below)
    assoc_laguerre(n, alpha, x)         -- generalized (associated) Laguerre polynomial L_n^alpha(x)
    assoc_legendre(l, m, x)             -- associated Legendre function P_l^m(x)
    spherical_harmonic(l, m, theta, phi)-- normalized spherical harmonic Y_l^m (complex)
"""

import math
import numpy as np


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("factorial undefined for negative integers")
    return math.factorial(n)


def double_factorial_odd(k: int) -> int:
    """(2k-1)!! = 1*3*5*...*(2k-1), with (-1)!! = 1 by convention."""
    result = 1
    for i in range(2 * k - 1, 0, -2):
        result *= i
    return result


def assoc_laguerre(n: int, alpha: int, x: np.ndarray) -> np.ndarray:
    """
    Generalized (associated) Laguerre polynomial L_n^alpha(x), built from
    the three-term recurrence relation:

        L_0^alpha(x) = 1
        L_1^alpha(x) = 1 + alpha - x
        k * L_k^alpha(x) = (2k - 1 + alpha - x) * L_{k-1}^alpha(x)
                            - (k - 1 + alpha) * L_{k-2}^alpha(x)

    x may be a scalar or numpy array.
    """
    x = np.asarray(x, dtype=float)
    if n < 0:
        raise ValueError("n must be >= 0")

    L_prev2 = np.ones_like(x)               # L_0
    if n == 0:
        return L_prev2

    L_prev1 = 1.0 + alpha - x               # L_1
    if n == 1:
        return L_prev1

    for k in range(2, n + 1):
        L_curr = ((2 * k - 1 + alpha - x) * L_prev1 - (k - 1 + alpha) * L_prev2) / k
        L_prev2, L_prev1 = L_prev1, L_curr

    return L_prev1


def assoc_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    Associated Legendre function P_l^m(x) for |x| <= 1, built from the
    standard recurrence relations (Condon-Shortley phase included):

        P_m^m(x)   = (-1)^m (2m-1)!! (1-x^2)^{m/2}
        P_{m+1}^m(x) = x (2m+1) P_m^m(x)
        (l-m) P_l^m(x) = x(2l-1) P_{l-1}^m(x) - (l+m-1) P_{l-2}^m(x)      for l > m+1

    Negative m is handled via:
        P_l^{-m}(x) = (-1)^m * (l-m)!/(l+m)! * P_l^m(x)
    """
    x = np.asarray(x, dtype=float)
    neg_m = m < 0
    m_abs = abs(m)
    if l < m_abs:
        raise ValueError("require l >= |m|")

    # Seed: P_m^m
    P_mm = ((-1) ** m_abs) * double_factorial_odd(m_abs) * (1.0 - x**2) ** (m_abs / 2.0) \
        if m_abs > 0 else np.ones_like(x)

    if l == m_abs:
        P_l_m = P_mm
    else:
        # P_{m+1}^m
        P_m1_m = x * (2 * m_abs + 1) * P_mm
        if l == m_abs + 1:
            P_l_m = P_m1_m
        else:
            P_prev2, P_prev1 = P_mm, P_m1_m
            for ll in range(m_abs + 2, l + 1):
                P_curr = (x * (2 * ll - 1) * P_prev1 - (ll + m_abs - 1) * P_prev2) / (ll - m_abs)
                P_prev2, P_prev1 = P_prev1, P_curr
            P_l_m = P_prev1

    if neg_m:
        sign_factor = ((-1) ** m_abs) * factorial(l - m_abs) / factorial(l + m_abs)
        return sign_factor * P_l_m
    return P_l_m


def spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Normalized spherical harmonic Y_l^m(theta, phi), complex-valued:

        Y_l^m(theta, phi) = sqrt( (2l+1)/(4*pi) * (l-|m|)!/(l+|m|)! )
                             * P_l^{|m|}(cos theta) * exp(i m phi)   [m >= 0]

        Y_l^{-m} = (-1)^m * conj(Y_l^m)                              [m < 0]

    theta in [0, pi] (polar angle), phi in [0, 2*pi) (azimuthal angle).
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    m_abs = abs(m)
    norm = math.sqrt(
        (2 * l + 1) / (4.0 * math.pi) * factorial(l - m_abs) / factorial(l + m_abs)
    )
    P = assoc_legendre(l, m_abs, np.cos(theta))
    Y_pos = norm * P * np.exp(1j * m_abs * phi)

    if m >= 0:
        return Y_pos
    else:
        return ((-1) ** m_abs) * np.conj(Y_pos)
