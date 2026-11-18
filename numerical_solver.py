"""
numerical_solver.py
====================
This is the actual "physics engine" part: instead of trusting the closed-form
hydrogen energies, we DERIVE them by numerically solving the radial
time-independent Schrödinger equation on a grid, from first principles.

Radial equation (atomic units, hbar = m_e = e = 1), after substituting
u(r) = r * R(r):

    -1/2 * u''(r)  +  V_eff(r) * u(r)  =  E * u(r)

    V_eff(r) = -1/r  +  l(l+1) / (2 r^2)      (Coulomb + centrifugal barrier)

    boundary conditions: u(0) = 0, u(r -> inf) = 0

We discretize u(r) on a uniform grid r_1..r_N (finite-difference second
derivative), turning this into a symmetric tridiagonal matrix eigenvalue
problem, and diagonalize it (via scipy.linalg.eigh_tridiagonal -- a
standard, well-tested numerical linear algebra routine; we are not
reinventing tridiagonal eigensolvers, but every physics term/matrix
element here is built from the Schrödinger equation itself).

The resulting lowest eigenvalues are then compared against the exact
analytic result E_n = -1/(2n^2) Hartree as a correctness check.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh_tridiagonal

from .constants import energy_level_hartree, HARTREE_TO_EV


@dataclass
class NumericalSolveResult:
    l: int
    r_grid: np.ndarray
    dr: float
    eigenvalues_hartree: np.ndarray     # lowest few energies found (Hartree)
    eigenvectors: np.ndarray            # columns = u(r) for each eigenvalue
    n_values: np.ndarray                # inferred principal quantum numbers
    analytic_hartree: np.ndarray        # E_n = -1/(2n^2) for comparison
    percent_error: np.ndarray


def solve_radial_schrodinger(
    l: int,
    r_max: float = 80.0,
    n_grid: int = 4000,
    n_states: int = 4,
) -> NumericalSolveResult:
    """
    Numerically solve the radial Schrödinger equation for angular momentum
    quantum number `l`, returning the lowest `n_states` bound-state energies
    (for n = l+1, l+2, ..., l+n_states).
    """
    # Start the grid one spacing away from r=0 (not an arbitrary tiny
    # epsilon) so the -1/r Coulomb term never blows up disproportionately
    # relative to the grid resolution -- otherwise the discretization
    # creates a spurious, unphysically deep "artifact" state at r ~ 0,
    # especially for l=0 where there is no centrifugal barrier to
    # suppress the wavefunction near the origin.
    dr = r_max / (n_grid + 1)
    r = dr * np.arange(1, n_grid + 1)

    # Effective potential: Coulomb attraction + centrifugal barrier
    V_eff = -1.0 / r + (l * (l + 1)) / (2.0 * r**2)

    # Finite-difference kinetic term: -1/2 * d^2/dr^2 discretized as a
    # tridiagonal matrix (standard 3-point stencil):
    #   diagonal:      1/dr^2 + V_eff(r_i)
    #   off-diagonal:  -1/(2 dr^2)
    diag = 1.0 / dr**2 + V_eff
    off_diag = -0.5 / dr**2 * np.ones(n_grid - 1)

    # Lowest `n_states + l` eigenvalues (ground + first few excited) since
    # for a given l the allowed states start at n = l+1
    n_wanted = n_states
    eigvals, eigvecs = eigh_tridiagonal(
        diag, off_diag,
        select="i", select_range=(0, n_wanted - 1),
    )

    n_values = np.arange(l + 1, l + 1 + n_wanted)
    analytic = np.array([energy_level_hartree(n) for n in n_values])
    pct_err = np.abs((eigvals - analytic) / analytic) * 100.0

    return NumericalSolveResult(
        l=l,
        r_grid=r,
        dr=dr,
        eigenvalues_hartree=eigvals,
        eigenvectors=eigvecs,
        n_values=n_values,
        analytic_hartree=analytic,
        percent_error=pct_err,
    )


def print_validation_report(l_values=(0, 1, 2), n_states=4, r_max=80.0, n_grid=4000):
    """
    Runs the numerical solver for several l and prints a side-by-side
    comparison against the exact analytic hydrogen energies -- i.e. proof
    that the engine's physics (not just its formulas) is correct.
    """
    print("=" * 78)
    print("NUMERICAL SCHRÖDINGER EQUATION SOLVER -- VALIDATION AGAINST ANALYTIC THEORY")
    print("=" * 78)
    for l in l_values:
        result = solve_radial_schrodinger(l, r_max=r_max, n_grid=n_grid, n_states=n_states)
        print(f"\nAngular momentum l = {l}:")
        print(f"{'n':>4} {'Numerical (eV)':>18} {'Analytic (eV)':>16} {'Error %':>10}")
        for n, E_num, E_an, err in zip(
            result.n_values,
            result.eigenvalues_hartree * HARTREE_TO_EV,
            result.analytic_hartree * HARTREE_TO_EV,
            result.percent_error,
        ):
            print(f"{n:>4} {E_num:>18.5f} {E_an:>16.5f} {err:>9.3f}%")
    print("\n(Small % error is expected -- it comes purely from finite grid")
    print(" resolution/domain size, not from any approximation in the physics.)")
