"""
engine.py
=========
Top-level AtomEngine: a single, simple entry point that ties together the
math (special_functions, wavefunctions), the physics validation
(numerical_solver), the dynamics (dynamics), the sampling (sampling) and
the rendering (visualization) subsystems.
"""

from .wavefunctions import (
    validate_quantum_numbers, radial_wavefunction, angular_wavefunction,
    full_wavefunction, probability_density, expectation_r,
)
from .constants import energy_level_eV, energy_level_hartree, ORBITAL_LABELS
from .numerical_solver import solve_radial_schrodinger, print_validation_report
from .dynamics import QuantumState, Superposition
from .sampling import sample_electron_cloud
from .visualization import plot_radial_density, plot_cross_section, render_interactive_offline_html


class AtomEngine:
    """
    Unified API for the hydrogen quantum atom engine.

    Example:
        atom = AtomEngine(n=3, l=1, m=0)
        atom.energy_eV()                      -> -1.51 eV
        atom.sample_cloud(50000)               -> (x, y, z) arrays
        atom.render_offline_html("out.html")   -> interactive 3D file
        atom.validate_against_numerical_solver()
    """

    def __init__(self, n: int, l: int, m: int):
        validate_quantum_numbers(n, l, m)
        self.n, self.l, self.m = n, l, m

    # ---- basic properties ----
    @property
    def label(self) -> str:
        return f"{self.n}{ORBITAL_LABELS.get(self.l, '?')}"

    def energy_eV(self) -> float:
        return energy_level_eV(self.n)

    def energy_hartree(self) -> float:
        return energy_level_hartree(self.n)

    def expectation_r(self) -> float:
        return expectation_r(self.n, self.l)

    # ---- wavefunction evaluation ----
    def psi(self, r, theta, phi):
        return full_wavefunction(self.n, self.l, self.m, r, theta, phi)

    def density(self, r, theta, phi):
        return probability_density(self.n, self.l, self.m, r, theta, phi)

    # ---- sampling ----
    def sample_cloud(self, n_samples: int = 50000, return_density: bool = False, seed: int = 0):
        return sample_electron_cloud(
            self.n, self.l, self.m, n_samples=n_samples, return_density=return_density, seed=seed
        )

    # ---- rendering ----
    def plot_radial(self, save_path=None):
        return plot_radial_density(self.n, self.l, save_path=save_path)

    def plot_cross_section(self, save_path=None):
        return plot_cross_section(self.n, self.l, self.m, save_path=save_path)

    def render_offline_html(self, out_path: str, n_samples: int = 60000):
        return render_interactive_offline_html(self.n, self.l, self.m, out_path, n_samples=n_samples)

    # ---- physics validation ----
    def validate_against_numerical_solver(self, n_states: int = 4, r_max: float = 80.0, n_grid: int = 4000):
        """
        Independently solves the Schrödinger equation numerically (finite
        differences + tridiagonal diagonalization) for this orbital's l and
        returns how closely the numerical ground-truth matches the analytic
        energy used elsewhere in the engine.
        """
        result = solve_radial_schrodinger(self.l, r_max=r_max, n_grid=n_grid, n_states=n_states)
        idx = list(result.n_values).index(self.n) if self.n in result.n_values else None
        return result, idx

    # ---- dynamics ----
    @staticmethod
    def superposition(states):
        """states: list of (n, l, m, coefficient) tuples."""
        return Superposition([QuantumState(n, l, m, c) for (n, l, m, c) in states])

    def __repr__(self):
        return f"AtomEngine(n={self.n}, l={self.l}, m={self.m})  # orbital {self.label}, E={self.energy_eV():.4f} eV"
