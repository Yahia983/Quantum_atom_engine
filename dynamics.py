"""
dynamics.py
===========
Time-dependent quantum mechanics: builds superpositions of stationary
states and evolves them in time according to the time-dependent
Schrödinger equation. For a superposition of energy eigenstates,

    Psi(r,theta,phi,t) = sum_i c_i * psi_i(r,theta,phi) * exp(-i E_i t / hbar)

each eigenstate's phase rotates at its own rate; interference between
different-energy components produces genuine time-dependent probability
density -- "quantum beats" -- something no single stationary orbital can
show. This is the atom's actual dynamics, not just a static snapshot.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .wavefunctions import full_wavefunction, validate_quantum_numbers
from .constants import energy_level_hartree, ATOMIC_TIME_UNIT_SI


@dataclass
class QuantumState:
    n: int
    l: int
    m: int
    coefficient: complex = 1.0 + 0.0j

    def __post_init__(self):
        validate_quantum_numbers(self.n, self.l, self.m)

    @property
    def energy_hartree(self) -> float:
        return energy_level_hartree(self.n)


class Superposition:
    """
    A normalized superposition of one or more hydrogen eigenstates,
    evolvable in time.
    """

    def __init__(self, states: List[QuantumState]):
        if not states:
            raise ValueError("Superposition requires at least one state")
        self.states = states
        self._normalize()

    def _normalize(self):
        norm_sq = sum(abs(s.coefficient) ** 2 for s in self.states)
        norm = np.sqrt(norm_sq)
        for s in self.states:
            s.coefficient = s.coefficient / norm

    def psi(self, r, theta, phi, t_atomic_units: float = 0.0):
        """
        Evaluate Psi(r, theta, phi, t) as a complex array.
        t_atomic_units: time expressed in atomic time units (hbar/Hartree).
        """
        total = np.zeros_like(np.asarray(r, dtype=complex))
        for s in self.states:
            phase = np.exp(-1j * s.energy_hartree * t_atomic_units)
            total = total + s.coefficient * phase * full_wavefunction(
                s.n, s.l, s.m, r, theta, phi
            )
        return total

    def probability_density(self, r, theta, phi, t_atomic_units: float = 0.0):
        return np.abs(self.psi(r, theta, phi, t_atomic_units)) ** 2

    def beat_period_atomic_units(self) -> float:
        """
        The characteristic oscillation period of the interference pattern
        between the two extreme energies present in the superposition:
            T = 2*pi / |E_i - E_j|
        Returns None for a single-state (stationary) superposition.
        """
        if len(self.states) < 2:
            return None
        energies = [s.energy_hartree for s in self.states]
        delta_E = max(energies) - min(energies)
        if delta_E == 0:
            return float("inf")  # degenerate states (same n) -> no beating, stationary superposition
        return 2 * np.pi / abs(delta_E)

    def beat_period_seconds(self) -> float:
        T = self.beat_period_atomic_units()
        if T is None or np.isinf(T):
            return T
        return T * ATOMIC_TIME_UNIT_SI


def sample_time_series_at_point(
    superposition: Superposition,
    r: float, theta: float, phi: float,
    t_values_atomic_units: np.ndarray,
) -> np.ndarray:
    """
    Probability density |Psi(r,theta,phi,t)|^2 evaluated at a single fixed
    point in space, over a range of times -- i.e. a literal time trace you
    could plot, showing the interference beats.
    """
    return np.array([
        superposition.probability_density(r, theta, phi, t)
        for t in t_values_atomic_units
    ])
