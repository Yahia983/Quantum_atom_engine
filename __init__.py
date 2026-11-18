"""
quantum_atom_engine
====================
A from-scratch quantum mechanical hydrogen atom engine: custom-built
special functions, analytic wavefunctions, a numerical Schrödinger-equation
solver (used to validate the analytic results from first principles),
time-dependent quantum dynamics (superpositions / interference), Monte
Carlo electron-cloud sampling, and offline-capable interactive 3D
rendering.

Quick start:
    from quantum_atom_engine import AtomEngine

    atom = AtomEngine(n=3, l=1, m=0)
    print(atom)
    atom.render_offline_html("my_orbital.html")
"""

from .engine import AtomEngine
from .constants import ORBITAL_LABELS, energy_level_eV, energy_level_hartree
from .numerical_solver import print_validation_report
from .dynamics import QuantumState, Superposition

__all__ = [
    "AtomEngine",
    "ORBITAL_LABELS",
    "energy_level_eV",
    "energy_level_hartree",
    "print_validation_report",
    "QuantumState",
    "Superposition",
]
