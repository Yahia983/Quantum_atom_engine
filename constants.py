"""
constants.py
============
Physical constants (SI) and atomic-unit conversion factors.

Atomic units (Hartree units) are used internally by the numerical solver
and time-evolution engine because they make the Schrödinger equation
dimensionless and numerically well-conditioned:

    hbar = m_e = e = 4*pi*eps0 = 1

In these units:
    length unit   -> 1 Bohr radius       (a0)
    energy unit   -> 1 Hartree           (= 27.211386 eV = 2 Rydberg)
    time unit     -> 1 atomic time unit  (= 2.4188843e-17 s)
"""

# ---- SI physical constants ----
A0_SI = 5.29177210903e-11        # Bohr radius, m
HBAR_SI = 1.054571817e-34        # J*s
ME_SI = 9.1093837015e-31         # electron rest mass, kg
E_SI = 1.602176634e-19           # elementary charge, C
EPS0_SI = 8.8541878128e-12       # vacuum permittivity, F/m
EV_SI = 1.602176634e-19          # 1 eV in Joules

# ---- Derived / atomic units ----
HARTREE_TO_EV = 27.211386245988
RYDBERG_TO_EV = 13.605693122994
ATOMIC_TIME_UNIT_SI = 2.4188843265857e-17   # seconds

# ---- Hydrogen ground-state reference ----
HYDROGEN_GROUND_STATE_EV = -RYDBERG_TO_EV

ORBITAL_LABELS = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}


def energy_level_hartree(n: int) -> float:
    """Exact hydrogen bound-state energy in Hartree: E_n = -1 / (2 n^2)."""
    return -1.0 / (2.0 * n * n)


def energy_level_eV(n: int) -> float:
    """Exact hydrogen bound-state energy in eV."""
    return energy_level_hartree(n) * HARTREE_TO_EV
