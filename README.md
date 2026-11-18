# Quantum Atom Engine

A from-scratch quantum mechanical hydrogen atom engine.

## What's actually "from scratch" here

- **special_functions.py** — associated Laguerre polynomials and associated
  Legendre functions / spherical harmonics implemented directly from their
  defining recurrence relations (not imported from scipy.special). Verified
  to match scipy to machine precision (~1e-13) as a correctness check.
- **numerical_solver.py** — the actual physics engine: solves the radial
  Schrödinger equation numerically from first principles (finite-difference
  discretization + tridiagonal eigenvalue solve), and independently
  reproduces the analytic hydrogen energy levels to within ~0.01-2%
  (residual error is pure grid-resolution error, not approximation error).
- **dynamics.py** — real time-dependent quantum mechanics: superpositions of
  eigenstates that genuinely evolve and interfere ("quantum beats"), not
  just static pictures.
- **sampling.py** — Monte Carlo rejection sampling of the electron's
  position from the true |psi|^2 distribution, used to build the visual
  "electron cloud".
- **visualization.py** — static matplotlib plots, plus a fully
  OFFLINE-capable interactive 3D viewer (Plotly.js is embedded directly
  in the HTML, so the file needs zero internet connection, ever, including
  first open).
- **engine.py** — a single clean `AtomEngine` class tying everything
  together.

Libraries used (for numerical linear algebra / arrays / plotting only —
never for the physics/math itself): numpy, scipy (only `eigh_tridiagonal`
for diagonalizing our own hand-built Hamiltonian matrix), matplotlib, plotly.

## Install

```bash
pip install numpy scipy matplotlib plotly
```

## Quick start (Python API)

```python
from quantum_atom_engine import AtomEngine

atom = AtomEngine(n=3, l=1, m=0)
print(atom)                                   # AtomEngine(n=3, l=1, m=0) ...
print(atom.energy_eV())                       # -1.5117 eV
atom.render_offline_html("my_orbital.html")   # interactive 3D, works offline
atom.plot_cross_section(save_path="cross.png")
```

## Command line

```bash
# Generate an offline interactive 3D electron cloud
python -m quantum_atom_engine.cli --n 3 --l 1 --m 0 --interactive orbital.html

# Also generate static plots
python -m quantum_atom_engine.cli --n 4 --l 2 --m 1 --static-prefix out/orbital_4d --interactive out/orbital_4d.html

# Run the numerical Schrödinger-equation solver and see it validate the
# analytic energies from first principles
python -m quantum_atom_engine.cli --validate-solver

# List valid (n, l, m) combinations
python -m quantum_atom_engine.cli --list-orbitals
```

## Quantum dynamics / superposition example

```python
from quantum_atom_engine import QuantumState, Superposition

# Superpose the ground state and an excited state -> real interference dynamics
sp = Superposition([QuantumState(1, 0, 0, 1.0), QuantumState(2, 1, 0, 1.0)])
print(sp.beat_period_seconds())     # characteristic oscillation period, seconds
density_now = sp.probability_density(r=1.5, theta=1.0, phi=0.5, t_atomic_units=0.0)
```

## Quantum numbers

- `n` (principal): 1, 2, 3, ...
- `l` (azimuthal): 0 to n-1  (0=s, 1=p, 2=d, 3=f, 4=g, 5=h)
- `m` (magnetic): -l to l
