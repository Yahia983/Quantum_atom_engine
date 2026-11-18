"""
cli.py
======
Command-line interface for the quantum atom engine.

Examples:
    python -m quantum_atom_engine.cli --n 3 --l 1 --m 0 --interactive out.html
    python -m quantum_atom_engine.cli --validate-solver
    python -m quantum_atom_engine.cli --list-orbitals
"""

import argparse
from .engine import AtomEngine
from .numerical_solver import print_validation_report
from .constants import ORBITAL_LABELS


def list_orbitals(max_n=4):
    print(f"\nAvailable orbitals up to n={max_n}:")
    for n in range(1, max_n + 1):
        for l in range(0, n):
            label = f"{n}{ORBITAL_LABELS.get(l, '?')}"
            ms = list(range(-l, l + 1))
            print(f"  {label:>4}  (n={n}, l={l}, m in {ms})")


def main():
    parser = argparse.ArgumentParser(description="Quantum Atom Engine CLI")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--interactive", type=str, default=None,
                         help="Path to write an offline-capable interactive 3D HTML file")
    parser.add_argument("--static-prefix", type=str, default=None,
                         help="Prefix for static radial/cross-section PNG plots")
    parser.add_argument("--validate-solver", action="store_true",
                         help="Run the numerical Schrödinger solver and print a validation report")
    parser.add_argument("--list-orbitals", action="store_true")
    args = parser.parse_args()

    if args.list_orbitals:
        list_orbitals()
        return

    if args.validate_solver:
        print_validation_report()
        return

    atom = AtomEngine(args.n, args.l, args.m)
    print(atom)

    if args.static_prefix:
        atom.plot_radial(save_path=f"{args.static_prefix}_radial.png")
        atom.plot_cross_section(save_path=f"{args.static_prefix}_cross_section.png")
        print(f"Saved static plots with prefix '{args.static_prefix}'")

    if args.interactive:
        atom.render_offline_html(args.interactive, n_samples=args.samples)
        print(f"Saved offline interactive 3D viewer to {args.interactive}")


if __name__ == "__main__":
    main()
