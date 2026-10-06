#!/usr/bin/env python3
"""Run every deterministic and seeded audit reported in the manuscript."""
from __future__ import annotations
import argparse
import platform
import subprocess
import sys
from pathlib import Path

COMMANDS = [
    ("Cycle automaton certificate", ["verify_cycle_automaton_certificate.py", "cycle_automaton_certificate.json"]),
    ("q=2 separator-rank certificate", ["verify_separator_rank_certificate.py", "separator_rank_certificate.json"]),
    ("Root normal form, pendant transfer, cut vertices", ["zero_forcing_tree_dp.py", "--audit-atlas"]),
    ("All ordered rooted wedges", ["zero_forcing_rooted_audits.py", "--audit-wedges"]),
    ("Seeded random rooted wedges", ["zero_forcing_rooted_audits.py", "--audit-random-wedges", "350", "--seed", "20260726"]),
    ("Nonisomorphic trees through order 12", ["zero_forcing_tree_dp.py", "--audit-trees", "12"]),
    ("Bounded-block graph atlas", ["zero_forcing_block_dp.py", "--audit-atlas"]),
    ("Seeded random bounded-block graphs", ["zero_forcing_block_dp.py", "--audit-random", "500", "--seed", "20260726"]),
    ("Cycle automaton and marked words", ["zero_forcing_cactus_dp.py", "--audit-automaton"]),
    ("Cactus graph atlas", ["zero_forcing_cactus_dp.py", "--audit-atlas"]),
    ("Seeded random cactus graphs", ["zero_forcing_cactus_dp.py", "--audit-random", "300", "--seed", "20260726"]),
    ("Clique/cycle-block graph atlas", ["zero_forcing_cactus_dp.py", "--audit-mixed-atlas"]),
    ("Seeded random clique/cycle-block graphs", ["zero_forcing_cactus_dp.py", "--audit-mixed-random", "300", "--seed", "20260906"]),
    ("Independent replacement, clique, block, and rank checks", ["zero_forcing_extended_audits.py", "--package", "."]),
    ("Arbitrary-separator algebra and realization checks", ["zero_forcing_separator_audits.py"]),
]


def run(command: list[str], root: Path) -> None:
    rendered = " ".join([sys.executable, *command])
    print(f"\n$ {rendered}", flush=True)
    subprocess.run([sys.executable, *command], cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    print(f"Python: {platform.python_version()}")
    try:
        import networkx as nx
    except ImportError as exc:
        raise SystemExit("NetworkX is unavailable. Install requirements.txt first.") from exc
    print(f"NetworkX: {nx.__version__}")
    try:
        import sympy as sp
    except ImportError as exc:
        raise SystemExit("SymPy is unavailable. Install requirements.txt first.") from exc
    print(f"SymPy: {sp.__version__}")

    selected = [
        item for item in COMMANDS
        if not (args.skip_random and "random" in item[0].lower())
    ]
    if args.list:
        for label, command in selected:
            print(f"{label}: {sys.executable} {' '.join(command)}")
        return

    if not args.skip_checksums:
        run(["verify_checksums.py", "SHA256SUMS.txt"], root)

    for label, command in selected:
        print(f"\n=== {label} ===", flush=True)
        run(command, root)

    print("\nALL REQUESTED AUDITS PASSED.")


if __name__ == "__main__":
    main()
