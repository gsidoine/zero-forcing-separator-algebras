#!/usr/bin/env python3
"""Verify the finite q=2 separator-rank certificate from first principles.

The verifier reconstructs the chain states and the integer coefficient matrices
from the atomic compatibility definitions in zero_forcing_separator_algebra.py.
It then checks exact coefficient-span ranks over Q, canonical SHA-256 matrix
hashes, and explicit nonsingular minors after specialization at x=2 over the
finite field F_p.  No floating-point rank decision is used.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import sympy as sp

from zero_forcing_separator_algebra import (
    chain_states,
    evaluate_coefficient_matrices,
    kernel_coefficient_matrices,
    rank_mod_p,
)


def canonical_matrix_bytes(mats: list[sp.Matrix]) -> bytes:
    payload = []
    for M in mats:
        payload.append([[int(M[i, j]) for j in range(M.cols)] for i in range(M.rows)])
    return json.dumps(payload, separators=(",", ":"), sort_keys=False).encode("utf-8")


def matrix_hash(mats: list[sp.Matrix]) -> str:
    return hashlib.sha256(canonical_matrix_bytes(mats)).hexdigest()


def independent_indices_mod_p(M: sp.Matrix, p: int, *, transpose: bool = False) -> list[int]:
    """Return pivot-column indices over F_p; transpose=True gives independent rows."""
    A = M.T if transpose else M
    rows, cols = A.rows, A.cols
    a = [[int(A[i, j]) % p for j in range(cols)] for i in range(rows)]
    r = 0
    pivots: list[int] = []
    for c in range(cols):
        pivot = next((i for i in range(r, rows) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        inv = pow(a[r][c], p - 2, p)
        a[r] = [(v * inv) % p for v in a[r]]
        for i in range(rows):
            if i != r and a[i][c]:
                f = a[i][c]
                a[i] = [(u - f * v) % p for u, v in zip(a[i], a[r])]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return pivots


def det_mod_p_from_minor(M: sp.Matrix, rows: list[int], cols: list[int], p: int) -> int:
    A = [[int(M[i, j]) % p for j in cols] for i in rows]
    n = len(A)
    det = 1
    for c in range(n):
        pivot = next((i for i in range(c, n) if A[i][c]), None)
        if pivot is None:
            return 0
        if pivot != c:
            A[c], A[pivot] = A[pivot], A[c]
            det = (-det) % p
        pv = A[c][c] % p
        det = (det * pv) % p
        inv = pow(pv, p - 2, p)
        for i in range(c + 1, n):
            if A[i][c]:
                f = A[i][c] * inv % p
                for j in range(c, n):
                    A[i][j] = (A[i][j] - f * A[c][j]) % p
    return det % p


def reconstruct():
    states = chain_states(2)
    _, independent = kernel_coefficient_matrices(2, (), chains_only=True)
    _, edge = kernel_coefficient_matrices(2, ((0, 1),), chains_only=True)
    mobius = [A - B for A, B in zip(independent, edge)]
    return states, independent, edge, mobius


def verify(path: Path) -> None:
    cert = json.loads(path.read_text())
    states, independent, edge, mobius = reconstruct()
    p = int(cert["prime"])
    x0 = int(cert["specialization_x"])

    if len(states) != cert["chain_count_q2"]:
        raise AssertionError("chain-count mismatch")
    encoded_states = [sorted([list(atom) for atom in state]) for state in states]
    if encoded_states != cert["chain_states_q2"]:
        raise AssertionError("chain-state order mismatch")

    sectors = {
        "independent": independent,
        "edge_present": edge,
        "edge_mobius": mobius,
    }

    for name, mats in sectors.items():
        entry = cert["sectors"][name]
        joined = mats[0].row_join(mats[1]).row_join(mats[2])
        exact_rank = int(joined.rank())
        if exact_rank != int(entry["coefficient_span_rank_Q"]):
            raise AssertionError(f"{name}: exact coefficient-span rank mismatch")
        h = matrix_hash(mats)
        if h != entry["coefficient_matrices_sha256"]:
            raise AssertionError(f"{name}: coefficient-matrix hash mismatch")

        Mx = evaluate_coefficient_matrices(mats, x0)
        mod_rank = rank_mod_p(Mx, p)
        if mod_rank != int(entry["rank_mod_p_at_specialization"]):
            raise AssertionError(f"{name}: specialization rank mismatch")
        rows = list(map(int, entry["minor_rows"]))
        cols = list(map(int, entry["minor_cols"]))
        if len(rows) != mod_rank or len(cols) != mod_rank:
            raise AssertionError(f"{name}: minor size mismatch")
        det = det_mod_p_from_minor(Mx, rows, cols, p)
        if det == 0 or det != int(entry["minor_det_mod_p"]):
            raise AssertionError(f"{name}: certified minor determinant mismatch")

    if int(cert["full_rank_q2"]) != int(cert["sectors"]["edge_present"]["coefficient_span_rank_Q"]) + int(cert["sectors"]["edge_mobius"]["coefficient_span_rank_Q"]):
        raise AssertionError("full q=2 rank decomposition mismatch")

    print(f"certificate: {path.name}")
    print(f"chain states q=2: {len(states)}")
    for name in ("independent", "edge_present", "edge_mobius"):
        entry = cert["sectors"][name]
        print(
            f"{name}: coefficient-span rank {entry['coefficient_span_rank_Q']}, "
            f"rank mod {p} at x={x0} {entry['rank_mod_p_at_specialization']}, "
            f"minor determinant {entry['minor_det_mod_p']}"
        )
    print(f"full q=2 rank: {cert['full_rank_q2']}")
    print("separator-rank certificate verified")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python verify_separator_rank_certificate.py separator_rank_certificate.json")
    verify(Path(sys.argv[1]))
