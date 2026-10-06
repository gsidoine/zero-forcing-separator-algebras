"""Finite separator algebra for the zero-forcing polynomial.

This module implements the abstract boundary-atom calculus used in the manuscript.
Symbols are encoded as 0, 1, 2 where 2 means ``I'' (the boundary vertex lies
in the prefort).  The all-zero word is excluded from the nonfatal atom poset.

The code is intentionally independent of the tree/block/cycle implementation.
It supports exact q=1 and q=2 rank certificates and chain-count calculations
for arbitrary q.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import comb
from typing import FrozenSet, Iterable, Iterator, Sequence, Tuple

import sympy as sp

Atom = Tuple[int, ...]
Edge = Tuple[int, int]
State = FrozenSet[Atom]


def atoms(q: int) -> tuple[Atom, ...]:
    """Return P_q = {0,1,I}^q \\ {0^q}, with I encoded by 2."""
    return tuple(a for a in product(range(3), repeat=q) if any(x != 0 for x in a))


def atom_join(a: Atom, b: Atom) -> Atom:
    return tuple(max(x, y) for x, y in zip(a, b))


def leq(a: Atom, b: Atom) -> bool:
    return all(x <= y for x, y in zip(a, b))


def trace(a: Atom) -> FrozenSet[int]:
    return frozenset(i for i, x in enumerate(a) if x == 2)


def is_chain(state: State) -> bool:
    seq = tuple(state)
    return all(leq(a, b) or leq(b, a) for a, b in combinations(seq, 2))


def is_join_closed(state: State) -> bool:
    return all(atom_join(a, b) in state for a in state for b in state)


def join_closed_states(q: int) -> tuple[State, ...]:
    """Enumerate all join-closed subsets for q <= 2.

    The power-set enumeration is deliberately guarded because |P_3|=26.
    """
    P = atoms(q)
    if len(P) > 12:
        raise ValueError("brute-force state enumeration is intended only for q <= 2")
    out = []
    for mask in range(1 << len(P)):
        s = frozenset(P[i] for i in range(len(P)) if (mask >> i) & 1)
        if is_join_closed(s):
            out.append(s)
    return tuple(out)


def chain_states(q: int) -> tuple[State, ...]:
    if q <= 2:
        return tuple(s for s in join_closed_states(q) if is_chain(s))
    raise ValueError("explicit chain enumeration is intended only for q <= 2")


def normalize_edges(edges: Iterable[Edge]) -> FrozenSet[Edge]:
    return frozenset(tuple(sorted(e)) for e in edges)


def compatible(a: Atom, b: Atom, edges: FrozenSet[Edge]) -> bool:
    """Atomic compatibility after gluing two pieces with boundary-edge union edges."""
    if trace(a) != trace(b):
        return False
    T = trace(a)
    q = len(a)
    for i in range(q):
        if i in T:
            continue
        d = sum(1 for j in T if tuple(sorted((i, j))) in edges)
        # Outside the common trace, a_i,b_i are 0/1.
        if d + a[i] + b[i] == 1:
            return False
    return True


def obstruction_traces(A: State, D: State, edges: FrozenSet[Edge]) -> FrozenSet[FrozenSet[int]]:
    out = set()
    for a in A:
        for b in D:
            if compatible(a, b, edges):
                out.add(trace(a))
    return frozenset(out)


def kernel_coefficients(A: State, D: State, edges: FrozenSet[Edge], q: int) -> tuple[int, ...]:
    """Coefficients of the boundary-selection kernel K_E(A,D;x)."""
    traces = obstruction_traces(A, D, edges)
    coeff = [0] * (q + 1)
    B = range(q)
    for r in range(q + 1):
        for X_tuple in combinations(B, r):
            X = frozenset(X_tuple)
            if all(X.intersection(T) for T in traces):
                coeff[r] += 1
    return tuple(coeff)


def kernel_coefficient_matrices(q: int, edges: Iterable[Edge], *, chains_only: bool = False):
    """Exact coefficient matrices [K_0,...,K_q].

    For q <= 2 the state index can be all join-closed states or just chains.
    """
    states = chain_states(q) if chains_only else join_closed_states(q)
    E = normalize_edges(edges)
    mats = [sp.zeros(len(states), len(states)) for _ in range(q + 1)]
    for i, A in enumerate(states):
        for j, D in enumerate(states):
            coeff = kernel_coefficients(A, D, E, q)
            for k, c in enumerate(coeff):
                mats[k][i, j] = c
    return states, mats


def evaluate_coefficient_matrices(mats: Sequence[sp.Matrix], value) -> sp.Matrix:
    out = sp.zeros(mats[0].rows, mats[0].cols)
    for k, M in enumerate(mats):
        out += value**k * M
    return out


def coefficient_span_rank(mats: Sequence[sp.Matrix]) -> int:
    joined = mats[0]
    for M in mats[1:]:
        joined = joined.row_join(M)
    return int(joined.rank())


def q2_full_block_coefficients(*, chains_only: bool = False):
    """Coefficient matrices for the full ordered two-terminal connection kernel.

    Local boundary-edge status is 0/1 and the glued edge status is OR.
    """
    _, K0 = kernel_coefficient_matrices(2, (), chains_only=chains_only)
    _, K1 = kernel_coefficient_matrices(2, ((0, 1),), chains_only=chains_only)
    blocks = []
    for k in range(3):
        A, E = K0[k], K1[k]
        blocks.append(A.row_join(E).col_join(E.row_join(E)))
    return blocks


def strict_chain_count(q: int, k: int) -> int:
    """Number of strict k-element chains in C_3^q."""
    return sum(
        (-1) ** (k - j) * comb(k - 1, j - 1) * comb(j + 2, 2) ** q
        for j in range(1, k + 1)
    )


def chain_count(q: int) -> int:
    """Number c_q of chains (including empty) in C_3^q \\ {0^q}."""
    total_full = 1 + sum(strict_chain_count(q, k) for k in range(1, 2 * q + 2))
    assert total_full % 2 == 0
    return total_full // 2


def chain_count_table(max_q: int = 7) -> list[tuple[int, int]]:
    return [(q, chain_count(q)) for q in range(max_q + 1)]


def rank_mod_p(M: sp.Matrix, p: int = 1000003) -> int:
    """Fast exact rank over F_p for an integer matrix."""
    A = [[int(M[i, j]) % p for j in range(M.cols)] for i in range(M.rows)]
    r = 0
    for c in range(M.cols):
        pivot = next((i for i in range(r, M.rows) if A[i][c]), None)
        if pivot is None:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        inv = pow(A[r][c], p - 2, p)
        A[r] = [(v * inv) % p for v in A[r]]
        for i in range(M.rows):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [(u - f * v) % p for u, v in zip(A[i], A[r])]
        r += 1
        if r == M.rows:
            break
    return r


def q2_rank_certificate() -> dict[str, object]:
    """Return q=2 modular rank certificates used in the audit.

    The manuscript's upper bounds are proved structurally; these modular ranks
    certify matching lower bounds and are independent of floating point.
    """
    _, empty = kernel_coefficient_matrices(2, (), chains_only=True)
    _, edge = kernel_coefficient_matrices(2, ((0, 1),), chains_only=True)
    E0 = evaluate_coefficient_matrices(empty, 2)
    E1 = evaluate_coefficient_matrices(edge, 2)
    full_coeff = q2_full_block_coefficients(chains_only=True)
    full_at_2 = evaluate_coefficient_matrices(full_coeff, 2)
    B0, B1, B2 = full_coeff
    concat = B0.row_join(B1).row_join(B2)
    empty_concat = empty[0].row_join(empty[1]).row_join(empty[2])
    edge_concat = edge[0].row_join(edge[1]).row_join(edge[2])
    diff_mats = [a - b for a, b in zip(empty, edge)]
    diff_concat = diff_mats[0].row_join(diff_mats[1]).row_join(diff_mats[2])
    return {
        "prime": 1000003,
        "chain_count_q2": len(chain_states(2)),
        "independent_boundary_coefficient_span_rank_Q": empty_concat.rank(),
        "edge_boundary_coefficient_span_rank_Q": edge_concat.rank(),
        "edge_mobius_coefficient_span_rank_Q": diff_concat.rank(),
        "full_coefficient_span_rank_Q": concat.rank(),
        "independent_boundary_rank_mod_p_at_x_2": rank_mod_p(E0),
        "edge_boundary_rank_mod_p_at_x_2": rank_mod_p(E1),
        "edge_mobius_correction_rank_mod_p_at_x_2": rank_mod_p(E0 - E1),
        "full_rank_mod_p_at_x_2": rank_mod_p(full_at_2),
        "full_coefficient_ranks_mod_p": tuple(rank_mod_p(M) for M in full_coeff),
    }


if __name__ == "__main__":
    print("chain counts:")
    for q, c in chain_count_table(7):
        print(f"  c_{q} = {c}")
    print("q=2 rank certificate:")
    for key, value in q2_rank_certificate().items():
        print(f"  {key}: {value}")
