"""Independent audits for the arbitrary-separator theory.

The checks here are intentionally small and exhaustive.  They do not prove the
paper's general theorems; they are designed to catch convention/signature errors.
"""
from __future__ import annotations

from itertools import combinations, product
import random

import networkx as nx

from zero_forcing_separator_algebra import (
    atom_join,
    atoms,
    chain_count,
    chain_states,
    compatible,
    is_chain,
    kernel_coefficient_matrices,
    q2_rank_certificate,
    trace,
)


def is_prefort(G: nx.Graph, B: set, S: set, F: set) -> bool:
    if not F or (F & S):
        return False
    for v in G.nodes:
        if v in F or v in B:
            continue
        if len(set(G.neighbors(v)) & F) == 1:
            return False
    return True


def atomic_refinements(G: nx.Graph, B_order: list, F: set):
    choices = []
    B = set(B_order)
    for b in B_order:
        if b in F:
            choices.append((2,))
            continue
        c = len((set(G.neighbors(b)) - B) & F)
        if c == 0:
            choices.append((0,))
        elif c == 1:
            choices.append((1,))
        else:
            choices.append((0, 1))
    return {tuple(v) for v in product(*choices)}


def atom_family(G: nx.Graph, B_order: list, S: set):
    B = set(B_order)
    V = [v for v in G.nodes if v not in S]
    out = set()
    for mask in range(1, 1 << len(V)):
        F = {V[i] for i in range(len(V)) if (mask >> i) & 1}
        if is_prefort(G, B, S, F):
            out |= atomic_refinements(G, B_order, F)
    # fatal all-zero is deliberately retained for auditing.
    return out


def add_equality(G, selected, u, v, name):
    c = ("eq", name)
    G.add_node(c)
    G.add_edges_from([(c, u), (c, v)])
    selected.add(c)


def add_implication(G, selected, u, v, name):
    """Enforce u <= v for prefort membership."""
    vp = ("dup", name)
    G.add_node(vp)
    add_equality(G, selected, v, vp, (name, "copy"))
    c = ("imp", name)
    G.add_node(c)
    G.add_edges_from([(c, u), (c, v), (c, vp)])
    selected.add(c)


def realize_chain(chain, q):
    """Construct the explicit decorated graph used in the realization lemma."""
    chain = sorted(chain, key=lambda a: (sum(a), a))
    # Since input is a chain, coordinatewise order agrees with this after checking.
    for a, b in zip(chain, chain[1:]):
        assert all(x <= y for x, y in zip(a, b))

    G = nx.Graph()
    B = [("b", i) for i in range(q)]
    G.add_nodes_from(B)
    selected = set()
    k = len(chain)
    if k == 0:
        # Pin every boundary vertex out and leave no other unselected vertex.
        for i, b in enumerate(B):
            p = ("pin", i)
            G.add_edge(p, b)
            selected.add(p)
        return G, B, selected

    t = [("t", j) for j in range(k)]
    G.add_nodes_from(t)
    for j in range(k - 1):
        # t_{j+1} <= t_j
        add_implication(G, selected, t[j + 1], t[j], ("chain", j))

    for i, b in enumerate(B):
        vals = [a[i] for a in chain]
        one_positions = [j for j, v in enumerate(vals) if v == 1]
        I_positions = [j for j, v in enumerate(vals) if v == 2]

        if one_positions:
            a = one_positions[0]
            m = ("marker", i)
            G.add_node(m)
            G.add_edge(m, b)
            add_equality(G, selected, m, t[a], ("mark", i))

        if I_positions:
            bpos = I_positions[0]
            add_equality(G, selected, b, t[bpos], ("boundary", i))
        else:
            p = ("pin", i)
            G.add_edge(p, b)
            selected.add(p)

    return G, B, selected


def audit_chain_realization_q2():
    checks = 0
    for C in chain_states(2):
        G, B, S = realize_chain(C, 2)
        fam = atom_family(G, B, S)
        if (0, 0) in fam:
            raise AssertionError(f"fatal atom in realization of {C}")
        if fam != set(C):
            raise AssertionError(f"chain realization mismatch: wanted {C}, got {fam}")
        checks += 1
    return checks


def audit_chain_realization_q3(samples=40, seed=20260907):
    rng = random.Random(seed)
    P = atoms(3)
    # Generate random chains by sampling a random monotone walk and subchains.
    checks = 0
    for _ in range(samples):
        cur = [0, 0, 0]
        walk = []
        events = [(i, step) for i in range(3) for step in (1, 2)]
        # Random valid order: repeatedly choose an available coordinate increment.
        remaining = [0, 0, 0]
        while any(v < 2 for v in remaining):
            avail = [i for i, v in enumerate(remaining) if v < 2]
            i = rng.choice(avail)
            remaining[i] += 1
            walk.append(tuple(remaining))
        sub = [a for a in walk if rng.random() < 0.65 and any(a)]
        # Deduplicate, keep order; empty chain allowed.
        C = frozenset(sub)
        assert is_chain(C)
        G, B, S = realize_chain(C, 3)
        fam = atom_family(G, B, S)
        if (0, 0, 0) in fam or fam != set(C):
            raise AssertionError(f"q3 chain realization mismatch: wanted {C}, got {fam}")
        checks += 1
    return checks


def audit_join_compatibility_closure(q=2):
    P = atoms(q)
    E_all = [frozenset(), frozenset({(0, 1)})] if q == 2 else [frozenset()]
    checks = 0
    for E in E_all:
        for a1, a2, b1, b2 in product(P, repeat=4):
            if compatible(a1, b1, E) and compatible(a2, b2, E):
                if not compatible(atom_join(a1, a2), atom_join(b1, b2), E):
                    raise AssertionError((E, a1, a2, b1, b2))
            checks += 1
    return checks



def zero_forcing_closure(G: nx.Graph, S: set) -> set:
    blue = set(S)
    changed = True
    while changed:
        changed = False
        for u in list(blue):
            white = [v for v in G.neighbors(u) if v not in blue]
            if len(white) == 1:
                blue.add(white[0])
                changed = True
    return blue


def audit_worked_q2_example():
    """Verify the worked q=2 example printed in the manuscript."""
    # H is the 4-cycle a-v-b-c-a with boundary a,b and selected c.
    H = nx.Graph()
    a, b, v, c = "a", "b", "v", "c"
    H.add_edges_from([(a, v), (v, b), (b, c), (c, a)])
    AH = atom_family(H, [a, b], {c})
    if AH != {(1, 1), (2, 2)}:
        raise AssertionError(f"worked H atom family mismatch: {AH}")

    # K is a length-two path a-w-b with selected middle vertex w.
    K = nx.Graph()
    w = "w"
    K.add_edges_from([(a, w), (w, b)])
    AK = atom_family(K, [a, b], {w})
    if AK != {(2, 2)}:
        raise AssertionError(f"worked K atom family mismatch: {AK}")

    # Independent boundary: only equal atoms are compatible.  The only
    # compatible pair is II/II, whose trace is {a,b}; its transversal
    # polynomial is therefore 2x+x^2.
    traces = set()
    for xatom in AH:
        for yatom in AK:
            if compatible(xatom, yatom, frozenset()):
                traces.add(trace(xatom))
    if traces != {frozenset({0, 1})}:
        raise AssertionError(f"worked trace family mismatch: {traces}")

    coeff = [0, 0, 0]
    for r in range(3):
        for X in combinations(range(2), r):
            X = frozenset(X)
            if all(X & T for T in traces):
                coeff[r] += 1
    if tuple(coeff) != (0, 2, 1):
        raise AssertionError(f"worked kernel mismatch: {coeff}")

    glued = nx.compose(H, K)
    direct = [0, 0, 0]
    for r in range(3):
        for X in combinations([a, b], r):
            initial = {c, w, *X}
            if zero_forcing_closure(glued, initial) == set(glued.nodes):
                direct[r] += 1
    if tuple(direct) != (0, 2, 1):
        raise AssertionError(f"worked direct zero-forcing mismatch: {direct}")
    return {"H_atoms": sorted(AH), "K_atoms": sorted(AK), "kernel_coefficients": tuple(coeff), "direct_coefficients": tuple(direct)}

def main():
    print("separator algebra audits")
    print("  chain counts:", [chain_count(q) for q in range(8)])
    print("  q=2 rank certificate:", q2_rank_certificate())
    print("  worked q=2 example:", audit_worked_q2_example())
    print("  q=2 chain realizations:", audit_chain_realization_q2())
    print("  q=3 sampled chain realizations:", audit_chain_realization_q3())
    print("  compatibility join-closure checks:", audit_join_compatibility_closure(2))
    print("all separator audits passed")


if __name__ == "__main__":
    main()
