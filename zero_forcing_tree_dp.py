#!/usr/bin/env python3
"""
Exact zero-forcing polynomial of a tree using the four-state root-response algebra.

The core algorithm uses no third-party libraries. NetworkX is used only by
optional verification and example helpers.

Examples
--------
python zero_forcing_tree_dp.py --path 8
python zero_forcing_tree_dp.py --star 6
python zero_forcing_tree_dp.py --random-tree 1000 --seed 20260726
python zero_forcing_tree_dp.py --audit-trees 12
python zero_forcing_tree_dp.py --audit-atlas
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


Polynomial = List[int]


def poly_trim(p: Polynomial) -> Polynomial:
    q = list(p)
    while len(q) > 1 and q[-1] == 0:
        q.pop()
    return q


def poly_add(*polynomials: Polynomial) -> Polynomial:
    size = max((len(p) for p in polynomials), default=1)
    result = [0] * size
    for p in polynomials:
        for i, coefficient in enumerate(p):
            result[i] += coefficient
    return poly_trim(result)


def poly_sub(p: Polynomial, q: Polynomial) -> Polynomial:
    size = max(len(p), len(q))
    result = [0] * size
    for i in range(size):
        result[i] = (
            (p[i] if i < len(p) else 0)
            - (q[i] if i < len(q) else 0)
        )
    return poly_trim(result)


def poly_mul(p: Polynomial, q: Polynomial) -> Polynomial:
    if not p or not q:
        return [0]
    result = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            if b:
                result[i + j] += a * b
    return poly_trim(result)


def poly_shift(p: Polynomial, amount: int = 1) -> Polynomial:
    return [0] * amount + list(p)


def polynomial_to_string(p: Polynomial, variable: str = "x") -> str:
    terms = []
    for exponent, coefficient in enumerate(p):
        if coefficient == 0:
            continue
        if exponent == 0:
            term = str(coefficient)
        elif exponent == 1:
            term = variable if coefficient == 1 else f"{coefficient}{variable}"
        else:
            term = (
                f"{variable}^{exponent}"
                if coefficient == 1
                else f"{coefficient}{variable}^{exponent}"
            )
        terms.append(term)
    return " + ".join(reversed(terms)) if terms else "0"


@dataclass(frozen=True)
class Profile:
    """
    Four rooted response polynomials.

    P0: passive, cannot source the initially white root
    P1: passive, can source the initially white root
    A0: active, cannot source the initially white root
    A1: active, can source the initially white root
    """

    P0: Polynomial
    P1: Polynomial
    A0: Polynomial
    A1: Polynomial

    @property
    def P(self) -> Polynomial:
        return poly_add(self.P0, self.P1)

    @property
    def A(self) -> Polynomial:
        return poly_add(self.A0, self.A1)


SINGLETON_PROFILE = Profile(P0=[1], P1=[0], A0=[0], A1=[0])


def wedge_product(left: Profile, right: Profile) -> Profile:
    """
    Exact rooted one-point-union product.
    """
    P = poly_mul(left.P, right.P)
    P0 = poly_mul(left.P0, right.P0)

    A = poly_add(
        poly_mul(left.A, right.P),
        poly_mul(left.P, right.A),
    )
    A0 = poly_add(
        poly_mul(left.A0, right.P0),
        poly_mul(left.P0, right.A0),
    )

    return Profile(
        P0=P0,
        P1=poly_sub(P, P0),
        A0=A0,
        A1=poly_sub(A, A0),
    )


def edge_extend(old: Profile) -> Profile:
    """
    Add a new pendant root adjacent to the old root.

    New formulas:
      P0' = A1 + x A
      P1' = P1 + x P
      A0' = P0 + A0
      A1' = 0
    """
    return Profile(
        P0=poly_add(old.A1, poly_shift(old.A)),
        P1=poly_add(old.P1, poly_shift(old.P)),
        A0=poly_add(old.P0, old.A0),
        A1=[0],
    )


def full_polynomial(profile: Profile) -> Polynomial:
    """
    Z(H;x) = (1+x)(P+A) - (P0+A0).
    """
    successful_with_root_blue = poly_add(profile.P, profile.A)
    source_zero = poly_add(profile.P0, profile.A0)
    return poly_sub(
        poly_mul([1, 1], successful_with_root_blue),
        source_zero,
    )


def validate_tree_adjacency(adjacency: Dict[int, Iterable[int]]) -> Dict[int, List[int]]:
    normalized = {v: sorted(set(neighbors)) for v, neighbors in adjacency.items()}

    if not normalized:
        raise ValueError("The tree must contain at least one vertex.")

    for v, neighbors in normalized.items():
        if v in neighbors:
            raise ValueError("Loops are not allowed.")
        for u in neighbors:
            if u not in normalized:
                raise ValueError(f"Neighbor {u} is missing from the adjacency dictionary.")
            if v not in normalized[u]:
                raise ValueError("Adjacency must be symmetric.")

    edge_count = sum(len(neighbors) for neighbors in normalized.values()) // 2
    n = len(normalized)
    if edge_count != n - 1:
        raise ValueError("The input does not have n-1 edges.")

    start = next(iter(normalized))
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for u in normalized[v]:
            if u not in seen:
                seen.add(u)
                stack.append(u)

    if len(seen) != n:
        raise ValueError("The input graph is disconnected.")

    return normalized


def tree_zero_forcing_polynomial(
    adjacency: Dict[int, Iterable[int]],
    root: int | None = None,
) -> Polynomial:
    """
    Compute the exact zero-forcing polynomial of a tree.

    Complexity: O(n^2) coefficient-arithmetic operations with schoolbook
    convolution.
    """
    tree = validate_tree_adjacency(adjacency)

    if root is None:
        root = next(iter(tree))
    if root not in tree:
        raise ValueError("The specified root is not a vertex of the tree.")

    def profile(v: int, parent: int | None) -> Profile:
        result = SINGLETON_PROFILE
        for child in tree[v]:
            if child == parent:
                continue
            child_profile = profile(child, v)
            branch_profile = edge_extend(child_profile)
            result = wedge_product(result, branch_profile)
        return result

    return full_polynomial(profile(root, None))


def adjacency_from_edges(
    edges: Iterable[Tuple[int, int]],
    vertices: Iterable[int] | None = None,
) -> Dict[int, List[int]]:
    adjacency: Dict[int, List[int]] = {}
    if vertices is not None:
        adjacency = {v: [] for v in vertices}

    for u, v in edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    return adjacency


# ---------------------------------------------------------------------------
# Independent brute-force verifier
# ---------------------------------------------------------------------------

def zero_forcing_closure(
    adjacency: Dict[int, Sequence[int]],
    initial: Iterable[int],
    forbidden_forcer: int | None = None,
) -> frozenset[int]:
    blue = set(initial)

    changed = True
    while changed:
        changed = False
        for u in list(blue):
            if u == forbidden_forcer:
                continue
            white = [v for v in adjacency[u] if v not in blue]
            if len(white) == 1:
                blue.add(white[0])
                changed = True
                break

    return frozenset(blue)


def brute_force_polynomial(
    adjacency: Dict[int, Iterable[int]],
) -> Polynomial:
    graph = validate_tree_adjacency(adjacency)
    vertices = list(graph)
    counts = [0] * (len(vertices) + 1)

    for mask in range(1 << len(vertices)):
        initial = {
            vertices[i]
            for i in range(len(vertices))
            if (mask >> i) & 1
        }
        if len(zero_forcing_closure(graph, initial)) == len(vertices):
            counts[len(initial)] += 1

    return poly_trim(counts)


def path_adjacency(n: int) -> Dict[int, List[int]]:
    if n < 1:
        raise ValueError("n must be positive")
    return adjacency_from_edges(
        ((i, i + 1) for i in range(n - 1)),
        vertices=range(n),
    )


def star_adjacency(leaves: int) -> Dict[int, List[int]]:
    if leaves < 1:
        raise ValueError("The number of leaves must be positive.")
    return adjacency_from_edges(
        ((0, i) for i in range(1, leaves + 1)),
        vertices=range(leaves + 1),
    )


def random_tree_adjacency(n: int, seed: int) -> Dict[int, List[int]]:
    if n < 1:
        raise ValueError("n must be positive")
    rng = random.Random(seed)
    if n == 1:
        return {0: []}

    # Prüfer-sequence construction.
    prufer = [rng.randrange(n) for _ in range(n - 2)]
    degree = [1] * n
    for v in prufer:
        degree[v] += 1

    import heapq

    leaves = [v for v, d in enumerate(degree) if d == 1]
    heapq.heapify(leaves)
    edges = []

    for v in prufer:
        leaf = heapq.heappop(leaves)
        edges.append((leaf, v))
        degree[leaf] -= 1
        degree[v] -= 1
        if degree[v] == 1:
            heapq.heappush(leaves, v)

    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    edges.append((u, v))

    return adjacency_from_edges(edges, vertices=range(n))


def audit_nonisomorphic_trees(max_n: int) -> None:
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "NetworkX is required for --audit-trees. Install it with "
            "'pip install networkx'."
        ) from exc

    checked = 0
    start = time.perf_counter()

    for n in range(1, max_n + 1):
        trees = [nx.empty_graph(1)] if n == 1 else nx.nonisomorphic_trees(n)
        count = 0

        for tree in trees:
            adjacency = {
                v: list(tree.neighbors(v))
                for v in tree.nodes()
            }
            exact = brute_force_polynomial(adjacency)
            dynamic = tree_zero_forcing_polynomial(adjacency, root=0)
            if exact != dynamic:
                raise AssertionError(
                    f"Mismatch for n={n}, edges={list(tree.edges())}: "
                    f"brute={exact}, dynamic={dynamic}"
                )
            count += 1
            checked += 1

        print(f"n={n}: {count} nonisomorphic trees passed")

    elapsed = time.perf_counter() - start
    print(f"Audit passed for {checked} trees in {elapsed:.3f} seconds.")


def audit_graph_atlas() -> None:
    """
    Exhaustively audit:
      - root normal form,
      - edge extension,
      - cut-vertex formula,
    on connected graph-atlas graphs of order at most seven.

    This verifier uses a generic rooted-profile enumerator, separate from the
    tree dynamic program.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "NetworkX is required for --audit-atlas. Install it with "
            "'pip install networkx'."
        ) from exc

    from collections import Counter

    def graph_closure(G, initial, forbidden=None):
        blue = set(initial)
        changed = True
        while changed:
            changed = False
            for u in list(blue):
                if u == forbidden:
                    continue
                white = [v for v in G.neighbors(u) if v not in blue]
                if len(white) == 1:
                    blue.add(white[0])
                    changed = True
                    break
        return frozenset(blue)

    def category(G, root, initial):
        minus = graph_closure(G, initial, forbidden=root)
        source = int(root in minus)

        plus = graph_closure(
            G,
            set(initial) | {root},
            forbidden=root,
        )
        if len(plus) == len(G):
            return "P", source

        white_neighbors = [
            v for v in G.neighbors(root)
            if v not in plus
        ]
        if len(white_neighbors) == 1:
            completed = graph_closure(
                G,
                set(plus) | {white_neighbors[0]},
                forbidden=root,
            )
            if len(completed) == len(G):
                return "A", source

        return "F", source

    def generic_profile(G, root):
        vertices = [v for v in G if v != root]
        counts = {
            ("P", 0): Counter(),
            ("P", 1): Counter(),
            ("A", 0): Counter(),
            ("A", 1): Counter(),
        }

        for mask in range(1 << len(vertices)):
            initial = {
                vertices[i]
                for i in range(len(vertices))
                if (mask >> i) & 1
            }
            state, source = category(G, root, initial)
            if state in ("P", "A"):
                counts[(state, source)][len(initial)] += 1

        def as_poly(counter):
            p = [0] * (len(vertices) + 1)
            for degree, value in counter.items():
                p[degree] = value
            return poly_trim(p)

        return Profile(
            P0=as_poly(counts[("P", 0)]),
            P1=as_poly(counts[("P", 1)]),
            A0=as_poly(counts[("A", 0)]),
            A1=as_poly(counts[("A", 1)]),
        )

    def brute_graph_poly(G):
        vertices = list(G)
        counts = [0] * (len(vertices) + 1)
        for mask in range(1 << len(vertices)):
            initial = {
                vertices[i]
                for i in range(len(vertices))
                if (mask >> i) & 1
            }
            if len(graph_closure(G, initial)) == len(vertices):
                counts[len(initial)] += 1
        return poly_trim(counts)

    rooted_cases = 0
    extension_cases = 0
    cut_cases = 0

    for G in nx.graph_atlas_g():
        if len(G) == 0 or not nx.is_connected(G):
            continue

        exact = brute_graph_poly(G)

        for root in G:
            profile = generic_profile(G, root)
            if full_polynomial(profile) != exact:
                raise AssertionError(
                    f"Root formula failed: n={len(G)}, root={root}"
                )
            rooted_cases += 1

            mapping = {v: v + 1 for v in G}
            H = nx.relabel_nodes(G, mapping)
            old_root = mapping[root]
            H.add_node(0)
            H.add_edge(0, old_root)

            observed = generic_profile(H, 0)
            predicted = edge_extend(profile)
            if observed != predicted:
                raise AssertionError(
                    f"Edge extension failed: n={len(G)}, root={root}"
                )
            extension_cases += 1

        for root in nx.articulation_points(G):
            components = list(nx.connected_components(nx.subgraph_view(
                G,
                filter_node=lambda v, root=root: v != root,
            )))
            result = SINGLETON_PROFILE
            for component in components:
                branch = G.subgraph(set(component) | {root}).copy()
                result = wedge_product(
                    result,
                    generic_profile(branch, root),
                )
            if full_polynomial(result) != exact:
                raise AssertionError(
                    f"Cut formula failed: n={len(G)}, root={root}"
                )
            cut_cases += 1

    print(f"Root normal-form cases passed: {rooted_cases}")
    print(f"Edge-extension cases passed: {extension_cases}")
    print(f"Cut-vertex cases passed: {cut_cases}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()

    group.add_argument("--path", type=int, metavar="N")
    group.add_argument("--star", type=int, metavar="LEAVES")
    group.add_argument("--random-tree", type=int, metavar="N")
    group.add_argument("--audit-trees", type=int, metavar="MAX_N")
    group.add_argument("--audit-atlas", action="store_true")

    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--verify-bruteforce",
        action="store_true",
        help="For a displayed example, compare with exhaustive enumeration.",
    )

    args = parser.parse_args()

    if args.audit_trees is not None:
        audit_nonisomorphic_trees(args.audit_trees)
        return

    if args.audit_atlas:
        audit_graph_atlas()
        return

    if args.path is not None:
        adjacency = path_adjacency(args.path)
        label = f"P_{args.path}"
    elif args.star is not None:
        adjacency = star_adjacency(args.star)
        label = f"K_{{1,{args.star}}}"
    elif args.random_tree is not None:
        adjacency = random_tree_adjacency(args.random_tree, args.seed)
        label = f"random tree n={args.random_tree}, seed={args.seed}"
    else:
        adjacency = path_adjacency(8)
        label = "P_8"

    start = time.perf_counter()
    polynomial = tree_zero_forcing_polynomial(adjacency)
    elapsed = time.perf_counter() - start

    print(label)
    print("Coefficient list:", polynomial)
    if len(polynomial) <= 50:
        print("Polynomial:", polynomial_to_string(polynomial))
    print("Minimum zero-forcing number:",
          next(i for i, value in enumerate(polynomial) if value))
    print("Total zero-forcing sets:", sum(polynomial))
    print(f"Dynamic-programming time: {elapsed:.6f} seconds")

    if args.verify_bruteforce:
        if len(adjacency) > 24:
            raise ValueError(
                "Brute-force verification is disabled above 24 vertices."
            )
        brute = brute_force_polynomial(adjacency)
        print("Brute-force polynomial:", brute)
        print("Match:", brute == polynomial)


if __name__ == "__main__":
    main()
