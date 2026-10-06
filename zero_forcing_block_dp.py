#!/usr/bin/env python3
"""
Exact zero-forcing polynomial by block-cut dynamic programming.

Main theorem implemented here
-----------------------------
If every block of G has at most b vertices, the complete zero-forcing
polynomial can be computed in O(4^b n^2) coefficient-arithmetic operations
(using ordinary convolution).  The implementation aggregates descendant
states and the local vertex-selection choice into four effective letters.
Complete blocks use a separate linear-scan clique transfer.  The general
kernel also works on arbitrary finite graphs, with running time exponential
only in the largest block.

Requirements
------------
    pip install networkx

Examples
--------
    python zero_forcing_block_dp.py --cycle-chain 8 5
    python zero_forcing_block_dp.py --audit-atlas
    python zero_forcing_block_dp.py --audit-random 300 --seed 20260726
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx

Vertex = Hashable
Polynomial = List[int]


def trim(p: Sequence[int]) -> Polynomial:
    q = list(p)
    while len(q) > 1 and q[-1] == 0:
        q.pop()
    return q


def poly_add(*polynomials: Sequence[int]) -> Polynomial:
    size = max((len(p) for p in polynomials), default=1)
    out = [0] * size
    for p in polynomials:
        for i, coefficient in enumerate(p):
            out[i] += coefficient
    return trim(out)


def poly_sub(p: Sequence[int], q: Sequence[int]) -> Polynomial:
    size = max(len(p), len(q))
    return trim([
        (p[i] if i < len(p) else 0)
        - (q[i] if i < len(q) else 0)
        for i in range(size)
    ])


def poly_mul(p: Sequence[int], q: Sequence[int]) -> Polynomial:
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            if b:
                out[i + j] += a * b
    return trim(out)


def poly_shift(p: Sequence[int], amount: int) -> Polynomial:
    return [0] * amount + list(p)


def polynomial_to_string(p: Sequence[int], variable: str = "x") -> str:
    terms: List[str] = []
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
    """The four rooted response polynomials."""

    P0: Tuple[int, ...]
    P1: Tuple[int, ...]
    A0: Tuple[int, ...]
    A1: Tuple[int, ...]

    @staticmethod
    def make(
        P0: Sequence[int] = (1,),
        P1: Sequence[int] = (0,),
        A0: Sequence[int] = (0,),
        A1: Sequence[int] = (0,),
    ) -> "Profile":
        return Profile(
            tuple(trim(P0)),
            tuple(trim(P1)),
            tuple(trim(A0)),
            tuple(trim(A1)),
        )

    @property
    def P(self) -> Polynomial:
        return poly_add(self.P0, self.P1)

    @property
    def A(self) -> Polynomial:
        return poly_add(self.A0, self.A1)

    def state_polynomial(self, state: str) -> Tuple[int, ...]:
        return getattr(self, state)


SINGLETON_PROFILE = Profile.make()
STATE_NAMES = ("P0", "P1", "A0", "A1")


def wedge_product(left: Profile, right: Profile) -> Profile:
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
    return Profile.make(
        P0=P0,
        P1=poly_sub(P, P0),
        A0=A0,
        A1=poly_sub(A, A0),
    )


def full_polynomial(profile: Profile) -> Polynomial:
    return poly_sub(
        poly_mul([1, 1], poly_add(profile.P, profile.A)),
        poly_add(profile.P0, profile.A0),
    )


def restricted_closure(
    graph: nx.Graph,
    initial: Iterable[Vertex],
    forbidden_root: Vertex,
) -> frozenset[Vertex]:
    blue = set(initial)
    changed = True
    while changed:
        changed = False
        for u in list(blue):
            if u == forbidden_root:
                continue
            white = [v for v in graph.neighbors(u) if v not in blue]
            if len(white) == 1:
                blue.add(white[0])
                changed = True
                break
    return frozenset(blue)


def response_category(
    graph: nx.Graph,
    root: Vertex,
    initial_without_root: Iterable[Vertex],
) -> Tuple[str, int]:
    """Return (P/A/F, source bit) for a concrete rooted configuration."""
    initial = set(initial_without_root)
    minus = restricted_closure(graph, initial, root)
    source = int(root in minus)

    plus = restricted_closure(graph, initial | {root}, root)
    if len(plus) == graph.number_of_nodes():
        return "P", source

    white_neighbors = [v for v in graph.neighbors(root) if v not in plus]
    if len(white_neighbors) == 1:
        completed = restricted_closure(
            graph,
            set(plus) | {white_neighbors[0]},
            root,
        )
        if len(completed) == graph.number_of_nodes():
            return "A", source

    return "F", source


LETTERS: Tuple[Tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1))


def local_letter_weights(profile: Profile) -> Dict[Tuple[int, int], Polynomial]:
    """Aggregate descendant state and selection of its attachment vertex.

    A letter is (effective-blue bit, pending-demand bit).  The four weights are
    exactly the source--demand aggregation used by the cycle transfer.
    """
    return {
        (0, 0): list(profile.P0),
        (1, 0): poly_add(
            poly_shift(profile.P0, 1),
            profile.P1,
            poly_shift(profile.P1, 1),
        ),
        (0, 1): list(profile.A0),
        (1, 1): poly_add(
            poly_shift(profile.A0, 1),
            profile.A1,
            poly_shift(profile.A1, 1),
        ),
    }


def effective_block_transfer(
    block_graph: nx.Graph,
    root: Vertex,
    child_profiles: Mapping[Vertex, Profile],
) -> Profile:
    """Exact four-letter kernel for an arbitrary block.

    A child articulation has four effective letters; an ordinary block vertex
    has only the two non-demand letters 00 and 10.  Hence a block of order m
    with q child articulations enumerates 4^q 2^(m-1-q) configurations, at
    most 4^(m-1), rather than the unaggregated 2^(m-1)4^q kernel.
    """
    vertices = list(block_graph.nodes())
    if root not in block_graph:
        raise ValueError("The root must belong to the block.")
    if any(v not in block_graph or v == root for v in child_profiles):
        raise ValueError("Child profiles must be rooted at non-root block vertices.")

    nonroot_vertices = [v for v in vertices if v != root]
    output: Dict[Tuple[str, int], Polynomial] = {
        ("P", 0): [0], ("P", 1): [0], ("A", 0): [0], ("A", 1): [0]
    }

    choices = []
    weights_by_vertex: Dict[Vertex, Dict[Tuple[int, int], Polynomial]] = {}
    for vertex in nonroot_vertices:
        if vertex in child_profiles:
            weights = local_letter_weights(child_profiles[vertex])
            choices.append(LETTERS)
        else:
            # No descendant: selecting the block vertex has weight x.
            weights = {(0, 0): [1], (1, 0): [0, 1]}
            choices.append(((0, 0), (1, 0)))
        weights_by_vertex[vertex] = weights

    for word in itertools.product(*choices):
        marker_graph = block_graph.copy()
        initial = set()
        weight: Polynomial = [1]

        for index, (vertex, letter) in enumerate(zip(nonroot_vertices, word)):
            letter_weight = weights_by_vertex[vertex][letter]
            if all(coefficient == 0 for coefficient in letter_weight):
                weight = [0]
                break
            weight = poly_mul(weight, letter_weight)
            blue, demand = letter
            if blue:
                initial.add(vertex)
            if demand:
                marker_graph.add_edge(vertex, ("pending-demand", index, vertex))

        if weight == [0]:
            continue
        category, source_bit = response_category(marker_graph, root, initial)
        if category != "F":
            key = (category, source_bit)
            output[key] = poly_add(output[key], weight)

    return Profile.make(
        P0=output[("P", 0)], P1=output[("P", 1)],
        A0=output[("A", 0)], A1=output[("A", 1)],
    )


def clique_transfer(
    block_graph: nx.Graph,
    root: Vertex,
    child_profiles: Mapping[Vertex, Profile],
) -> Profile:
    """Linear-scan transfer for a complete block, valid for every order >= 2."""
    if root not in block_graph:
        raise ValueError("The root must belong to the block.")
    m = block_graph.number_of_nodes()
    if m < 2 or block_graph.number_of_edges() != m * (m - 1) // 2:
        raise ValueError("clique_transfer requires a complete block of order at least two.")
    if any(v not in block_graph or v == root for v in child_profiles):
        raise ValueError("Child profiles must be rooted at non-root block vertices.")

    # B = all effective vertices blue; D = all blue and demanded;
    # W = exactly one effective white vertex; E = exactly one white and every
    # blue effective vertex demanded.  Updates use the old accumulators.
    B: Polynomial = [1]
    D: Polynomial = [1]
    W: Polynomial = [0]
    E: Polynomial = [0]

    for vertex in block_graph:
        if vertex == root:
            continue
        weights = local_letter_weights(child_profiles.get(vertex, SINGLETON_PROFILE))
        blue = poly_add(weights[(1, 0)], weights[(1, 1)])
        demanded_blue = weights[(1, 1)]
        white = poly_add(weights[(0, 0)], weights[(0, 1)])
        old_B, old_D, old_W, old_E = B, D, W, E
        B = poly_mul(old_B, blue)
        D = poly_mul(old_D, demanded_blue)
        W = poly_add(poly_mul(old_W, blue), poly_mul(old_B, white))
        E = poly_add(poly_mul(old_E, demanded_blue), poly_mul(old_D, white))

    return Profile.make(
        P0=poly_sub(poly_add(D, W), E),
        P1=poly_sub(B, D),
        A0=E,
        A1=[0],
    )


def block_transfer(
    block_graph: nx.Graph,
    root: Vertex,
    child_profiles: Mapping[Vertex, Profile],
) -> Profile:
    """Dispatch to the clique transfer when possible, otherwise use the
    exact effective four-letter arbitrary-block kernel."""
    m = block_graph.number_of_nodes()
    if m >= 2 and block_graph.number_of_edges() == m * (m - 1) // 2:
        return clique_transfer(block_graph, root, child_profiles)
    return effective_block_transfer(block_graph, root, child_profiles)


def connected_block_polynomial(graph: nx.Graph) -> Polynomial:
    if graph.number_of_nodes() == 0:
        return [1]
    if graph.number_of_nodes() == 1:
        return [0, 1]
    if not nx.is_connected(graph):
        raise ValueError("connected_block_polynomial requires a connected graph.")

    blocks = [frozenset(block) for block in nx.biconnected_components(graph)]
    articulation_vertices = set(nx.articulation_points(graph))

    block_cut_tree = nx.Graph()
    for index, block in enumerate(blocks):
        block_node = ("block", index)
        block_cut_tree.add_node(block_node)
        for vertex in block & articulation_vertices:
            block_cut_tree.add_edge(block_node, ("articulation", vertex))

    # Root the block-cut tree at a leaf block.  Such a block contains a
    # non-articulation vertex (unless it is the only block, in which case every
    # vertex is available).  Choosing a non-articulation root avoids an
    # additional descendant branch attached at the global root.
    block_degrees = {
        ("block", i): block_cut_tree.degree(("block", i))
        for i in range(len(blocks))
    }
    root_block_node = next(
        node for node, degree in block_degrees.items() if degree <= 1
    )
    root_block_index = int(root_block_node[1])
    root_vertex = next(
        vertex
        for vertex in blocks[root_block_index]
        if vertex not in articulation_vertices
    )

    parent: Dict[Tuple[str, object], Tuple[str, object] | None] = {
        root_block_node: None
    }
    order = [root_block_node]
    for node in order:
        for neighbor in block_cut_tree.neighbors(node):
            if neighbor == parent[node]:
                continue
            parent[neighbor] = node
            order.append(neighbor)

    profiles: Dict[Tuple[str, object], Profile] = {}

    for node in reversed(order):
        node_type, value = node

        if node_type == "articulation":
            response = SINGLETON_PROFILE
            for child in block_cut_tree.neighbors(node):
                if parent.get(child) == node:
                    response = wedge_product(response, profiles[child])
            profiles[node] = response
            continue

        block_index = int(value)
        block_vertices = blocks[block_index]
        parent_node = parent[node]
        root = root_vertex if parent_node is None else parent_node[1]

        child_profiles: Dict[Vertex, Profile] = {}
        for child in block_cut_tree.neighbors(node):
            if parent.get(child) == node:
                child_profiles[child[1]] = profiles[child]

        profiles[node] = block_transfer(
            graph.subgraph(block_vertices).copy(),
            root,
            child_profiles,
        )

    return full_polynomial(profiles[root_block_node])


def zero_forcing_polynomial_by_blocks(graph: nx.Graph) -> Polynomial:
    """Compute the exact polynomial, component by component."""
    if graph.number_of_nodes() == 0:
        return [1]

    result: Polynomial = [1]
    for component in nx.connected_components(graph):
        result = poly_mul(
            result,
            connected_block_polynomial(graph.subgraph(component).copy()),
        )
    return result


# ---------------------------------------------------------------------------
# Independent exhaustive verifier
# ---------------------------------------------------------------------------


def ordinary_closure(graph: nx.Graph, initial: Iterable[Vertex]) -> frozenset[Vertex]:
    blue = set(initial)
    changed = True
    while changed:
        changed = False
        for u in list(blue):
            white = [v for v in graph.neighbors(u) if v not in blue]
            if len(white) == 1:
                blue.add(white[0])
                changed = True
                break
    return frozenset(blue)


def brute_force_polynomial(graph: nx.Graph) -> Polynomial:
    vertices = list(graph.nodes())
    counts = [0] * (len(vertices) + 1)
    for mask in range(1 << len(vertices)):
        initial = {
            vertices[i]
            for i in range(len(vertices))
            if (mask >> i) & 1
        }
        if len(ordinary_closure(graph, initial)) == len(vertices):
            counts[len(initial)] += 1
    return trim(counts)


def make_cycle_chain(number_of_cycles: int, cycle_length: int) -> nx.Graph:
    if number_of_cycles < 1 or cycle_length < 3:
        raise ValueError("Invalid cycle-chain parameters.")

    graph = nx.Graph()
    shared = 0
    graph.add_node(shared)
    next_vertex = 1

    for _ in range(number_of_cycles):
        fresh = list(range(next_vertex, next_vertex + cycle_length - 1))
        next_vertex += cycle_length - 1
        cycle = [shared] + fresh
        nx.add_cycle(graph, cycle)
        shared = fresh[-1]

    return graph


def make_random_bounded_block_graph(
    rng: random.Random,
    number_of_blocks: int,
    maximum_block_order: int,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0)
    existing_vertices = [0]
    next_vertex = 1

    for _ in range(number_of_blocks):
        articulation = rng.choice(existing_vertices)
        order = rng.randint(2, maximum_block_order)
        fresh = list(range(next_vertex, next_vertex + order - 1))
        next_vertex += order - 1
        block_vertices = [articulation] + fresh

        # A random connected block: bridge when order=2; otherwise begin with a
        # cycle to guarantee 2-connectivity and add random chords.
        if order == 2:
            graph.add_edge(*block_vertices)
        else:
            nx.add_cycle(graph, block_vertices)
            for i in range(order):
                for j in range(i + 1, order):
                    u, v = block_vertices[i], block_vertices[j]
                    if not graph.has_edge(u, v) and rng.random() < 0.25:
                        graph.add_edge(u, v)

        existing_vertices.extend(fresh)

    return graph


def audit_atlas() -> None:
    checked = 0
    start = time.perf_counter()

    for graph in nx.graph_atlas_g():
        if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
            continue
        predicted = zero_forcing_polynomial_by_blocks(graph)
        observed = brute_force_polynomial(graph)
        if predicted != observed:
            raise AssertionError(
                f"Atlas failure: n={len(graph)}, edges={list(graph.edges())}, "
                f"predicted={predicted}, observed={observed}"
            )
        checked += 1

    elapsed = time.perf_counter() - start
    print(f"All {checked} connected graph-atlas graphs passed in {elapsed:.3f}s.")


def audit_random(number_of_cases: int, seed: int) -> None:
    rng = random.Random(seed)
    checked = 0
    start = time.perf_counter()

    while checked < number_of_cases:
        graph = make_random_bounded_block_graph(
            rng,
            number_of_blocks=rng.randint(1, 5),
            maximum_block_order=rng.randint(2, 5),
        )
        if graph.number_of_nodes() > 18:
            continue

        predicted = zero_forcing_polynomial_by_blocks(graph)
        observed = brute_force_polynomial(graph)
        if predicted != observed:
            raise AssertionError(
                f"Random failure: edges={list(graph.edges())}, "
                f"predicted={predicted}, observed={observed}"
            )
        checked += 1

    elapsed = time.perf_counter() - start
    print(
        f"All {checked} random bounded-block graphs passed "
        f"(seed={seed}) in {elapsed:.3f}s."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cycle-chain",
        nargs=2,
        type=int,
        metavar=("CYCLES", "LENGTH"),
    )
    group.add_argument("--audit-atlas", action="store_true")
    group.add_argument("--audit-random", type=int, metavar="CASES")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--verify-bruteforce", action="store_true")
    args = parser.parse_args()

    if args.audit_atlas:
        audit_atlas()
        return
    if args.audit_random is not None:
        audit_random(args.audit_random, args.seed)
        return

    if args.cycle_chain is None:
        graph = make_cycle_chain(5, 4)
        label = "chain of five 4-cycles"
    else:
        cycles, length = args.cycle_chain
        graph = make_cycle_chain(cycles, length)
        label = f"chain of {cycles} cycles of length {length}"

    start = time.perf_counter()
    polynomial = zero_forcing_polynomial_by_blocks(graph)
    elapsed = time.perf_counter() - start

    print(label)
    print("Vertices:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())
    print("Largest block order:", max(map(len, nx.biconnected_components(graph))))
    print("Coefficient list:", polynomial)
    print("Polynomial:", polynomial_to_string(polynomial))
    print("Zero-forcing number:", next(i for i, value in enumerate(polynomial) if value))
    print("Total zero-forcing sets:", sum(polynomial))
    print(f"Block dynamic-programming time: {elapsed:.6f}s")

    if args.verify_bruteforce:
        if graph.number_of_nodes() > 24:
            raise ValueError("Brute-force verification is disabled above 24 vertices.")
        observed = brute_force_polynomial(graph)
        print("Brute-force match:", observed == polynomial)


if __name__ == "__main__":
    main()
