#!/usr/bin/env python3
"""
Explicit cycle and clique transfers for the zero-forcing polynomial.

Main theorem implemented
------------------------
The complete zero-forcing polynomial of an n-vertex graph whose blocks are
cliques or cycles is computed exactly in O(n^2) coefficient-arithmetic
operations, using ordinary polynomial convolution. Cactus graphs are the
cycle/bridge special case.

Requirements
------------
    pip install networkx

Examples
--------
    python zero_forcing_cactus_dp.py --cycle-chain 20 17
    python zero_forcing_cactus_dp.py --audit-automaton
    python zero_forcing_cactus_dp.py --audit-atlas
    python zero_forcing_cactus_dp.py --audit-random 300 --seed 20260726
"""

from __future__ import annotations

import argparse
import collections
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


def poly_shift(p: Sequence[int], amount: int = 1) -> Polynomial:
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


SINGLETON_PROFILE = Profile.make()


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


def edge_extend(old: Profile) -> Profile:
    """Adjoin a new pendant root adjacent to the old root."""
    return Profile.make(
        P0=poly_add(old.A1, poly_shift(old.A)),
        P1=poly_add(old.P1, poly_shift(old.P)),
        A0=poly_add(old.P0, old.A0),
        A1=[0],
    )


def full_polynomial(profile: Profile) -> Polynomial:
    return poly_sub(
        poly_mul([1, 1], poly_add(profile.P, profile.A)),
        poly_add(profile.P0, profile.A0),
    )


# ---------------------------------------------------------------------------
# The explicit 32-state cycle automaton
# ---------------------------------------------------------------------------

# Letter order: (effective initial-blue bit, pending-demand bit).
LETTERS: Tuple[Tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1))

# State 0 is the initial state.  Rows are ordered by LETTERS.
CYCLE_TRANSITIONS: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 2, 3, 4),
    (1, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 7),
    (15, 16, 15, 4),
    (1, 9, 17, 11),
    (14, 18, 14, 7),
    (14, 19, 14, 7),
    (8, 2, 20, 4),
    (9, 9, 11, 11),
    (21, 2, 15, 4),
    (22, 9, 22, 11),
    (12, 23, 24, 24),
    (25, 16, 3, 4),
    (14, 7, 24, 24),
    (15, 4, 24, 24),
    (16, 16, 4, 4),
    (26, 5, 14, 7),
    (27, 19, 6, 7),
    (19, 19, 7, 7),
    (15, 28, 15, 4),
    (21, 29, 24, 24),
    (22, 11, 24, 24),
    (12, 16, 12, 4),
    (24, 24, 24, 24),
    (25, 13, 6, 7),
    (26, 30, 24, 24),
    (27, 18, 6, 7),
    (31, 16, 20, 4),
    (21, 9, 21, 11),
    (26, 9, 26, 11),
    (31, 28, 20, 4),
)

CYCLE_OUTPUTS: Tuple[str, ...] = (
    "P0", "F", "P0", "F", "P0", "P0", "F", "A0",
    "P0", "P1", "P0", "P1", "F", "P0", "F", "A0",
    "P1", "F", "A0", "A1", "P0", "A0", "A1", "P0",
    "F", "F", "F", "F", "P0", "P0", "P0", "P0",
)


def local_letter_weights(profile: Profile) -> Dict[Tuple[int, int], Polynomial]:
    """
    Aggregate the choice of the block vertex itself and the four descendant
    response states into the four local cycle letters.
    """
    return {
        (0, 0): list(profile.P0),
        (1, 0): poly_add(
            poly_shift(profile.P0),
            poly_mul([1, 1], profile.P1),
        ),
        (0, 1): list(profile.A0),
        (1, 1): poly_add(
            poly_shift(profile.A0),
            poly_mul([1, 1], profile.A1),
        ),
    }


def cycle_transfer(
    cyclic_order: Sequence[Vertex],
    root: Vertex,
    child_profiles: Mapping[Vertex, Profile],
) -> Profile:
    """
    Exact rooted profile of a cycle block with arbitrary descendant profiles.

    ``cyclic_order`` must begin with ``root`` and list every cycle vertex once.
    """
    if len(cyclic_order) < 3 or cyclic_order[0] != root:
        raise ValueError("A cycle order of length at least three must begin at root.")
    if len(set(cyclic_order)) != len(cyclic_order):
        raise ValueError("The cycle order contains repeated vertices.")
    if root in child_profiles:
        raise ValueError("The parent attachment cannot be a child articulation.")

    dp: List[Polynomial] = [[0] for _ in range(32)]
    dp[0] = [1]

    for vertex in cyclic_order[1:]:
        weights = local_letter_weights(
            child_profiles.get(vertex, SINGLETON_PROFILE)
        )
        next_dp: List[Polynomial] = [[0] for _ in range(32)]

        for state, polynomial in enumerate(dp):
            if polynomial == [0]:
                continue
            for letter_index, letter in enumerate(LETTERS):
                weight = weights[letter]
                if weight == [0]:
                    continue
                target = CYCLE_TRANSITIONS[state][letter_index]
                next_dp[target] = poly_add(
                    next_dp[target],
                    poly_mul(polynomial, weight),
                )
        dp = next_dp

    output: Dict[str, Polynomial] = {
        "P0": [0], "P1": [0], "A0": [0], "A1": [0]
    }
    for state, polynomial in enumerate(dp):
        category = CYCLE_OUTPUTS[state]
        if category != "F":
            output[category] = poly_add(output[category], polynomial)

    return Profile.make(**output)


def clique_transfer(
    block: nx.Graph,
    root: Vertex,
    child_profiles: Mapping[Vertex, Profile],
) -> Profile:
    """Exact linear-scan transfer for a complete block of arbitrary order."""
    m = block.number_of_nodes()
    if root not in block or m < 2 or block.number_of_edges() != m * (m - 1) // 2:
        raise ValueError("clique_transfer requires a complete block of order at least two.")
    if any(v not in block or v == root for v in child_profiles):
        raise ValueError("Child profiles must be rooted at non-root block vertices.")

    B: Polynomial = [1]
    D: Polynomial = [1]
    W: Polynomial = [0]
    E: Polynomial = [0]
    for vertex in block:
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


# ---------------------------------------------------------------------------
# Clique/cycle block-cut dynamic programming
# ---------------------------------------------------------------------------


def cycle_order(block: nx.Graph, root: Vertex) -> List[Vertex]:
    if root not in block:
        raise ValueError("The root is not in the block.")
    if block.number_of_nodes() < 3:
        raise ValueError("A cycle block has at least three vertices.")
    if block.number_of_edges() != block.number_of_nodes():
        raise ValueError("The block is not a cycle.")
    if any(block.degree(v) != 2 for v in block):
        raise ValueError("The block is not a cycle.")

    neighbours = list(block.neighbors(root))
    order = [root, neighbours[0]]
    previous, current = root, neighbours[0]

    while True:
        candidates = [v for v in block.neighbors(current) if v != previous]
        if len(candidates) != 1:
            raise ValueError("The block is not a simple cycle.")
        following = candidates[0]
        if following == root:
            break
        order.append(following)
        previous, current = current, following

    if len(order) != block.number_of_nodes():
        raise ValueError("The block is not a simple cycle.")
    return order


def _is_cycle_block(block: nx.Graph) -> bool:
    return (
        block.number_of_nodes() >= 3
        and block.number_of_edges() == block.number_of_nodes()
        and all(block.degree(v) == 2 for v in block)
    )


def _is_clique_block(block: nx.Graph) -> bool:
    m = block.number_of_nodes()
    return m >= 2 and block.number_of_edges() == m * (m - 1) // 2


def _validate_cactus_blocks(graph: nx.Graph) -> List[frozenset[Vertex]]:
    blocks = [frozenset(block) for block in nx.biconnected_components(graph)]
    for block_vertices in blocks:
        block = graph.subgraph(block_vertices)
        if len(block_vertices) == 2:
            if block.number_of_edges() != 1:
                raise ValueError("A two-vertex block must be a bridge.")
            continue
        if not _is_cycle_block(block):
            raise ValueError("The input graph is not a cactus.")
    return blocks


def _validate_clique_cycle_blocks(graph: nx.Graph) -> List[frozenset[Vertex]]:
    blocks = [frozenset(block) for block in nx.biconnected_components(graph)]
    for block_vertices in blocks:
        block = graph.subgraph(block_vertices)
        if not (_is_clique_block(block) or _is_cycle_block(block)):
            raise ValueError("Every nontrivial block must be a clique or a simple cycle.")
    return blocks


def _connected_from_blocks(graph: nx.Graph, blocks: List[frozenset[Vertex]]) -> Polynomial:
    if graph.number_of_nodes() == 0:
        return [1]
    if graph.number_of_nodes() == 1:
        return [0, 1]
    if not nx.is_connected(graph):
        raise ValueError("The graph must be connected.")

    articulation_vertices = set(nx.articulation_points(graph))
    block_cut_tree = nx.Graph()
    for index, block in enumerate(blocks):
        block_node = ("block", index)
        block_cut_tree.add_node(block_node)
        for vertex in block & articulation_vertices:
            block_cut_tree.add_edge(block_node, ("articulation", vertex))

    root_block_node = next(
        ("block", index)
        for index in range(len(blocks))
        if block_cut_tree.degree(("block", index)) <= 1
    )
    root_block_index = int(root_block_node[1])
    root_vertex = next(
        vertex for vertex in blocks[root_block_index]
        if vertex not in articulation_vertices
    )

    parent: Dict[Tuple[str, object], Tuple[str, object] | None] = {root_block_node: None}
    order = [root_block_node]
    for node in order:
        for neighbour in block_cut_tree.neighbors(node):
            if neighbour == parent[node]:
                continue
            parent[neighbour] = node
            order.append(neighbour)

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

        block = graph.subgraph(block_vertices).copy()
        if len(block_vertices) == 2:
            other = next(v for v in block_vertices if v != root)
            profiles[node] = edge_extend(child_profiles.get(other, SINGLETON_PROFILE))
        elif _is_clique_block(block):
            profiles[node] = clique_transfer(block, root, child_profiles)
        else:
            profiles[node] = cycle_transfer(cycle_order(block, root), root, child_profiles)

    return full_polynomial(profiles[root_block_node])


def connected_clique_cycle_polynomial(graph: nx.Graph) -> Polynomial:
    if graph.number_of_nodes() <= 1:
        return [1] if graph.number_of_nodes() == 0 else [0, 1]
    if not nx.is_connected(graph):
        raise ValueError("The graph must be connected.")
    return _connected_from_blocks(graph, _validate_clique_cycle_blocks(graph))


def zero_forcing_polynomial_clique_cycle(graph: nx.Graph) -> Polynomial:
    if graph.number_of_nodes() == 0:
        return [1]
    result: Polynomial = [1]
    for component in nx.connected_components(graph):
        result = poly_mul(result, connected_clique_cycle_polynomial(graph.subgraph(component).copy()))
    return result


def connected_cactus_polynomial(graph: nx.Graph) -> Polynomial:
    if graph.number_of_nodes() <= 1:
        return [1] if graph.number_of_nodes() == 0 else [0, 1]
    if not nx.is_connected(graph):
        raise ValueError("The graph must be connected.")
    return _connected_from_blocks(graph, _validate_cactus_blocks(graph))


def zero_forcing_polynomial_cactus(graph: nx.Graph) -> Polynomial:
    if graph.number_of_nodes() == 0:
        return [1]
    result: Polynomial = [1]
    for component in nx.connected_components(graph):
        result = poly_mul(result, connected_cactus_polynomial(graph.subgraph(component).copy()))
    return result


# ---------------------------------------------------------------------------
# Independent automaton certificate
# ---------------------------------------------------------------------------

FortState = Tuple[int, int, int]
BoolMatrix = Tuple[Tuple[bool, ...], ...]
FORT_STATES: Tuple[FortState, ...] = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 1),
)
FORT_INDEX = {state: index for index, state in enumerate(FORT_STATES)}


def _letter_matrix(blue: int, demand: int) -> BoolMatrix:
    matrix = [[False] * 5 for _ in range(5)]
    for row, (previous, current, seen) in enumerate(FORT_STATES):
        if blue and current:
            continue
        for following in (0, 1):
            if current == 0 and demand == 0 and previous != following:
                continue
            target = (
                current,
                following,
                int(bool(seen or following)),
            )
            matrix[row][FORT_INDEX[target]] = True
    return tuple(tuple(row) for row in matrix)


def _boolean_product(left: BoolMatrix, right: BoolMatrix) -> BoolMatrix:
    return tuple(
        tuple(
            any(left[i][k] and right[k][j] for k in range(5))
            for j in range(5)
        )
        for i in range(5)
    )


def _response_output(matrix: BoolMatrix) -> str:
    q0 = FORT_INDEX[(0, 0, 0)]
    q1 = FORT_INDEX[(0, 0, 1)]
    q2 = FORT_INDEX[(0, 1, 1)]
    q3 = FORT_INDEX[(1, 0, 1)]
    q4 = FORT_INDEX[(1, 1, 1)]

    restricted_fort = any(
        matrix[a][b]
        for a in (q0, q2)
        for b in (q1, q3)
    )
    ordinary_fort = matrix[q0][q1] or matrix[q2][q3]
    source_blocking_fort = any(
        matrix[a][b]
        for a in (q3, q4)
        for b in (q2, q4)
    )

    if ordinary_fort:
        return "F"
    source = int(not source_blocking_fort)
    if not restricted_fort:
        return f"P{source}"
    return f"A{source}"


def build_cycle_automaton() -> Tuple[
    Tuple[Tuple[int, int, int, int], ...], Tuple[str, ...], int
]:
    """
    Reconstruct the response-equivalence quotient from the five-state fort
    frontier.  Returns (transition table, outputs, transition-monoid size).
    """
    generators = {
        letter: _letter_matrix(*letter)
        for letter in LETTERS
    }
    identity: BoolMatrix = tuple(
        tuple(i == j for j in range(5))
        for i in range(5)
    )

    monoid = [identity]
    monoid_index = {identity: 0}
    queue = collections.deque([identity])
    while queue:
        matrix = queue.popleft()
        for letter in LETTERS:
            product = _boolean_product(matrix, generators[letter])
            if product not in monoid_index:
                monoid_index[product] = len(monoid)
                monoid.append(product)
                queue.append(product)

    transitions = [
        [
            monoid_index[_boolean_product(matrix, generators[letter])]
            for letter in LETTERS
        ]
        for matrix in monoid
    ]
    outputs = [_response_output(matrix) for matrix in monoid]

    # Moore partition refinement, retaining only future response behaviour.
    groups: Dict[str, List[int]] = {}
    for state, output in enumerate(outputs):
        groups.setdefault(output, []).append(state)
    blocks = list(groups.values())

    while True:
        block_of = {
            state: block_index
            for block_index, block in enumerate(blocks)
            for state in block
        }
        refined: List[List[int]] = []
        changed = False
        for block in blocks:
            cells: Dict[Tuple[int, ...], List[int]] = {}
            for state in block:
                signature = tuple(
                    block_of[transitions[state][letter_index]]
                    for letter_index in range(4)
                )
                cells.setdefault(signature, []).append(state)
            refined.extend(cells.values())
            changed = changed or len(cells) > 1
        blocks = refined
        if not changed:
            break

    block_of = {
        state: block_index
        for block_index, block in enumerate(blocks)
        for state in block
    }
    initial_block = block_of[0]

    quotient_transition: Dict[int, Tuple[int, ...]] = {}
    quotient_output: Dict[int, str] = {}
    for block_index, block in enumerate(blocks):
        representative = block[0]
        quotient_transition[block_index] = tuple(
            block_of[transitions[representative][letter_index]]
            for letter_index in range(4)
        )
        quotient_output[block_index] = outputs[representative]

    # Canonical breadth-first numbering from the initial state.
    canonical = {initial_block: 0}
    bfs = collections.deque([initial_block])
    order: List[int] = []
    while bfs:
        state = bfs.popleft()
        order.append(state)
        for target in quotient_transition[state]:
            if target not in canonical:
                canonical[target] = len(canonical)
                bfs.append(target)

    quotient = tuple(
        tuple(canonical[target] for target in quotient_transition[state])
        for state in order
    )
    quotient_outputs = tuple(quotient_output[state] for state in order)
    return quotient, quotient_outputs, len(monoid)


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def _direct_marker_response(labels: Sequence[Tuple[int, int]]) -> str:
    """Independent bitset simulation of one concrete marked cycle."""
    cycle_order_size = len(labels) + 1
    number_of_demands = sum(demand for _, demand in labels)
    adjacency = [0] * (cycle_order_size + number_of_demands)

    for vertex in range(cycle_order_size):
        neighbour = (vertex + 1) % cycle_order_size
        adjacency[vertex] |= 1 << neighbour
        adjacency[neighbour] |= 1 << vertex

    initial = 0
    next_leaf = cycle_order_size
    for vertex, (blue, demand) in enumerate(labels, start=1):
        if blue:
            initial |= 1 << vertex
        if demand:
            adjacency[vertex] |= 1 << next_leaf
            adjacency[next_leaf] |= 1 << vertex
            next_leaf += 1

    full = (1 << len(adjacency)) - 1

    def closure(blue: int, forbid_root: bool) -> int:
        while True:
            new_vertex = 0
            candidates = blue
            while candidates:
                least = candidates & -candidates
                vertex = least.bit_length() - 1
                candidates -= least
                if forbid_root and vertex == 0:
                    continue
                white = adjacency[vertex] & ~blue
                if white and (white & (white - 1)) == 0:
                    new_vertex = white
                    break
            if new_vertex == 0:
                return blue
            blue |= new_vertex

    minus = closure(initial, True)
    source = int(bool(minus & 1))
    plus = closure(initial | 1, True)
    if plus == full:
        return f"P{source}"

    white_root_neighbours = adjacency[0] & ~plus
    if (
        white_root_neighbours
        and (white_root_neighbours & (white_root_neighbours - 1)) == 0
        and closure(plus | white_root_neighbours, True) == full
    ):
        return f"A{source}"
    return "F"


def audit_automaton(max_cycle_length: int = 9) -> None:
    generated_transitions, generated_outputs, monoid_size = build_cycle_automaton()
    if generated_transitions != CYCLE_TRANSITIONS:
        raise AssertionError("The embedded transition table is incorrect.")
    if generated_outputs != CYCLE_OUTPUTS:
        raise AssertionError("The embedded output table is incorrect.")
    if monoid_size != 100:
        raise AssertionError(f"Expected a 100-element monoid, obtained {monoid_size}.")

    checked = 0
    for cycle_length in range(3, max_cycle_length + 1):
        for labels in itertools.product(LETTERS, repeat=cycle_length - 1):
            state = 0
            for letter in labels:
                state = CYCLE_TRANSITIONS[state][LETTERS.index(letter)]
            observed = _direct_marker_response(labels)
            predicted = CYCLE_OUTPUTS[state]
            if observed != predicted:
                raise AssertionError(
                    f"Automaton mismatch for C_{cycle_length}, {labels}: "
                    f"direct={observed}, automaton={predicted}"
                )
            checked += 1

    print("Transition monoid size: 100")
    print("Response quotient size: 32")
    print(f"Concrete marked cycles checked: {checked}")


def brute_force_polynomial(graph: nx.Graph) -> Polynomial:
    vertices = list(graph)
    n = len(vertices)
    index = {vertex: i for i, vertex in enumerate(vertices)}
    adjacency = [0] * n
    for vertex in vertices:
        mask = 0
        for neighbour in graph.neighbors(vertex):
            mask |= 1 << index[neighbour]
        adjacency[index[vertex]] = mask

    full = (1 << n) - 1
    counts = [0] * (n + 1)
    for initial in range(1 << n):
        blue = initial
        while True:
            new_vertex = 0
            candidates = blue
            while candidates:
                least = candidates & -candidates
                vertex = least.bit_length() - 1
                candidates -= least
                white = adjacency[vertex] & ~blue
                if white and (white & (white - 1)) == 0:
                    new_vertex = white
                    break
            if new_vertex == 0:
                break
            blue |= new_vertex
        if blue == full:
            counts[initial.bit_count()] += 1
    return trim(counts)


def is_cactus(graph: nx.Graph) -> bool:
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        for block_vertices in nx.biconnected_components(subgraph):
            block = subgraph.subgraph(block_vertices)
            if len(block_vertices) == 2:
                continue
            if (
                block.number_of_edges() != len(block_vertices)
                or any(block.degree(v) != 2 for v in block)
            ):
                return False
    return True


def is_clique_cycle_block_graph(graph: nx.Graph) -> bool:
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        for block_vertices in nx.biconnected_components(subgraph):
            block = subgraph.subgraph(block_vertices)
            if not (_is_clique_block(block) or _is_cycle_block(block)):
                return False
    return True


def audit_mixed_atlas() -> None:
    checked = 0
    for graph in nx.graph_atlas_g():
        if graph.number_of_nodes() == 0 or not is_clique_cycle_block_graph(graph):
            continue
        predicted = zero_forcing_polynomial_clique_cycle(graph)
        observed = brute_force_polynomial(graph)
        if predicted != observed:
            raise AssertionError(
                f"Mixed atlas mismatch: edges={list(graph.edges())}, predicted={predicted}, observed={observed}"
            )
        checked += 1
    print(f"All {checked} clique/cycle-block graph-atlas graphs passed.")


def random_clique_cycle_graph(number_of_blocks: int, maximum_order: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    graph = nx.Graph()
    graph.add_node(0)
    next_vertex = 1
    for _ in range(number_of_blocks):
        articulation = rng.choice(list(graph.nodes()))
        mode = rng.choice(("bridge", "cycle", "clique"))
        if mode == "bridge":
            graph.add_edge(articulation, next_vertex)
            next_vertex += 1
            continue
        order = rng.randint(3, maximum_order)
        vertices = [articulation] + list(range(next_vertex, next_vertex + order - 1))
        next_vertex += order - 1
        if mode == "cycle":
            nx.add_cycle(graph, vertices)
        else:
            for i, u in enumerate(vertices):
                for v in vertices[i + 1:]:
                    graph.add_edge(u, v)
    return graph


def audit_mixed_random(number_of_cases: int, seed: int) -> None:
    rng = random.Random(seed)
    checked = 0
    while checked < number_of_cases:
        graph = random_clique_cycle_graph(
            number_of_blocks=rng.randint(1, 6), maximum_order=6, seed=rng.randrange(1 << 63)
        )
        if graph.number_of_nodes() > 16:
            continue
        predicted = zero_forcing_polynomial_clique_cycle(graph)
        observed = brute_force_polynomial(graph)
        if predicted != observed:
            raise AssertionError(
                f"Mixed random mismatch: edges={list(graph.edges())}, predicted={predicted}, observed={observed}"
            )
        checked += 1
    print(f"All {checked} random clique/cycle-block graphs passed (seed={seed}).")


def audit_atlas() -> None:
    checked = 0
    for graph in nx.graph_atlas_g():
        if graph.number_of_nodes() == 0 or not is_cactus(graph):
            continue
        predicted = zero_forcing_polynomial_cactus(graph)
        observed = brute_force_polynomial(graph)
        if predicted != observed:
            raise AssertionError(
                f"Atlas mismatch: edges={list(graph.edges())}, "
                f"predicted={predicted}, observed={observed}"
            )
        checked += 1
    print(f"All {checked} cactus graph-atlas graphs passed.")


def random_cactus(number_of_blocks: int, maximum_cycle_length: int, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    graph = nx.Graph()
    graph.add_node(0)
    next_vertex = 1

    for _ in range(number_of_blocks):
        articulation = rng.choice(list(graph.nodes()))
        if rng.random() < 0.45:
            graph.add_edge(articulation, next_vertex)
            next_vertex += 1
        else:
            length = rng.randint(3, maximum_cycle_length)
            cycle = [articulation] + list(
                range(next_vertex, next_vertex + length - 1)
            )
            next_vertex += length - 1
            nx.add_cycle(graph, cycle)
    return graph


def audit_random(number_of_cases: int, seed: int) -> None:
    rng = random.Random(seed)
    checked = 0
    while checked < number_of_cases:
        graph = random_cactus(
            number_of_blocks=rng.randint(1, 6),
            maximum_cycle_length=7,
            seed=rng.randrange(1 << 63),
        )
        if graph.number_of_nodes() > 16:
            continue
        predicted = zero_forcing_polynomial_cactus(graph)
        observed = brute_force_polynomial(graph)
        if predicted != observed:
            raise AssertionError(
                f"Random mismatch: edges={list(graph.edges())}, "
                f"predicted={predicted}, observed={observed}"
            )
        checked += 1
    print(f"All {checked} random cactus graphs passed (seed={seed}).")


def make_cycle_chain(number_of_cycles: int, cycle_length: int) -> nx.Graph:
    if number_of_cycles < 1 or cycle_length < 3:
        raise ValueError("Use at least one cycle of length at least three.")
    graph = nx.Graph()
    articulation = 0
    graph.add_node(articulation)
    next_vertex = 1
    for _ in range(number_of_cycles):
        cycle = [articulation] + list(
            range(next_vertex, next_vertex + cycle_length - 1)
        )
        next_vertex += cycle_length - 1
        nx.add_cycle(graph, cycle)
        articulation = cycle[-1]
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-automaton", action="store_true")
    parser.add_argument("--audit-atlas", action="store_true")
    parser.add_argument("--audit-random", type=int, metavar="N")
    parser.add_argument("--audit-mixed-atlas", action="store_true")
    parser.add_argument("--audit-mixed-random", type=int, metavar="N")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--cycle-chain",
        nargs=2,
        type=int,
        metavar=("NUMBER", "LENGTH"),
    )
    args = parser.parse_args()

    if args.audit_automaton:
        audit_automaton()
        return
    if args.audit_atlas:
        audit_atlas()
        return
    if args.audit_random is not None:
        audit_random(args.audit_random, args.seed)
        return
    if args.audit_mixed_atlas:
        audit_mixed_atlas()
        return
    if args.audit_mixed_random is not None:
        audit_mixed_random(args.audit_mixed_random, args.seed)
        return

    if args.cycle_chain is None:
        graph = make_cycle_chain(5, 7)
        label = "chain of five 7-cycles"
    else:
        number, length = args.cycle_chain
        graph = make_cycle_chain(number, length)
        label = f"chain of {number} cycles of length {length}"

    start = time.perf_counter()
    polynomial = zero_forcing_polynomial_cactus(graph)
    elapsed = time.perf_counter() - start

    print(label)
    print("Vertices:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())
    print("Coefficient list:", polynomial)
    if len(polynomial) <= 60:
        print("Polynomial:", polynomial_to_string(polynomial))
    print("Zero-forcing number:", next(i for i, value in enumerate(polynomial) if value))
    print("Total zero-forcing sets:", sum(polynomial))
    print(f"Elapsed time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
