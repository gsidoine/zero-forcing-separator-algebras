#!/usr/bin/env python3
"""Noninteractive audits for the rooted response algebra.

This script exposes the two audit rows that were previously available only in
notebook form:

* all 19,044 ordered wedges of rooted connected atlas graphs of order at most 5;
* 350 seeded random rooted wedges.

It also provides an independent direct rooted-profile enumerator used to check
the polynomial wedge product coefficient by coefficient.
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from typing import Hashable, Iterable

import networkx as nx

from zero_forcing_tree_dp import Profile, wedge_product

Vertex = Hashable


def trim(polynomial: list[int]) -> list[int]:
    output = list(polynomial)
    while len(output) > 1 and output[-1] == 0:
        output.pop()
    return output


def restricted_closure(
    graph: nx.Graph,
    initial: Iterable[Vertex],
    forbidden_root: Vertex | None,
) -> frozenset[Vertex]:
    blue = set(initial)
    changed = True
    while changed:
        changed = False
        for vertex in list(blue):
            if vertex == forbidden_root:
                continue
            white = [neighbour for neighbour in graph.neighbors(vertex) if neighbour not in blue]
            if len(white) == 1:
                blue.add(white[0])
                changed = True
                break
    return frozenset(blue)


def response_category(
    graph: nx.Graph,
    root: Vertex,
    initial_without_root: set[Vertex],
) -> tuple[str, int]:
    minus = restricted_closure(graph, initial_without_root, root)
    source = int(root in minus)

    plus = restricted_closure(graph, initial_without_root | {root}, root)
    if len(plus) == graph.number_of_nodes():
        return "P", source

    white_root_neighbours = [
        neighbour for neighbour in graph.neighbors(root) if neighbour not in plus
    ]
    if len(white_root_neighbours) == 1:
        completed = restricted_closure(
            graph,
            set(plus) | {white_root_neighbours[0]},
            root,
        )
        if len(completed) == graph.number_of_nodes():
            return "A", source
    return "F", source


def direct_profile(graph: nx.Graph, root: Vertex) -> Profile:
    """Enumerate a rooted response profile using an independent bitset closure."""
    ordered = [root] + [vertex for vertex in graph if vertex != root]
    index = {vertex: position for position, vertex in enumerate(ordered)}
    adjacency = [0] * len(ordered)
    for vertex in ordered:
        mask = 0
        for neighbour in graph.neighbors(vertex):
            mask |= 1 << index[neighbour]
        adjacency[index[vertex]] = mask

    full = (1 << len(ordered)) - 1
    root_bit = 1
    nonroot_order = len(ordered) - 1
    counts = {
        ("P", 0): [0] * (nonroot_order + 1),
        ("P", 1): [0] * (nonroot_order + 1),
        ("A", 0): [0] * (nonroot_order + 1),
        ("A", 1): [0] * (nonroot_order + 1),
    }

    def closure(blue: int, forbid_root: bool) -> int:
        while True:
            new_vertex = 0
            candidates = blue & (~root_bit if forbid_root else full)
            while candidates:
                least = candidates & -candidates
                vertex = least.bit_length() - 1
                candidates -= least
                white = adjacency[vertex] & ~blue
                if white and (white & (white - 1)) == 0:
                    new_vertex = white
                    break
            if new_vertex == 0:
                return blue
            blue |= new_vertex

    for compact_mask in range(1 << nonroot_order):
        initial = compact_mask << 1
        minus = closure(initial, True)
        source = int(bool(minus & root_bit))
        plus = closure(initial | root_bit, True)
        degree = compact_mask.bit_count()
        if plus == full:
            counts[("P", source)][degree] += 1
            continue
        white_root_neighbours = adjacency[0] & ~plus
        if (
            white_root_neighbours
            and (white_root_neighbours & (white_root_neighbours - 1)) == 0
            and closure(plus | white_root_neighbours, True) == full
        ):
            counts[("A", source)][degree] += 1

    return Profile(
        P0=trim(counts[("P", 0)]),
        P1=trim(counts[("P", 1)]),
        A0=trim(counts[("A", 0)]),
        A1=trim(counts[("A", 1)]),
    )


def canonical_rooted_copy(
    graph: nx.Graph,
    root: Vertex,
    side: str,
) -> nx.Graph:
    mapping = {root: ("root", 0)}
    next_index = 0
    for vertex in graph:
        if vertex == root:
            continue
        mapping[vertex] = (side, next_index)
        next_index += 1
    return nx.relabel_nodes(graph, mapping, copy=True)


def rooted_wedge(
    first: nx.Graph,
    first_root: Vertex,
    second: nx.Graph,
    second_root: Vertex,
) -> tuple[nx.Graph, tuple[str, int]]:
    left = canonical_rooted_copy(first, first_root, "L")
    right = canonical_rooted_copy(second, second_root, "R")
    root = ("root", 0)
    return nx.compose(left, right), root


def rooted_atlas_graphs(maximum_order: int = 5) -> list[tuple[nx.Graph, Vertex, Profile]]:
    rooted: list[tuple[nx.Graph, Vertex, Profile]] = []
    for graph in nx.graph_atlas_g():
        if (
            graph.number_of_nodes() == 0
            or graph.number_of_nodes() > maximum_order
            or not nx.is_connected(graph)
        ):
            continue
        for root in graph:
            rooted.append((graph.copy(), root, direct_profile(graph, root)))
    return rooted


def audit_ordered_wedges() -> None:
    rooted = rooted_atlas_graphs(5)
    if len(rooted) != 138:
        raise AssertionError(f"Expected 138 rooted atlas graphs, obtained {len(rooted)}.")

    checked = 0
    start = time.perf_counter()
    for first_graph, first_root, first_profile in rooted:
        for second_graph, second_root, second_profile in rooted:
            wedge, root = rooted_wedge(
                first_graph,
                first_root,
                second_graph,
                second_root,
            )
            observed = direct_profile(wedge, root)
            predicted = wedge_product(first_profile, second_profile)
            if observed != predicted:
                raise AssertionError(
                    "Ordered-wedge mismatch: "
                    f"left_edges={list(first_graph.edges())}, left_root={first_root}, "
                    f"right_edges={list(second_graph.edges())}, right_root={second_root}, "
                    f"observed={observed}, predicted={predicted}"
                )
            checked += 1

    if checked != 19_044:
        raise AssertionError(f"Expected 19,044 ordered wedges, obtained {checked}.")
    elapsed = time.perf_counter() - start
    print(f"Ordered rooted wedges passed: {checked} (138 x 138) in {elapsed:.3f}s.")


def random_connected_graph(order: int, rng: random.Random) -> nx.Graph:
    if order == 1:
        return nx.empty_graph(1)
    graph = nx.random_labeled_tree(order, seed=rng.randrange(1 << 63))
    for first in range(order):
        for second in range(first + 1, order):
            if not graph.has_edge(first, second) and rng.random() < 0.25:
                graph.add_edge(first, second)
    return graph


def audit_random_wedges(number_of_cases: int, seed: int) -> None:
    rng = random.Random(seed)
    checked = 0
    largest_order = 0
    start = time.perf_counter()

    while checked < number_of_cases:
        first_order = rng.randint(1, 8)
        second_order = rng.randint(1, 8)
        total_order = first_order + second_order - 1
        if total_order > 16:
            continue

        first = random_connected_graph(first_order, rng)
        second = random_connected_graph(second_order, rng)
        first_root = rng.choice(list(first))
        second_root = rng.choice(list(second))
        first_profile = direct_profile(first, first_root)
        second_profile = direct_profile(second, second_root)

        wedge, root = rooted_wedge(first, first_root, second, second_root)
        observed = direct_profile(wedge, root)
        predicted = wedge_product(first_profile, second_profile)
        if observed != predicted:
            raise AssertionError(
                "Random-wedge mismatch: "
                f"first_edges={list(first.edges())}, first_root={first_root}, "
                f"second_edges={list(second.edges())}, second_root={second_root}, "
                f"observed={observed}, predicted={predicted}"
            )
        checked += 1
        largest_order = max(largest_order, total_order)

    elapsed = time.perf_counter() - start
    print(
        f"Random rooted wedges passed: {checked} "
        f"(seed={seed}, maximum order={largest_order}) in {elapsed:.3f}s."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit-wedges", action="store_true")
    group.add_argument("--audit-random-wedges", type=int, metavar="CASES")
    group.add_argument("--audit-all", action="store_true")
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    if args.audit_wedges:
        audit_ordered_wedges()
    elif args.audit_random_wedges is not None:
        audit_random_wedges(args.audit_random_wedges, args.seed)
    else:
        audit_ordered_wedges()
        audit_random_wedges(350, args.seed)


if __name__ == "__main__":
    main()
