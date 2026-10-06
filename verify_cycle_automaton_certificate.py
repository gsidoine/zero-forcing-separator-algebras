#!/usr/bin/env python3
"""Independently verify the zero-forcing cycle-automaton certificate.

The verifier reconstructs the four Boolean generators, breadth-first generates
all reachable transition-monoid matrices from the identity, independently
computes their response outputs, reconstructs the canonical Moore quotient,
checks every certificate encoding and table, verifies reachability of all
quotient states, checks exact coverage of all unordered state pairs by the
minimality witnesses, and compares the certified quotient with the table used
by ``zero_forcing_cactus_dp.py``.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Iterable, Sequence

BoolMatrix = tuple[tuple[int, ...], ...]
FortState = tuple[int, int, int]


def matrix_product(left: BoolMatrix, right: BoolMatrix) -> BoolMatrix:
    return tuple(
        tuple(
            int(any(left[i][k] and right[k][j] for k in range(5)))
            for j in range(5)
        )
        for i in range(5)
    )


def encode_matrix(matrix: BoolMatrix) -> str:
    bits = "".join(str(value) for row in matrix for value in row)
    return f"{int(bits, 2):07x}"


def letter_matrix(
    letter: str,
    states: Sequence[FortState],
    state_index: dict[FortState, int],
) -> BoolMatrix:
    if len(letter) != 2 or any(bit not in "01" for bit in letter):
        raise AssertionError(f"Invalid letter {letter!r}.")
    blue, demand = map(int, letter)
    matrix = [[0] * 5 for _ in range(5)]
    for row, (previous, current, seen) in enumerate(states):
        if blue and current:
            continue
        for following in (0, 1):
            if current == 0 and demand == 0 and previous != following:
                continue
            target = (current, following, int(bool(seen or following)))
            matrix[row][state_index[target]] = 1
    return tuple(tuple(row) for row in matrix)


def response_output(
    matrix: BoolMatrix,
    state_index: dict[FortState, int],
) -> str:
    q0 = state_index[(0, 0, 0)]
    q1 = state_index[(0, 0, 1)]
    q2 = state_index[(0, 1, 1)]
    q3 = state_index[(1, 0, 1)]
    q4 = state_index[(1, 1, 1)]

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


def breadth_first_monoid(
    generators: Sequence[BoolMatrix],
) -> tuple[list[BoolMatrix], list[list[int]]]:
    identity: BoolMatrix = tuple(
        tuple(int(i == j) for j in range(5))
        for i in range(5)
    )
    matrices = [identity]
    index = {identity: 0}
    queue: collections.deque[BoolMatrix] = collections.deque([identity])

    while queue:
        matrix = queue.popleft()
        for generator in generators:
            product = matrix_product(matrix, generator)
            if product not in index:
                index[product] = len(matrices)
                matrices.append(product)
                queue.append(product)

    transitions = [
        [index[matrix_product(matrix, generator)] for generator in generators]
        for matrix in matrices
    ]
    return matrices, transitions


def canonical_moore_quotient(
    transitions: Sequence[Sequence[int]],
    outputs: Sequence[str],
) -> tuple[list[int], list[list[int]], list[str], list[list[int]]]:
    groups: dict[str, list[int]] = {}
    for state, output in enumerate(outputs):
        groups.setdefault(output, []).append(state)
    blocks = list(groups.values())

    while True:
        block_of = {
            state: block_index
            for block_index, block in enumerate(blocks)
            for state in block
        }
        refined: list[list[int]] = []
        changed = False
        for block in blocks:
            cells: dict[tuple[int, ...], list[int]] = {}
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
    raw_transitions: dict[int, list[int]] = {}
    raw_outputs: dict[int, str] = {}
    for block_index, block in enumerate(blocks):
        representative = block[0]
        raw_transitions[block_index] = [
            block_of[transitions[representative][letter_index]]
            for letter_index in range(4)
        ]
        raw_outputs[block_index] = outputs[representative]

    initial_block = block_of[0]
    canonical = {initial_block: 0}
    queue: collections.deque[int] = collections.deque([initial_block])
    order: list[int] = []
    while queue:
        state = queue.popleft()
        order.append(state)
        for target in raw_transitions[state]:
            if target not in canonical:
                canonical[target] = len(canonical)
                queue.append(target)

    if len(order) != len(blocks):
        raise AssertionError("The Moore quotient contains unreachable blocks.")

    quotient_transitions = [
        [canonical[target] for target in raw_transitions[state]]
        for state in order
    ]
    quotient_outputs = [raw_outputs[state] for state in order]
    quotient_map = [canonical[block_of[state]] for state in range(len(transitions))]
    quotient_blocks = [
        [state for state, quotient_state in enumerate(quotient_map)
         if quotient_state == canonical[raw_block]]
        for raw_block in order
    ]
    return quotient_map, quotient_transitions, quotient_outputs, quotient_blocks


def load_algorithm_module(path: Path):
    spec = importlib.util.spec_from_file_location("zero_forcing_cactus_dp", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_certificate(certificate_path: Path, algorithm_path: Path | None) -> None:
    data = json.loads(certificate_path.read_text(encoding="utf-8"))
    if data.get("format") != "zero-forcing-cycle-automaton-certificate-v1":
        raise AssertionError("Unexpected certificate format.")

    letters = data["letter_order"]
    if letters != ["00", "10", "01", "11"]:
        raise AssertionError(f"Unexpected letter order: {letters!r}")

    states = [tuple(state) for state in data["frontier_states"]]
    expected_states = [
        (0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)
    ]
    if states != expected_states:
        raise AssertionError("Unexpected frontier-state order.")
    state_index = {state: index for index, state in enumerate(states)}

    generators = [letter_matrix(letter, states, state_index) for letter in letters]
    certificate_generators = data["generator_matrices"]
    if len(certificate_generators) != 4:
        raise AssertionError("The certificate must contain four generators.")
    for index, (letter, generated, entry) in enumerate(
        zip(letters, generators, certificate_generators)
    ):
        rows = tuple(tuple(row) for row in entry["rows"])
        if entry.get("letter") != letter:
            raise AssertionError(f"Generator {index} has the wrong letter.")
        if rows != generated:
            raise AssertionError(f"Generator rows disagree for letter {letter}.")
        if entry.get("encoding") != encode_matrix(generated):
            raise AssertionError(f"Generator encoding disagrees for letter {letter}.")

    generated_matrices, generated_transitions = breadth_first_monoid(generators)
    listed_entries = data["monoid_matrices"]
    listed_matrices = [
        tuple(tuple(row) for row in entry["rows"])
        for entry in listed_entries
    ]
    if len(generated_matrices) != data.get("monoid_size"):
        raise AssertionError("Generated monoid size disagrees with certificate metadata.")
    if generated_matrices != listed_matrices:
        raise AssertionError(
            "The listed matrices are not exactly the canonical breadth-first "
            "generation from the identity."
        )

    generated_outputs = [
        response_output(matrix, state_index) for matrix in generated_matrices
    ]
    for index, (matrix, transitions, output, entry) in enumerate(
        zip(generated_matrices, generated_transitions, generated_outputs, listed_entries)
    ):
        if entry.get("index") != index:
            raise AssertionError(f"Incorrect matrix index at position {index}.")
        if entry.get("encoding") != encode_matrix(matrix):
            raise AssertionError(f"Incorrect matrix encoding at position {index}.")
        if entry.get("transitions") != transitions:
            raise AssertionError(f"Incorrect matrix transitions at position {index}.")
        if entry.get("output") != output:
            raise AssertionError(f"Incorrect matrix output at position {index}.")

    quotient_map, quotient_transitions, quotient_outputs, quotient_blocks = (
        canonical_moore_quotient(generated_transitions, generated_outputs)
    )
    quotient_size = data.get("quotient_size")
    if quotient_size != len(quotient_transitions) or quotient_size != 32:
        raise AssertionError("Unexpected quotient size.")
    if data.get("quotient_map") != quotient_map:
        raise AssertionError("The quotient map is incorrect.")
    if data.get("quotient_blocks") != quotient_blocks:
        raise AssertionError("The quotient blocks are incorrect.")
    if data.get("quotient_transitions") != quotient_transitions:
        raise AssertionError("The quotient transition table is incorrect.")
    if data.get("quotient_outputs") != quotient_outputs:
        raise AssertionError("The quotient output table is incorrect.")
    if set(quotient_map) != set(range(quotient_size)):
        raise AssertionError("The quotient map is not surjective.")

    reachable = {0}
    queue: collections.deque[int] = collections.deque([0])
    while queue:
        state = queue.popleft()
        for target in quotient_transitions[state]:
            if target not in reachable:
                reachable.add(target)
                queue.append(target)
    if reachable != set(range(quotient_size)):
        raise AssertionError("Not every quotient state is reachable from state 0.")

    witness_entries = data["distinguishing_words"]
    expected_pairs = set(itertools.combinations(range(quotient_size), 2))
    listed_pairs: list[tuple[int, int]] = []
    for item in witness_entries:
        pair = tuple(sorted(item["states"]))
        if len(pair) != 2 or pair[0] == pair[1]:
            raise AssertionError(f"Invalid state pair {item['states']!r}.")
        listed_pairs.append(pair)
        first, second = pair
        for letter in item["word"]:
            if letter not in letters:
                raise AssertionError(f"Invalid witness letter {letter!r}.")
            letter_index = letters.index(letter)
            first = quotient_transitions[first][letter_index]
            second = quotient_transitions[second][letter_index]
        if quotient_outputs[first] == quotient_outputs[second]:
            raise AssertionError(f"Witness does not distinguish pair {pair}.")

    if len(listed_pairs) != len(set(listed_pairs)):
        raise AssertionError("The distinguishing-pair list contains duplicates.")
    if set(listed_pairs) != expected_pairs:
        missing = sorted(expected_pairs - set(listed_pairs))
        extra = sorted(set(listed_pairs) - expected_pairs)
        raise AssertionError(
            f"Distinguishing-pair coverage is incomplete; missing={missing}, extra={extra}."
        )

    if algorithm_path is not None:
        module = load_algorithm_module(algorithm_path)
        embedded_transitions = [list(row) for row in module.CYCLE_TRANSITIONS]
        embedded_outputs = list(module.CYCLE_OUTPUTS)
        if embedded_transitions != quotient_transitions:
            raise AssertionError("The algorithm's embedded transition table differs.")
        if embedded_outputs != quotient_outputs:
            raise AssertionError("The algorithm's embedded output table differs.")

    print(f"certificate verified: {certificate_path.name}")
    print(f"generated monoid size: {len(generated_matrices)}")
    print(f"reachable quotient states: {len(reachable)}")
    print(f"distinguishing pairs: {len(listed_pairs)}")
    if algorithm_path is not None:
        print(f"embedded algorithm table verified: {algorithm_path.name}")
    print(f"file sha256: {hashlib.sha256(certificate_path.read_bytes()).hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "certificate",
        nargs="?",
        default="cycle_automaton_certificate.json",
        type=Path,
    )
    parser.add_argument(
        "--algorithm",
        type=Path,
        default=Path("zero_forcing_cactus_dp.py"),
        help="Cactus implementation whose embedded quotient table is checked.",
    )
    parser.add_argument(
        "--skip-algorithm-table",
        action="store_true",
        help="Verify only the certificate, without importing the cactus implementation.",
    )
    args = parser.parse_args()
    algorithm = None if args.skip_algorithm_table else args.algorithm
    verify_certificate(args.certificate, algorithm)


if __name__ == "__main__":
    main()
