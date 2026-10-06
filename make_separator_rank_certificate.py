#!/usr/bin/env python3
"""Regenerate separator_rank_certificate.json deterministically."""
from __future__ import annotations

import json
from pathlib import Path

from verify_separator_rank_certificate import (
    independent_indices_mod_p,
    det_mod_p_from_minor,
    matrix_hash,
    reconstruct,
)
from zero_forcing_separator_algebra import evaluate_coefficient_matrices, rank_mod_p

PRIME = 1000003
X0 = 2
OUT = Path("separator_rank_certificate.json")


def sector_entry(mats):
    joined = mats[0].row_join(mats[1]).row_join(mats[2])
    exact_rank = int(joined.rank())
    Mx = evaluate_coefficient_matrices(mats, X0)
    mod_rank = rank_mod_p(Mx, PRIME)
    rows = independent_indices_mod_p(Mx, PRIME, transpose=True)
    cols = independent_indices_mod_p(Mx, PRIME, transpose=False)
    if len(rows) != mod_rank or len(cols) != mod_rank:
        raise AssertionError("failed to construct full-rank minor indices")
    det = det_mod_p_from_minor(Mx, rows, cols, PRIME)
    if det == 0:
        raise AssertionError("constructed minor is singular")
    return {
        "coefficient_span_rank_Q": exact_rank,
        "coefficient_matrices_sha256": matrix_hash(mats),
        "rank_mod_p_at_specialization": mod_rank,
        "minor_rows": rows,
        "minor_cols": cols,
        "minor_det_mod_p": det,
    }


def main():
    states, independent, edge, mobius = reconstruct()
    data = {
        "schema_version": 1,
        "description": "Exact finite certificate for the q=2 separator-rank calculation in Separator algebras and transfer theory for the zero-forcing polynomial.",
        "prime": PRIME,
        "specialization_x": X0,
        "chain_count_q2": len(states),
        "chain_states_q2": [sorted([list(atom) for atom in state]) for state in states],
        "sectors": {
            "independent": sector_entry(independent),
            "edge_present": sector_entry(edge),
            "edge_mobius": sector_entry(mobius),
        },
    }
    data["full_rank_q2"] = data["sectors"]["edge_present"]["coefficient_span_rank_Q"] + data["sectors"]["edge_mobius"]["coefficient_span_rank_Q"]
    OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
