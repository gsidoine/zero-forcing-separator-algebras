# Separator algebras and transfer theory for the zero-forcing polynomial

This repository contains the manuscript source, reference implementation, exact finite certificates, and independent verification scripts for

**Guillaume Sidoine, _Separator algebras and transfer theory for the zero-forcing polynomial_.**

The paper develops an exact finite interface for the zero-forcing polynomial across an arbitrary labelled separator. The main structural ingredients are a fort-based atomic boundary language, join closure, chain compression by Möbius cancellation, realization of every chain mode by finite simple graph gadgets, and an exact connection-rank formula. The first full connection ranks are

\[
r_0=1,\qquad r_1=4,\qquad r_2=60.
\]

The one-terminal specialization recovers the four-dimensional algebra

\[
\mathbb Q(x)[s,a]/(s^2-s,a^2),
\]

and gives explicit transfer algorithms for trees, bounded blocks, cliques, cycles, and block-cactus graphs.

## Repository contents

The main reproducibility files are:

- `zero_forcing_separator_algebras.tex` / `.pdf` — manuscript source and compiled paper.
- `zero_forcing_separator_algebra.py` — arbitrary-separator atoms, chains, kernels, and the exact `q=2` reconstruction.
- `zero_forcing_separator_audits.py` — separator realization and compatibility audits.
- `separator_rank_certificate.json` — finite certificate for the `q=2` rank computation.
- `verify_separator_rank_certificate.py` — independent reconstruction and verification of that certificate.
- `cycle_automaton_certificate.json` / `verify_cycle_automaton_certificate.py` — the complete `100`-matrix / `32`-state / `496`-witness cycle certificate.
- `zero_forcing_tree_dp.py`, `zero_forcing_block_dp.py`, `zero_forcing_cactus_dp.py` — explicit transfer implementations.
- `zero_forcing_rooted_audits.py`, `zero_forcing_extended_audits.py` — independent small-graph and replacement checks.
- `run_all_audits.py` — one-command audit runner.
- `REPRODUCIBILITY.md` and `BUILD.md` — detailed instructions.

No external datasets are required.

## Quick start

Python 3.11 or later is required. The reference environment used Python 3.13.5, NetworkX 3.6.1, and SymPy 1.14.0.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_all_audits.py
```

To verify only the new separator theory and its `q=2` certificate:

```bash
python zero_forcing_separator_audits.py
python verify_separator_rank_certificate.py separator_rank_certificate.json
```

To verify the cycle automaton certificate:

```bash
python verify_cycle_automaton_certificate.py cycle_automaton_certificate.json
```

Expected outputs are recorded in `EXPECTED_AUDIT_OUTPUTS.txt` and the last complete validation is recorded in `FINAL_VALIDATION_LOG.txt`.

## What is proved symbolically and what is certified computationally?

The arbitrary-separator splitting theorem, chain compression, chain realization, selection projectors, and equality between the infinite connection rank and the finite chain-edge kernel are proved in the manuscript.

Finite assertions are separately certified. In particular:

- the exact `q=2` rank calculation yields `r_2=60`;
- the cycle transition monoid has `100` Boolean matrices;
- its deterministic response quotient has `32` states;
- all `496` unordered pairs of quotient states have explicit distinguishing continuations.

The verification scripts reconstruct these claims independently from the certificate files.

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the complete audit procedure. The repository is intentionally self-contained and does not download data during verification.

## Citation

GitHub can read the repository's `CITATION.cff`. A permanent archive DOI will be added after the first archival release. The article's arXiv identifier will also be added once assigned.

## License

The software and computational verification material are released under the MIT License; see `LICENSE`.

The manuscript is not covered by the MIT License. See `LICENSE-PAPER.md`.

## Author

Guillaume Sidoine  
ORCID: 0009-0005-4525-1182
