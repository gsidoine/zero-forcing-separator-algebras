# Reproducibility guide

## Scope

This package contains two independent layers of verification.

1. **General separator algebra.** The scripts reconstruct the atomic interface, chain counts, q=2 finite kernels, chain-state realization gadgets, the worked q=2 example, and compatibility-closure checks.
2. **Explicit one-terminal/block algorithms.** The existing scripts reproduce the rooted/tree, bounded-block, cactus, clique/cycle-block, cycle-automaton, and replacement audits.

No external data are required.

## Supported environment

- Python 3.11 or later.
- Pinned dependency: NetworkX 3.6.1 and SymPy 1.14.0.
- Reference audit environment: Python 3.13.5, NetworkX 3.6.1 and SymPy 1.14.0.
- Manuscript build: pdfTeX 1.40.26, TeX Live 2025/dev.

### Clean environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Package integrity

```bash
python verify_checksums.py SHA256SUMS.txt
```

## Exact q=2 separator certificate

```bash
python verify_separator_rank_certificate.py separator_rank_certificate.json
```

The verifier reconstructs the 52 chain states and all q=2 integer coefficient matrices from the definitions. It checks canonical SHA-256 matrix hashes, exact coefficient-span ranks over `Q`, and explicit nonsingular minors at `x=2` over `F_1000003`. No floating-point rank decision is used.

Reference certificate digest:

```text
4c7e2230e8dc088168288a82108e7b43d8b52bd2158344233fdc5b2ee81fe1c6
```

The finite certificate establishes the numerical sector ranks used for `r_2=60`; the arbitrary-q splitting, chain compression, realization, pinning, and master-rank results are proved symbolically in the manuscript.

## Separator-algebra audit

```bash
python zero_forcing_separator_audits.py
```

This audit recomputes `c_q` for `q=0,...,7`, verifies the worked q=2 example, checks every q=2 chain-realization gadget by direct prefort enumeration, checks 40 seeded q=3 realizations, exhaustively checks join stability of q=2 atomic compatibility, and independently reconstructs the q=2 rank summary.

## Cycle certificate

```bash
python verify_cycle_automaton_certificate.py cycle_automaton_certificate.json
```

Reference digest:

```text
c0fbcf130f0083b875537ffd7d4100d7a9c02397ac149e1681ff4e27bab30dc9
```

The verifier reconstructs all four Boolean generators, all 100 monoid elements, the 32-state quotient, and all 496 pairwise distinguishing continuations.

## One-command audit

```bash
python run_all_audits.py
```

`--skip-random` omits commands explicitly labeled as random; `--list` prints the commands without running them.

## Explicit-transfer command mapping

| Manuscript audit row | Command | Expected count |
|---|---|---:|
| Root normal form / pendant edge / cut vertex | `python zero_forcing_tree_dp.py --audit-atlas` | 6,781 / 6,781 / 682 |
| Ordered rooted wedges | `python zero_forcing_rooted_audits.py --audit-wedges` | 19,044 |
| Random rooted wedges | `python zero_forcing_rooted_audits.py --audit-random-wedges 350 --seed 20260726` | 350 |
| Nonisomorphic trees through order 12 | `python zero_forcing_tree_dp.py --audit-trees 12` | 987 |
| Graph-atlas bounded-block transfer | `python zero_forcing_block_dp.py --audit-atlas` | 996 |
| Random bounded-block graphs | `python zero_forcing_block_dp.py --audit-random 500 --seed 20260726` | 500 |
| Marked cycle words | `python zero_forcing_cactus_dp.py --audit-automaton` | 87,376 |
| Cactus graph atlas | `python zero_forcing_cactus_dp.py --audit-atlas` | 220 |
| Random cacti | `python zero_forcing_cactus_dp.py --audit-random 300 --seed 20260726` | 300 |
| Clique/cycle-block atlas | `python zero_forcing_cactus_dp.py --audit-mixed-atlas` | 267 |
| Random clique/cycle-block graphs | `python zero_forcing_cactus_dp.py --audit-mixed-random 300 --seed 20260906` | 300 |
| Independent replacement / clique / block / rank-four checks | `python zero_forcing_extended_audits.py --package .` | see `EXPECTED_AUDIT_OUTPUTS.txt` |

## Manuscript build

See `BUILD.md`.

## Public archival release

The package is complete as a journal supplement. The public repository is https://github.com/gsidoine/zero-forcing-separator-algebras. The v1.0.0 archival release has the reserved Zenodo DOI [10.5281/zenodo.23183837](https://doi.org/10.5281/zenodo.23183837). Publish the Zenodo record only after uploading the checksum-verified final snapshot; the mathematical certificate files must not be altered between verification and deposition.


## Public repository

https://github.com/gsidoine/zero-forcing-separator-algebras
