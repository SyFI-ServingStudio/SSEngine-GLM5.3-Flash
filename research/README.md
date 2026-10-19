# research/

The bring-up and research history of ppimpl. This directory exists only on the `dev` branch; `main` is `dev` without
it (`scripts/release.sh`). Nothing here is needed to serve, to run the tests in `tests/`, or to reproduce the results
in `docs/results/`.

| Path | Content |
|---|---|
| `PLAN.md` | the original build plan (reuse boundary, process model, milestones M1–M6) |
| `TIERS.md` | the full design log of the DRAM and SSD tiers; code comments cite its sections ("TIERS.md §5") |
| `bench/` | one-off probes and milestone harnesses: kernel re-measure and identity checks (`kernels/`, `nodecheck/`), KV-cache audit against vLLM (`kvaudit/`), correctness reference (`reference/`, `chain.py`), per-layer and per-stage benches, CUDA graph and clock probes, H2D and tier bandwidth probes, the SSD I/O benchmark (`ssd_io.py`), TTFT attribution, extended prefix targets (`targets_ext/`) |
| `bench/slurm/` | the job scripts of those harnesses, written for the cluster the project was developed on |
| `measured/` | raw and intermediate measurements of the milestones: iteration, prefix cache, scheduler, kernels, KV audit, correctness reference, margins, known issues (`issues.md`) |
| `tests/` | tests of research harnesses (`python -m pytest research/tests`; GPU ones with `-m gpu`) |

Run the harnesses from the repository root as modules, e.g. `python -m research.bench.graph_probe --help`. They import
the product package and `benchmarks/` and share `benchmarks/jobs/env.sh` for paths.

The documents here are historical: they keep the names and paths of the time they were written (the package was
`ppimpl`, the harnesses lived in `bench/`, the measurements in `measured/`).
