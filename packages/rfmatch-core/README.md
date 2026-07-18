# rfmatch-core

Independent RF matching optimization kernel.

This package provides the core numerical engine for RF impedance matching
network synthesis, evaluation, search, and benchmarking.

## Source layout

```
src/rfmatch_core/
├── __init__.py       # Package root; exports public API
├── network/          # Network models (S-params, Z/Y conversion, cascading)
├── evaluator/        # Cost / fitness functions for matching quality
├── search/           # Optimization and candidate search algorithms
├── transmission_line.py # Lossy uniform lines and open/short shunt stubs
├── line_optimizer.py # Bounded, cancellable line/stub topology synthesis
├── microstrip.py # Manufacturable PCB microstrip geometry, dispersion and loss
└── benchmarks/       # Performance and accuracy benchmarks
```

Implemented capabilities:

- reference-impedance-aware S/Z conversion and exact ideal series/shunt embedding;
- stable original port indexing for multiport work;
- Optenni-style minimum/average efficiency objectives;
- complete coupled-matrix evaluation at every frequency;
- deterministic multi-start continuous optimization followed by discrete snapping;
- external tutorial-data regression cases.
- physical node graphs with separate external reference planes and DUT nodes;
- parameterized lossy transmission lines, open/short shunt stubs, and conversion
  of a line into a general S2P model;
- deterministic multi-start synthesis across through-line, single-stub, and
  connector/DUT-side line-plus-stub topologies with bounded Z0 and electrical length;
- Hammerstad/Jensen microstrip geometry with finite copper thickness,
  Kirschning/Jansen dispersion, conductor/dielectric loss, width inversion, and
  manufacturability-constrained physical synthesis;
- measured component S2P interpolation with per-component dissipated-power accounting;
- reproducible Monte Carlo tolerance/yield analysis with Wilson confidence
  intervals, Gaussian-copula batch correlation, assembly-level temperature
  draws, L/C temperature coefficients, and lower-bound candidate ranking;
- physically grounded measured-S2P variation: only the nominal L/C reactance
  changes while measured ESR, shunt parasitics, asymmetry, and residual error
  remain intact;
- point-by-point Optenni golden CSV comparison.
- shared-network multi-scenario ideal and measured-S2P synthesis with explicit
  weighted-average/worst-scenario dB objectives;
- topology-diverse physical refinement that reproduces the official
  0402CS 4.3 nH + GJM15 5.6 pF multiple-configuration tutorial result.
- bounded exhaustive measured-S2P calibration with exact part-number and
  topology top-k recall, best-score gap, evaluation-count, and model-load metrics;
- MDIF multi-state parsing (RI/MA/DB and Hz/kHz/MHz/GHz), deterministic state
  assignment, and measured shared-network evaluation with exact power balance;
- a reproducible variable-capacitor benchmark that recovers the official
  `8 pF / 2 pF / 1 pF` state map with a `-1.6406 dB` balanced score.

## External references

Optenni tutorial files used as external reference inputs during development
remain at the user-provided external installation path. They are **not** copied
or vendored into this package. Benchmark manifests accept that path through
`--tutorial-root`.

## Requirements

- Python >= 3.10
- numpy (runtime)
- pytest (optional, for development)

## Run tests

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m unittest discover -s tests -v
```

## Run a reference case

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m rfmatch_core.benchmarks `
  --tutorial-root "E:\ProgramX\OptenniLab\Optenni Lab Tutorials" `
  --case quick-start
```

The numerical evaluator is the stable contract. New global-search strategies
can be introduced without changing network calculations or benchmark inputs.
