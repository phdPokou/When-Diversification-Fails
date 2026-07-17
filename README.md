# When Diversification Fails:
## Geopolitical Contagion and Hidden Vulnerabilities in Europe's Energy Transition

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-Auto--Detected-green.svg)]()
[![Reproducible](https://img.shields.io/badge/Reproducibility-Full-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-orange.svg)]()

Official implementation accompanying the paper

> **When Diversification Fails: Geopolitical Contagion and Hidden Vulnerabilities in Europe's Energy Transition**

This repository contains the complete reproducible computational framework
used to construct the multilayer dependency network, calibrate the empirical
model, reproduce all numerical experiments, generate every figure of the
paper, and perform all robustness analyses requested during peer review.

---

# Overview

The framework studies how geopolitical disruptions propagate through
critical-mineral supply chains supporting Europe's energy transition.

Rather than considering supplier diversification alone, the model builds
a multilayer dependency architecture connecting

```
Supplier Countries
        │
        ▼
Critical Minerals
        │
        ▼
Clean-Energy Technologies
        │
        ▼
European Economies
```

from which structural vulnerability, propagation dynamics, counterfactual
stress scenarios, and resilience-allocation policies are computed within
a unified analytical framework.

---

# Repository Structure

```
.
│
├── Data_geo/
│      Supplier/resource matrices
│      Technology/resource matrices
│      Technology portfolios
│
├── Data_geo_V10/
│      Nordic extension
│
├── Results_Geo/
│      Baseline outputs
│
├── Results_Geo_V10/
│      Reviewer robustness experiments
│
├── geo_transition_network_v8.py
│
├── geo_transition_network_v10.py
│
└── README.md
```

---

# Scientific Pipeline

The computational workflow follows the complete analytical pipeline
developed in the paper.

```
Observed datasets
        │
        ▼
Calibration
        │
        ▼
Structural matrices

A_RC
A_TR
X

        │
        ▼

Propagation operators

Q_e

        │
        ▼

Spectral quantities

χ_e
HHI
Centralities

        │
        ▼

Counterfactual scenarios

        │
        ▼

Policy-constrained resilience allocation
```

---

# Version 8

Version 8 implements the complete reproducible pipeline used in the
original computational study.

## Main Features

- automatic data download and caching
- multilayer dependency construction
- calibrated supplier-resource matrix
- calibrated technology-resource matrix
- economy-specific technology portfolios
- propagation operator construction
- spectral vulnerability computation
- resilience optimisation
- figure generation
- CSV export
- automatic CUDA support

### Computed quantities

The framework computes

- Transition capacity

\[
K_e
\]

- Transition-capacity loss

\[
L_e
\]

- Worst-case transition loss

\[
V_e
\]

- Propagation operator

\[
Q_e
\]

- Spectral structural vulnerability

\[
\chi_e
\]

- Herfindahl concentration

HHI

- Counterfactual scenario losses

- Optimal resilience allocation

---

## Design Principles

The implementation follows five principles.

### Matrix-first

No computation starts before all calibrated structural matrices satisfy
their economic consistency checks.

### Fully reproducible

Every calibrated matrix and every numerical output is automatically
stored as CSV.

### Transparent

Each generated result is associated with documented calibration metadata.

### Robust

The numerical experiments are reproducible over multiple stochastic
perturbation seeds.

### GPU-ready

CUDA acceleration is automatically enabled whenever available.

---

## Running Version 8

Rebuild everything

```bash
python geo_transition_network_v8.py \
    --rebuild-matrices \
    --force-compute
```

Generate figures only

```bash
python geo_transition_network_v8.py \
    --figures-only
```

---

# Version 10

Version 10 extends the validated Version 8 pipeline and incorporates
all empirical robustness analyses introduced during peer review. :contentReference[oaicite:0]{index=0}

## New Features

### Exact baseline replication

Replicates the complete five-country benchmark analysis.

---

### Extended European sample

Adds

- Sweden
- Denmark
- Finland

allowing structural comparisons across eight European economies. :contentReference[oaicite:1]{index=1}

---

### Structural benchmark comparison

The repository compares five complementary structural indicators:

- Technology Portfolio HHI
- Weighted-Degree Concentration
- Eigenvector Concentration
- Betweenness Concentration
- Spectral Vulnerability

---

### Sensitivity analyses

Shock intensity

- 10%
- 20%
- 30%
- 40%
- 50%

---

### Robustness analyses

- Leave-one-resource-out
- Battery-mineral bundles
- Infrastructure-mineral bundles
- Multi-seed uncertainty
- Trend-line diagnostics

---

### Figure reproducibility

Every figure is generated directly from exported CSV files.

No numerical experiment needs to be rerun to regenerate publication
figures.

---

### Safe outputs

Version 10 never overwrites Version 8 results.

---

## Running Version 10

```bash
python geo_transition_network_v10.py \
    --study both \
    --force-compute \
    --seeds 150 \
    --allow-calibrated-nordic
```

---

# Expected Inputs

```
Data_geo/

A_RC_supplier_resource.csv

A_TR_resource_technology.csv

X_ET_economy_technology.csv
```

Optional

```
Data_geo_V10/

nordic_technology_portfolios.csv
```

---

# Outputs

The repository automatically generates

- calibrated matrices
- CSV tables
- publication-quality figures
- robustness analyses
- counterfactual scenarios
- statistical summaries
- reviewer supplementary material

---

# Reproducibility

All computational experiments are deterministic after fixing the random
seed.

Every published figure can be regenerated directly from the exported CSV
files.

Version hashes of calibrated inputs are stored to guarantee complete
reproducibility of the computational pipeline. :contentReference[oaicite:2]{index=2}

---

# Citation

If you use this repository, please cite

```
Pokou F.,
Sadefo-Kamdem J.

When Diversification Fails:
Geopolitical Contagion and Hidden Vulnerabilities in Europe's Energy Transition.

Energy Policy.
```

---

# Contact

**Frédy Pokou**

INRIA Lille – CRIStAL

University of Lille

France

fredypokou@gmx.fr
