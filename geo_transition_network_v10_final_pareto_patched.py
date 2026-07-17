#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geopolitical contagion in the energy transition — final revision and manuscript-packaging pipeline (V10)
==================================================================================

V10 integrates the validated V8 pipeline and adds the empirical extensions requested
during peer review:

1. Exact five-economy baseline replication.
2. Extended Nordic sample: Sweden, Denmark, and Finland.
3. Side-by-side comparison of:
      - technology-portfolio HHI,
      - weighted-degree concentration,
      - eigenvector-centrality concentration,
      - betweenness-centrality concentration,
      - spectral vulnerability chi.
4. Shock-intensity sensitivity for 10%, 20%, 30%, 40%, and 50%.
5. Leave-one-resource-out robustness.
6. Battery-mineral and infrastructure-mineral bundle robustness.
7. Scatterplots with trend lines and seed-based uncertainty.
8. CSV export for every figure, allowing figures to be regenerated without
   rerunning the numerical experiments.
9. Separate output folders, so the original V8 results are never overwritten.

Expected files in the same directory:
    geo_transition_network_v8.py

Expected baseline matrices:
    Data_geo/A_RC_supplier_resource.csv
    Data_geo/A_TR_resource_technology.csv
    Data_geo/X_ET_economy_technology.csv

Optional Nordic data file:
    Data_geo_V10/nordic_technology_portfolios.csv

Accepted Nordic formats
-----------------------
Wide format:
    economy,EV batteries,Wind,Solar PV,Electrolysers,Power grids

Long format:
    economy,technology,weight

If the Nordic file is missing, the script can use explicitly flagged calibrated
extension rows with --allow-calibrated-nordic. These fallback rows are intended for
pipeline testing and preliminary robustness only. For the revised manuscript,
replace them with harmonised Eurostat/IRENA/EHO data.

Examples
--------
Full reviewer package:
    python geo_transition_network_v10.py --study both --force-compute --seeds 150 \
        --allow-calibrated-nordic

Use observed Nordic portfolio file:
    python geo_transition_network_v10.py --study both --force-compute --seeds 150 \
        --nordic-file Data_geo_V10/nordic_technology_portfolios.csv

Regenerate all figures from CSV files only:
    python geo_transition_network_v10.py --figures-only

Create a Nordic input template:
    python geo_transition_network_v10.py --create-nordic-template

Outputs
-------
Results_Geo_V10/
    Baseline_5/
    Extended_Nordic_8/
    Comparative/
    Figures/
    Tables/
    Figure_Data/
    run_manifest_v10.json
"""

from __future__ import annotations

# Windows / Anaconda OpenMP safety: set before NumPy, SciPy, Torch, or matplotlib.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import networkx as nx
except Exception:
    nx = None


# =============================================================================
# 1. Configuration
# =============================================================================

BASE_ECONOMIES = ["France", "Germany", "Italy", "Netherlands", "Spain"]
NORDIC_ECONOMIES = ["Sweden", "Denmark", "Finland"]
EXTENDED_ECONOMIES = BASE_ECONOMIES + NORDIC_ECONOMIES

RESOURCES = ["Lithium", "Nickel", "Cobalt", "Graphite", "Rare earths", "Copper"]
TECHNOLOGIES = ["EV batteries", "Wind", "Solar PV", "Electrolysers", "Power grids"]

SHOCK_INTENSITIES = [0.10, 0.20, 0.30, 0.40, 0.50]
DEFAULT_SEED_START = 2027
DEFAULT_N_SEEDS = 150

ROOT = Path(".")
BASE_DATA_DIR = ROOT / "Data_geo"
REVIEW_DATA_DIR = ROOT / "Data_geo_V10"
REVIEW_RESULTS_DIR = ROOT / "Results_Geo_V10"
BASELINE_DIR = REVIEW_RESULTS_DIR / "Baseline_5"
EXTENDED_DIR = REVIEW_RESULTS_DIR / "Extended_Nordic_8"
COMPARATIVE_DIR = REVIEW_RESULTS_DIR / "Comparative"
FIGURE_DIR = REVIEW_RESULTS_DIR / "Figures"
TABLE_DIR = REVIEW_RESULTS_DIR / "Tables"
FIGURE_DATA_DIR = REVIEW_RESULTS_DIR / "Figure_Data"

# Publication-facing package. The folders below contain only the results selected
# for the revised manuscript; all remaining outputs are preserved in Appendix.
MANUSCRIPT_DIR = REVIEW_RESULTS_DIR / "00_Manuscript"
MAIN_FIGURES_DIR = MANUSCRIPT_DIR / "Main_Figures"
MAIN_FIGURE_DATA_DIR = MANUSCRIPT_DIR / "Main_Figure_Data"
MAIN_TABLES_DIR = MANUSCRIPT_DIR / "Main_Tables"
APPENDIX_DIR = REVIEW_RESULTS_DIR / "04_Appendix"
APPENDIX_FIGURES_DIR = APPENDIX_DIR / "Figures"
APPENDIX_FIGURE_DATA_DIR = APPENDIX_DIR / "Figure_Data"
APPENDIX_TABLES_DIR = APPENDIX_DIR / "Tables"
REPRO_DIR = REVIEW_RESULTS_DIR / "05_Reproducibility"

for directory in [
    REVIEW_DATA_DIR, REVIEW_RESULTS_DIR, BASELINE_DIR, EXTENDED_DIR,
    COMPARATIVE_DIR, FIGURE_DIR, TABLE_DIR, FIGURE_DATA_DIR,
    MANUSCRIPT_DIR, MAIN_FIGURES_DIR, MAIN_FIGURE_DATA_DIR, MAIN_TABLES_DIR,
    APPENDIX_DIR, APPENDIX_FIGURES_DIR, APPENDIX_FIGURE_DATA_DIR,
    APPENDIX_TABLES_DIR, REPRO_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# Economy-level implementation sensitivity. Values for the original five economies
# reproduce V8. Nordic values are neutral unless replaced by an observed covariate.
COUNTRY_SENSITIVITY_V10 = {
    "France": 0.90,
    "Germany": 1.01,
    "Italy": 1.07,
    "Netherlands": 1.14,
    "Spain": 1.18,
    "Sweden": 0.96,
    "Denmark": 0.98,
    "Finland": 1.00,
}

# Explicitly labelled fallback rows. These are not presented as raw observed data.
CALIBRATED_NORDIC_PORTFOLIOS = pd.DataFrame(
    {
        "EV batteries": [0.27, 0.24, 0.20],
        "Wind": [0.31, 0.42, 0.30],
        "Solar PV": [0.12, 0.15, 0.10],
        "Electrolysers": [0.11, 0.10, 0.12],
        "Power grids": [0.19, 0.09, 0.28],
    },
    index=NORDIC_ECONOMIES,
)


@dataclass(frozen=True)
class StudySpec:
    name: str
    economies: Tuple[str, ...]
    output_dir: Path


BASELINE_SPEC = StudySpec(
    name="baseline_5",
    economies=tuple(BASE_ECONOMIES),
    output_dir=BASELINE_DIR,
)

EXTENDED_SPEC = StudySpec(
    name="extended_nordic_8",
    economies=tuple(EXTENDED_ECONOMIES),
    output_dir=EXTENDED_DIR,
)


# =============================================================================
# 2. V8 import and generic utilities
# =============================================================================

def import_v8(module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(
            f"Cannot find {module_path}. Place geo_transition_network_v8.py "
            "in the same directory as this V10 script or provide --v8-script."
        )
    spec = importlib.util.spec_from_file_location("geo_v8", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load V8 module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["geo_v8"] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.astype(float).copy()
    sums = out.sum(axis=1)
    if (sums <= 0).any():
        bad = sums[sums <= 0].index.tolist()
        raise ValueError(f"Rows with non-positive sum: {bad}")
    return out.div(sums, axis=0)


def safe_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.std(x, ddof=0)
    return (x - np.mean(x)) / sd if sd > 1e-12 else np.zeros_like(x)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum((y - fitted) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-12 else np.nan
    return float(beta[0]), float(beta[1]), r2


def rank_stability(df: pd.DataFrame, metric: str, outcome: str) -> Tuple[float, float]:
    rows = []
    for _, group in df.groupby("seed"):
        if group[metric].nunique() < 2 or group[outcome].nunique() < 2:
            continue
        if stats is not None:
            rho, _ = stats.spearmanr(group[metric], group[outcome])
        else:
            rho = pd.Series(group[metric]).rank().corr(pd.Series(group[outcome]).rank())
        if np.isfinite(rho):
            rows.append(float(rho))
    return (
        float(np.mean(rows)) if rows else np.nan,
        float(np.std(rows, ddof=1)) if len(rows) > 1 else np.nan,
    )


# =============================================================================
# 3. Portfolio data and study matrices
# =============================================================================

def create_nordic_template(path: Path) -> None:
    template = pd.DataFrame(
        [
            {
                "economy": economy,
                "technology": technology,
                "weight": "",
                "raw_value": "",
                "year": "",
                "unit": "",
                "source": "",
                "dataset_code": "",
            }
            for economy in NORDIC_ECONOMIES
            for technology in TECHNOLOGIES
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    print(f"[done] Nordic data template created: {path}")


def load_nordic_portfolios(
    path: Path,
    allow_calibrated: bool,
) -> Tuple[pd.DataFrame, str]:
    if path.exists():
        raw = pd.read_csv(path)
        if {"economy", "technology", "weight"}.issubset(raw.columns):
            pivot = raw.pivot_table(
                index="economy",
                columns="technology",
                values="weight",
                aggfunc="sum",
            )
        elif "economy" in raw.columns and set(TECHNOLOGIES).issubset(raw.columns):
            pivot = raw.set_index("economy")[TECHNOLOGIES]
        else:
            raise ValueError(
                f"{path} must be either long format "
                "(economy, technology, weight) or wide format "
                f"(economy plus {TECHNOLOGIES})."
            )
        missing_e = [e for e in NORDIC_ECONOMIES if e not in pivot.index]
        missing_t = [t for t in TECHNOLOGIES if t not in pivot.columns]
        if missing_e or missing_t:
            raise ValueError(
                f"Nordic portfolio file is incomplete. Missing economies={missing_e}; "
                f"missing technologies={missing_t}."
            )
        out = normalize_rows(pivot.loc[NORDIC_ECONOMIES, TECHNOLOGIES])
        return out, f"observed_or_user_supplied:{path.as_posix()}"

    if not allow_calibrated:
        raise FileNotFoundError(
            f"Nordic portfolio file not found: {path}\n"
            "Provide a harmonised file or rerun with --allow-calibrated-nordic "
            "for a clearly flagged preliminary calibration."
        )

    warnings.warn(
        "Nordic portfolio file is missing. Using explicitly flagged calibrated "
        "fallback rows. Replace these rows with harmonised observed data before "
        "the final revised submission."
    )
    return normalize_rows(CALIBRATED_NORDIC_PORTFOLIOS), "calibrated_fallback_v10"


def prepare_study_matrices(
    v8,
    spec: StudySpec,
    nordic_file: Path,
    allow_calibrated_nordic: bool,
    rebuild: bool,
) -> Tuple[object, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    data_dir = spec.output_dir / "Data"
    results_dir = spec.output_dir / "Results"
    tables_dir = spec.output_dir / "Tables"
    figures_dir = spec.output_dir / "Figures"
    ensure_dirs(data_dir, results_dir, tables_dir, figures_dir)

    paths = v8.Paths(
        root=ROOT,
        data=data_dir,
        results=results_dir,
        tables=tables_dir,
        figures=figures_dir,
    )
    paths.ensure()

    # The global matrices are copied from the validated V8 data directory.
    arc_src = BASE_DATA_DIR / "A_RC_supplier_resource.csv"
    atr_src = BASE_DATA_DIR / "A_TR_resource_technology.csv"
    x_src = BASE_DATA_DIR / "X_ET_economy_technology.csv"

    missing = [p for p in [arc_src, atr_src, x_src] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Baseline V8 matrices are missing: "
            + ", ".join(str(p) for p in missing)
            + ". Run V8 with --rebuild-matrices first."
        )

    A_RC = pd.read_csv(arc_src, index_col=0)
    A_TR = pd.read_csv(atr_src, index_col=0)
    X_base = pd.read_csv(x_src, index_col=0)

    if spec.name == "baseline_5":
        X = normalize_rows(X_base.loc[BASE_ECONOMIES, TECHNOLOGIES])
        x_source = f"frozen_v8:{x_src.as_posix()}"
    else:
        X_nordic, x_source = load_nordic_portfolios(
            nordic_file, allow_calibrated=allow_calibrated_nordic
        )
        X = pd.concat(
            [
                normalize_rows(X_base.loc[BASE_ECONOMIES, TECHNOLOGIES]),
                X_nordic,
            ],
            axis=0,
        ).loc[list(spec.economies), TECHNOLOGIES]
        X = normalize_rows(X)

    A_RC.to_csv(data_dir / "A_RC_supplier_resource.csv")
    A_TR.to_csv(data_dir / "A_TR_resource_technology.csv")
    X.to_csv(data_dir / "X_ET_economy_technology.csv")

    # V8's summary-table exporter requires a file named exactly
    # ``matrix_metadata.csv`` with the columns below.  V9 also writes the same
    # information under a versioned filename for auditability.
    metadata = pd.DataFrame(
        [
            {
                "component": "A_RC",
                "description": "Supplier-country shares in global critical-mineral production",
                "primary_source": f"Validated V8 matrix ({arc_src.as_posix()})",
                "status": "frozen_v8",
                "sample": spec.name,
            },
            {
                "component": "A_TR",
                "description": "Normalized material dependence of clean-energy technologies",
                "primary_source": f"Validated V8 matrix ({atr_src.as_posix()})",
                "status": "frozen_v8",
                "sample": spec.name,
            },
            {
                "component": "X_ET",
                "description": "Economy-specific clean-energy technology portfolio shares",
                "primary_source": x_source,
                "status": "observed_or_calibrated",
                "sample": spec.name,
            },
            {
                "component": "WGI",
                "description": "Political-stability risk proxy (contextual only in V9)",
                "primary_source": "World Bank WGI when available; not a direct driver of the structural matrices",
                "status": "contextual_optional",
                "sample": spec.name,
            },
        ]
    )
    metadata.to_csv(data_dir / "matrix_metadata.csv", index=False)
    metadata.to_csv(data_dir / "matrix_metadata_v10.csv", index=False)

    # V8 relies on module-level economy lists and sensitivity values.
    v8.ECONOMIES = list(spec.economies)
    v8.COUNTRY_SENSITIVITY = {
        e: COUNTRY_SENSITIVITY_V10.get(e, 1.0) for e in spec.economies
    }

    # Dynamic structural validation. Strict V8 checks are retained for the original
    # five economies; the Nordic rows are checked for simplex feasibility.
    v8.validate_matrices(A_RC, A_TR, X, strict=False)
    if not np.allclose(X.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Technology portfolio rows must sum to one.")
    if (X.values < -1e-12).any():
        raise ValueError("Technology portfolios contain negative entries.")

    return paths, A_RC, A_TR, X, x_source


# =============================================================================
# 4. Alternative vulnerability metrics
# =============================================================================

def projected_supplier_matrix(
    A_RC: pd.DataFrame,
    A_TR: pd.DataFrame,
    x_row: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    M = A_TR.values @ A_RC.values  # technology x supplier
    Q = np.diag(x_row.values.astype(float)) @ M
    W = Q.T @ Q
    W = np.asarray(W, dtype=float)
    np.fill_diagonal(W, 0.0)
    return Q, W


def weighted_degree_hhi(Q: np.ndarray) -> float:
    strength = np.maximum(Q.sum(axis=0), 0.0)
    total = strength.sum()
    if total <= 1e-12:
        return 0.0
    p = strength / total
    return float(np.sum(p ** 2))


def eigenvector_concentration(W: np.ndarray) -> float:
    if W.size == 0 or np.max(np.abs(W)) <= 1e-14:
        return 0.0
    # Symmetric weighted supplier projection: principal eigenvector is deterministic.
    eigvals, eigvecs = np.linalg.eigh(W)
    vec = np.abs(eigvecs[:, np.argmax(eigvals)])
    total = vec.sum()
    if total <= 1e-12:
        return 0.0
    p = vec / total
    return float(np.sum(p ** 2))


def betweenness_concentration(
    Q: np.ndarray,
    technology_names: Sequence[str],
    supplier_names: Sequence[str],
) -> float:
    if nx is None:
        return np.nan

    graph = nx.Graph()
    technology_nodes = [f"T::{t}" for t in technology_names]
    supplier_nodes = [f"S::{s}" for s in supplier_names]
    graph.add_nodes_from(technology_nodes, bipartite=0)
    graph.add_nodes_from(supplier_nodes, bipartite=1)

    for i, t_node in enumerate(technology_nodes):
        for j, s_node in enumerate(supplier_nodes):
            weight = float(Q[i, j])
            if weight > 1e-12:
                graph.add_edge(
                    t_node,
                    s_node,
                    weight=weight,
                    distance=1.0 / (weight + 1e-12),
                )

    centrality = nx.betweenness_centrality(
        graph,
        normalized=True,
        weight="distance",
    )
    vals = np.array([centrality.get(node, 0.0) for node in supplier_nodes], dtype=float)
    total = vals.sum()
    if total <= 1e-12:
        return 0.0
    p = vals / total
    return float(np.sum(p ** 2))


def compute_alternative_metrics(
    A_RC: pd.DataFrame,
    A_TR: pd.DataFrame,
    X: pd.DataFrame,
    v8,
) -> pd.DataFrame:
    chi = v8.spectral_chi(A_RC, A_TR, X)
    rows = []
    for economy in X.index:
        x_row = X.loc[economy]
        Q, W = projected_supplier_matrix(A_RC, A_TR, x_row)
        rows.append(
            {
                "economy": economy,
                "HHI": float(np.sum(x_row.values ** 2)),
                "chi": float(chi.loc[economy]),
                "weighted_degree_hhi": weighted_degree_hhi(Q),
                "eigenvector_concentration": eigenvector_concentration(W),
                "betweenness_concentration": betweenness_concentration(
                    Q, A_TR.index.tolist(), A_RC.columns.tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 5. Reviewer extension experiments
# =============================================================================

def run_metric_panel(
    v8,
    spec: StudySpec,
    paths,
    A_RC0: pd.DataFrame,
    A_TR0: pd.DataFrame,
    X0: pd.DataFrame,
    seeds: Sequence[int],
    force_compute: bool,
) -> pd.DataFrame:
    out_file = spec.output_dir / "Results" / "alternative_metrics_seed_level.csv"
    if out_file.exists() and not force_compute:
        return pd.read_csv(out_file)

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        A_RC = v8.perturb_matrix(A_RC0, v8.PERTURB_CONCENTRATION_ARC, rng)
        A_TR = v8.perturb_matrix(A_TR0, v8.PERTURB_CONCENTRATION_ATR, rng)
        X = v8.perturb_matrix(X0, v8.PERTURB_CONCENTRATION_X, rng)
        macro = v8.sample_macro_sensitivity(seed, list(X.index))

        metric_df = compute_alternative_metrics(A_RC, A_TR, X, v8)
        losses = {}
        for economy in X.index:
            loss, _ = v8.worst_case_loss_for_economy(
                A_RC, A_TR, X, economy, delta=v8.DELTA_WORSTCASE, seed=seed
            )
            losses[economy] = float(loss) * float(macro.loc[economy])

        metric_df["seed"] = seed
        metric_df["sample"] = spec.name
        metric_df["worstcase_loss_pct"] = metric_df["economy"].map(losses)
        rows.append(metric_df)

    result = pd.concat(rows, ignore_index=True)
    save_dataframe(result, out_file)
    return result


def scenario_vectors_at_intensity(
    A_RC: pd.DataFrame,
    intensity: float,
) -> Dict[str, np.ndarray]:
    columns = list(A_RC.columns)

    def vector(assignments: Mapping[str, float]) -> np.ndarray:
        z = np.zeros(len(columns), dtype=float)
        for supplier, value in assignments.items():
            if supplier in columns:
                z[columns.index(supplier)] = value
        return z

    # Each listed supplier receives the same proportional supply loss. This mirrors
    # the original V8 counterfactual construction and makes the 10–50% sequence
    # directly interpretable.
    return {
        "China REE+Graphite shock": vector({"China": intensity}),
        "Indonesia nickel shock": vector({"Indonesia": intensity}),
        "Lithium shock (Chile+Australia)": vector(
            {"Chile": intensity, "Australia": intensity}
        ),
        "Dual shock (China+Russia)": vector(
            {"China": intensity, "Russia": intensity}
        ),
        "Copper shock (Chile+Peru)": vector(
            {"Chile": intensity, "Peru": intensity}
        ),
    }


def run_shock_intensity_sensitivity(
    v8,
    spec: StudySpec,
    A_RC0: pd.DataFrame,
    A_TR0: pd.DataFrame,
    X0: pd.DataFrame,
    seeds: Sequence[int],
    force_compute: bool,
) -> pd.DataFrame:
    out_file = spec.output_dir / "Results" / "shock_intensity_sensitivity_seed_level.csv"
    if out_file.exists() and not force_compute:
        return pd.read_csv(out_file)

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        A_RC = v8.perturb_matrix(A_RC0, v8.PERTURB_CONCENTRATION_ARC, rng)
        A_TR = v8.perturb_matrix(A_TR0, v8.PERTURB_CONCENTRATION_ATR, rng)
        X = v8.perturb_matrix(X0, v8.PERTURB_CONCENTRATION_X, rng)
        macro = v8.sample_macro_sensitivity(seed, list(X.index))

        for intensity in SHOCK_INTENSITIES:
            for scenario, z_vec in scenario_vectors_at_intensity(A_RC, intensity).items():
                losses = v8.compute_loss_pct(A_RC, A_TR, X, z_vec)
                for economy, value in losses.items():
                    rows.append(
                        {
                            "sample": spec.name,
                            "seed": seed,
                            "intensity": intensity,
                            "scenario": scenario,
                            "economy": economy,
                            "loss_pct": float(value) * float(macro.loc[economy]),
                        }
                    )

    result = pd.DataFrame(rows)
    save_dataframe(result, out_file)
    return result


_RESOURCE_SUBSET_WARNED: set = set()


def subset_resource_matrices(
    A_RC: pd.DataFrame,
    A_TR: pd.DataFrame,
    X: pd.DataFrame,
    kept_resources: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a coherent resource-subset system.

    Some restricted resource bundles do not support every technology. For example,
    the battery-mineral bundle excludes both copper and rare earths, leaving the
    Wind row of A_TR with zero mass. Rather than forcing an arbitrary uniform row,
    V9 removes technologies with no remaining material support and renormalizes
    each economy's portfolio over the active technologies.

    This preserves the economic meaning of the robustness exercise:
    the restricted system is evaluated only over technologies that can actually
    be represented by the retained resource set.
    """
    kept = [r for r in kept_resources if r in A_RC.index and r in A_TR.columns]
    if len(kept) < 2:
        raise ValueError("At least two resources are required for a robustness specification.")

    arc = A_RC.loc[kept].copy()
    atr_raw = A_TR.loc[:, kept].copy()

    row_sums = atr_raw.sum(axis=1)
    active_technologies = row_sums[row_sums > 1e-12].index.tolist()
    inactive_technologies = row_sums[row_sums <= 1e-12].index.tolist()

    if not active_technologies:
        raise ValueError(
            "No technology remains supported by the retained resources: "
            + ", ".join(kept)
        )

    atr = normalize_rows(atr_raw.loc[active_technologies])

    x_active = X.loc[:, active_technologies].copy()
    x_sums = x_active.sum(axis=1)
    if (x_sums <= 1e-12).any():
        bad = x_sums[x_sums <= 1e-12].index.tolist()
        raise ValueError(
            "Some economies have no portfolio mass on technologies supported by "
            f"the retained resources {kept}: {bad}"
        )
    x_active = x_active.div(x_sums, axis=0)

    if inactive_technologies:
        warning_key = (tuple(inactive_technologies), tuple(kept))
        if warning_key not in _RESOURCE_SUBSET_WARNED:
            warnings.warn(
                "Resource-subset robustness removed unsupported technologies "
                f"{inactive_technologies} for retained resources {kept}. "
                "This message is emitted once per unique specification."
            )
            _RESOURCE_SUBSET_WARNED.add(warning_key)

    return arc, atr, x_active


def run_resource_robustness(
    v8,
    spec: StudySpec,
    A_RC0: pd.DataFrame,
    A_TR0: pd.DataFrame,
    X0: pd.DataFrame,
    seeds: Sequence[int],
    force_compute: bool,
) -> pd.DataFrame:
    out_file = spec.output_dir / "Results" / "resource_robustness_seed_level.csv"
    if out_file.exists() and not force_compute:
        return pd.read_csv(out_file)

    specifications: Dict[str, List[str]] = {"Full specification": list(RESOURCES)}
    for omitted in RESOURCES:
        specifications[f"Without {omitted}"] = [r for r in RESOURCES if r != omitted]
    specifications["Battery-mineral bundle"] = [
        "Lithium", "Nickel", "Cobalt", "Graphite"
    ]
    specifications["Infrastructure-mineral bundle"] = ["Copper", "Rare earths"]

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        A_RC_full = v8.perturb_matrix(
            A_RC0, v8.PERTURB_CONCENTRATION_ARC, rng
        )
        A_TR_full = v8.perturb_matrix(
            A_TR0, v8.PERTURB_CONCENTRATION_ATR, rng
        )
        X = v8.perturb_matrix(X0, v8.PERTURB_CONCENTRATION_X, rng)
        macro = v8.sample_macro_sensitivity(seed, list(X.index))

        for specification, resources in specifications.items():
            A_RC, A_TR, X_sub = subset_resource_matrices(
                A_RC_full, A_TR_full, X, resources
            )
            chi = v8.spectral_chi(A_RC, A_TR, X_sub)
            for economy in X_sub.index:
                loss, _ = v8.worst_case_loss_for_economy(
                    A_RC,
                    A_TR,
                    X_sub,
                    economy,
                    delta=v8.DELTA_WORSTCASE,
                    seed=seed,
                )
                rows.append(
                    {
                        "sample": spec.name,
                        "seed": seed,
                        "specification": specification,
                        "economy": economy,
                        "n_resources": len(resources),
                        "n_active_technologies": int(X_sub.shape[1]),
                        "active_technologies": "|".join(X_sub.columns.tolist()),
                        "chi": float(chi.loc[economy]),
                        "worstcase_loss_pct": float(loss) * float(macro.loc[economy]),
                    }
                )

    result = pd.DataFrame(rows)
    save_dataframe(result, out_file)
    return result


# =============================================================================
# 6. Comparative statistical outputs
# =============================================================================

METRIC_COLUMNS = [
    "HHI",
    "weighted_degree_hhi",
    "eigenvector_concentration",
    "betweenness_concentration",
    "chi",
]

METRIC_LABELS = {
    "HHI": "Technology HHI",
    "weighted_degree_hhi": "Weighted-degree HHI",
    "eigenvector_concentration": "Eigenvector-centrality concentration",
    "betweenness_concentration": "Betweenness-centrality concentration",
    "chi": "Spectral vulnerability (chi)",
}


def univariate_performance(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clean_panel = panel.replace([np.inf, -np.inf], np.nan)
    for metric in METRIC_COLUMNS:
        data = clean_panel[[metric, "worstcase_loss_pct"]].dropna()
        x = data[metric].to_numpy(float)
        y = data["worstcase_loss_pct"].to_numpy(float)
        intercept, slope, r2 = linear_fit(x, y)

        if stats is not None and len(data) >= 3:
            pearson, pearson_p = stats.pearsonr(x, y)
            spearman, spearman_p = stats.spearmanr(x, y)
        else:
            pearson = np.corrcoef(x, y)[0, 1] if len(data) >= 2 else np.nan
            pearson_p = np.nan
            spearman = pd.Series(x).rank().corr(pd.Series(y).rank())
            spearman_p = np.nan

        rank_mean, rank_sd = rank_stability(data.assign(seed=panel.loc[data.index, "seed"]), metric, "worstcase_loss_pct")

        rows.append(
            {
                "sample": panel["sample"].iloc[0],
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "n_obs": len(data),
                "intercept": intercept,
                "slope": slope,
                "r2": r2,
                "pearson_r": pearson,
                "pearson_p": pearson_p,
                "spearman_rho": spearman,
                "spearman_p": spearman_p,
                "seed_rank_rho_mean": rank_mean,
                "seed_rank_rho_sd": rank_sd,
            }
        )
    return pd.DataFrame(rows)


def incremental_r2(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["HHI", "chi", "weighted_degree_hhi", "eigenvector_concentration",
                "betweenness_concentration", "worstcase_loss_pct"]
    )
    y = data["worstcase_loss_pct"].to_numpy(float)

    def model_r2(cols: Sequence[str]) -> float:
        X = np.column_stack(
            [np.ones(len(data))]
            + [safe_standardize(data[c].to_numpy(float)) for c in cols]
        )
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted = X @ beta
        sst = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - float(np.sum((y - fitted) ** 2)) / sst if sst > 1e-12 else np.nan

    base_r2 = model_r2(["HHI"])
    rows = []
    for metric in [m for m in METRIC_COLUMNS if m != "HHI"]:
        full_r2 = model_r2(["HHI", metric])
        rows.append(
            {
                "sample": panel["sample"].iloc[0],
                "baseline_model": "HHI",
                "added_metric": metric,
                "added_metric_label": METRIC_LABELS[metric],
                "baseline_r2": base_r2,
                "full_r2": full_r2,
                "incremental_r2": full_r2 - base_r2,
                "n_obs": len(data),
            }
        )
    return pd.DataFrame(rows)


def summarize_resource_robustness(df: pd.DataFrame) -> pd.DataFrame:
    full = df[df["specification"] == "Full specification"][
        ["seed", "economy", "chi", "worstcase_loss_pct"]
    ].rename(
        columns={
            "chi": "chi_full",
            "worstcase_loss_pct": "loss_full",
        }
    )
    merged = df.merge(full, on=["seed", "economy"], how="left")
    merged["chi_change_pct"] = 100.0 * (
        merged["chi"] - merged["chi_full"]
    ) / np.maximum(np.abs(merged["chi_full"]), 1e-12)
    merged["loss_change_pct"] = 100.0 * (
        merged["worstcase_loss_pct"] - merged["loss_full"]
    ) / np.maximum(np.abs(merged["loss_full"]), 1e-12)

    summary = (
        merged.groupby(["sample", "specification"], as_index=False)
        .agg(
            chi_change_mean=("chi_change_pct", "mean"),
            chi_change_sd=("chi_change_pct", "std"),
            loss_change_mean=("loss_change_pct", "mean"),
            loss_change_sd=("loss_change_pct", "std"),
        )
    )

    rank_rows = []
    for (sample, specification, seed), group in merged.groupby(
        ["sample", "specification", "seed"]
    ):
        if group["chi"].nunique() > 1 and group["chi_full"].nunique() > 1:
            rho_chi = group["chi"].rank().corr(group["chi_full"].rank())
        else:
            rho_chi = np.nan
        if group["worstcase_loss_pct"].nunique() > 1 and group["loss_full"].nunique() > 1:
            rho_loss = group["worstcase_loss_pct"].rank().corr(group["loss_full"].rank())
        else:
            rho_loss = np.nan
        rank_rows.append(
            {
                "sample": sample,
                "specification": specification,
                "seed": seed,
                "rank_rho_chi": rho_chi,
                "rank_rho_loss": rho_loss,
            }
        )
    rank_summary = (
        pd.DataFrame(rank_rows)
        .groupby(["sample", "specification"], as_index=False)
        .agg(
            rank_rho_chi_mean=("rank_rho_chi", "mean"),
            rank_rho_loss_mean=("rank_rho_loss", "mean"),
        )
    )
    return summary.merge(rank_summary, on=["sample", "specification"], how="left")


# =============================================================================
# 7. Figure-data preparation and Q1++ figures
# =============================================================================

def aggregate_metric_panel(panel: pd.DataFrame) -> pd.DataFrame:
    cols = METRIC_COLUMNS + ["worstcase_loss_pct"]
    summary = panel.groupby(["sample", "economy"])[cols].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


def prepare_all_figure_data(
    baseline_panel: pd.DataFrame,
    extended_panel: pd.DataFrame,
    baseline_intensity: pd.DataFrame,
    extended_intensity: pd.DataFrame,
    extended_resource: pd.DataFrame,
) -> None:
    metric_summary = pd.concat(
        [
            aggregate_metric_panel(baseline_panel),
            aggregate_metric_panel(extended_panel),
        ],
        ignore_index=True,
    )
    save_dataframe(
        metric_summary,
        FIGURE_DATA_DIR / "FigR1_R2_metric_comparison.csv",
    )

    ranking_cols = METRIC_COLUMNS + ["worstcase_loss_pct"]
    ranks = []
    for sample, group in metric_summary.groupby("sample"):
        temp = group[["sample", "economy"]].copy()
        for metric in ranking_cols:
            col = f"{metric}_mean"
            temp[f"rank_{metric}"] = group[col].rank(
                method="average", ascending=False
            )
        ranks.append(temp)
    save_dataframe(
        pd.concat(ranks, ignore_index=True),
        FIGURE_DATA_DIR / "FigR3_country_rankings.csv",
    )

    intensity_summary = (
        pd.concat([baseline_intensity, extended_intensity], ignore_index=True)
        .groupby(["sample", "intensity", "scenario"], as_index=False)
        .agg(
            loss_mean=("loss_pct", "mean"),
            loss_sd=("loss_pct", "std"),
        )
    )
    save_dataframe(
        intensity_summary,
        FIGURE_DATA_DIR / "FigR4_shock_intensity_sensitivity.csv",
    )

    extended_baseline = aggregate_metric_panel(extended_panel)
    save_dataframe(
        extended_baseline,
        FIGURE_DATA_DIR / "FigR5_extended_sample_vulnerability.csv",
    )

    resource_summary = summarize_resource_robustness(extended_resource)
    save_dataframe(
        resource_summary,
        FIGURE_DATA_DIR / "FigR6_resource_robustness.csv",
    )


def make_figure_r1() -> None:
    df = pd.read_csv(FIGURE_DATA_DIR / "FigR1_R2_metric_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=False)

    for ax, sample, title in zip(
        axes,
        ["baseline_5", "extended_nordic_8"],
        ["Original five-economy sample", "Extended Nordic sample"],
    ):
        data = df[df["sample"] == sample]
        ax.errorbar(
            data["HHI_mean"],
            data["chi_mean"],
            xerr=data["HHI_std"],
            yerr=data["chi_std"],
            fmt="o",
            capsize=3,
            linewidth=1,
        )
        for _, row in data.iterrows():
            ax.annotate(
                row["economy"],
                (row["HHI_mean"], row["chi_mean"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
        x = data["HHI_mean"].to_numpy(float)
        y = data["chi_mean"].to_numpy(float)
        intercept, slope, r2 = linear_fit(x, y)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, linewidth=1.5)
        ax.text(
            0.03,
            0.96,
            f"$R^2={r2:.2f}$",
            transform=ax.transAxes,
            va="top",
        )
        ax.set_title(title)
        ax.set_xlabel("Technology-portfolio HHI")
        ax.set_ylabel(r"Spectral vulnerability $\chi$")
        ax.grid(alpha=0.2)

    save_figure(fig, "FigR1_HHI_vs_spectral_concentration")


def make_figure_r2() -> None:
    df = pd.read_csv(FIGURE_DATA_DIR / "FigR1_R2_metric_comparison.csv")
    data = df[df["sample"] == "extended_nordic_8"].copy()

    metrics = [
        ("HHI_mean", "Technology HHI"),
        ("eigenvector_concentration_mean", "Eigenvector concentration"),
        ("betweenness_concentration_mean", "Betweenness concentration"),
        ("chi_mean", r"Spectral vulnerability $\chi$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharey=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        x = data[metric].to_numpy(float)
        y = data["worstcase_loss_pct_mean"].to_numpy(float)
        ax.scatter(x, y, s=42)
        for _, row in data.iterrows():
            ax.annotate(
                row["economy"],
                (row[metric], row["worstcase_loss_pct_mean"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.5,
            )
        intercept, slope, r2 = linear_fit(x, y)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, linewidth=1.5)
        ax.text(0.03, 0.96, f"$R^2={r2:.2f}$", transform=ax.transAxes, va="top")
        ax.set_xlabel(label)
        ax.set_ylabel("Worst-case transition-capacity loss (%)")
        ax.grid(alpha=0.2)

    save_figure(fig, "FigR2_metric_predictive_comparison")


def make_figure_r3() -> None:
    df = pd.read_csv(FIGURE_DATA_DIR / "FigR3_country_rankings.csv")
    data = df[df["sample"] == "extended_nordic_8"].copy()
    rank_cols = [
        ("rank_HHI", "HHI"),
        ("rank_weighted_degree_hhi", "Degree HHI"),
        ("rank_eigenvector_concentration", "Eigenvector"),
        ("rank_betweenness_concentration", "Betweenness"),
        ("rank_chi", r"$\chi$"),
        ("rank_worstcase_loss_pct", "Worst-case loss"),
    ]
    matrix = data[[col for col, _ in rank_cols]].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(rank_cols)))
    ax.set_xticklabels([label for _, label in rank_cols], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data["economy"])
    ax.set_title("Cross-metric vulnerability rankings in the extended sample")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Rank (1 = highest vulnerability)")
    save_figure(fig, "FigR3_cross_metric_country_rankings")


def make_figure_r4() -> None:
    df = pd.read_csv(FIGURE_DATA_DIR / "FigR4_shock_intensity_sensitivity.csv")
    data = df[df["sample"] == "extended_nordic_8"]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for scenario, group in data.groupby("scenario"):
        group = group.sort_values("intensity")
        ax.plot(
            100.0 * group["intensity"],
            group["loss_mean"],
            marker="o",
            linewidth=1.7,
            label=scenario,
        )
        ax.fill_between(
            100.0 * group["intensity"].to_numpy(float),
            (group["loss_mean"] - group["loss_sd"]).to_numpy(float),
            (group["loss_mean"] + group["loss_sd"]).to_numpy(float),
            alpha=0.10,
        )
    ax.set_xlabel("Supplier disruption intensity (%)")
    ax.set_ylabel("Mean transition-capacity loss (%)")
    ax.set_title("Shock-intensity sensitivity in the extended Nordic sample")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    save_figure(fig, "FigR4_shock_intensity_sensitivity")


def make_figure_r5() -> None:
    df = pd.read_csv(FIGURE_DATA_DIR / "FigR5_extended_sample_vulnerability.csv")
    data = df.sort_values("worstcase_loss_pct_mean", ascending=False)

    x = np.arange(len(data))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(9.0, 4.9))
    ax2 = ax1.twinx()

    bars = ax1.bar(
        x - width / 2,
        data["worstcase_loss_pct_mean"],
        width,
        yerr=data["worstcase_loss_pct_std"],
        capsize=3,
        label="Worst-case loss",
    )
    line = ax2.plot(
        x + width / 2,
        data["chi_mean"],
        marker="o",
        linewidth=1.7,
        label=r"Spectral vulnerability $\chi$",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(data["economy"], rotation=25, ha="right")
    ax1.set_ylabel("Worst-case transition-capacity loss (%)")
    ax2.set_ylabel(r"Spectral vulnerability $\chi$")
    ax1.set_title("Baseline vulnerability in the extended Nordic sample")
    handles = [bars, line[0]]
    labels = ["Worst-case loss", r"Spectral vulnerability $\chi$"]
    ax1.legend(handles, labels, frameon=False, loc="upper left")
    ax1.grid(axis="y", alpha=0.2)
    save_figure(fig, "FigR5_extended_sample_baseline_vulnerability")


def make_figure_r6() -> None:
    df = pd.read_csv(FIGURE_DATA_DIR / "FigR6_resource_robustness.csv")
    data = df[
        ~df["specification"].isin(
            ["Full specification", "Battery-mineral bundle", "Infrastructure-mineral bundle"]
        )
    ].copy()
    data["resource_omitted"] = data["specification"].str.replace(
        "Without ", "", regex=False
    )
    data = data.set_index("resource_omitted").reindex(RESOURCES).reset_index()

    matrix = data[["chi_change_mean", "loss_change_mean"]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"Change in $\chi$ (%)", "Change in worst-case loss (%)"])
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data["resource_omitted"])
    ax.set_title("Leave-one-resource-out robustness")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Mean percentage change")
    save_figure(fig, "FigR6_leave_one_resource_out_robustness")


def make_all_figures() -> None:
    required = [
        "FigR1_R2_metric_comparison.csv",
        "FigR3_country_rankings.csv",
        "FigR4_shock_intensity_sensitivity.csv",
        "FigR5_extended_sample_vulnerability.csv",
        "FigR6_resource_robustness.csv",
    ]
    missing = [f for f in required if not (FIGURE_DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot regenerate figures because figure-data CSV files are missing: "
            + ", ".join(missing)
        )
    make_figure_r1()
    make_figure_r2()
    make_figure_r3()
    make_figure_r4()
    make_figure_r5()
    make_figure_r6()
    print(f"[done] Reviewer figures saved in {FIGURE_DIR}")



# =============================================================================
# 7B. Policy-constrained resilience allocation and Pareto frontier
# =============================================================================

RESILIENCE_ALPHA = 0.65
RESILIENCE_POLICY_CAP = 0.40
RESILIENCE_BUDGET_GRID = np.linspace(0.0, 1.0, 51)


def _find_resilience_priority_file(spec: StudySpec) -> Path:
    candidates = [
        spec.output_dir / "Results" / "resilience_priority_seed_level.csv",
        spec.output_dir / "Results" / "raw_resilience_priority_seed_level.csv",
        MAIN_FIGURE_DATA_DIR / spec.name / "raw_resilience_priority_seed_level.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the seed-level resilience-priority CSV. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _find_seed_metrics_file(spec: StudySpec) -> Path:
    candidates = [
        spec.output_dir / "Results" / "seed_level_metrics.csv",
        spec.output_dir / "Results" / "raw_seed_level_metrics.csv",
        MAIN_FIGURE_DATA_DIR / spec.name / "raw_seed_level_metrics.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the seed-level metrics CSV. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _capped_concave_allocation(
    marginal_benefits: np.ndarray,
    budget: float,
    alpha: float = RESILIENCE_ALPHA,
    cap: float = RESILIENCE_POLICY_CAP,
) -> Tuple[np.ndarray, float]:
    """Solve max sum_r g_r u_r**alpha subject to sum(u)<=B and 0<=u<=cap.

    The KKT solution is obtained by bisection on the common multiplier. This
    avoids an optional SciPy dependency and is deterministic.
    """
    g = np.maximum(np.asarray(marginal_benefits, dtype=float), 0.0)
    n = len(g)
    B = float(np.clip(budget, 0.0, n * cap))
    if B <= 1e-14 or np.all(g <= 1e-14):
        return np.zeros(n, dtype=float), 0.0

    # For alpha in (0,1), unconstrained u_r(lambda) = (alpha*g_r/lambda)^(1/(1-alpha)).
    power = 1.0 / (1.0 - alpha)

    def allocation(lam: float) -> np.ndarray:
        raw = np.zeros_like(g)
        positive = g > 0
        raw[positive] = (alpha * g[positive] / lam) ** power
        return np.minimum(raw, cap)

    lo, hi = 1e-16, 1.0
    while allocation(hi).sum() > B:
        hi *= 2.0
        if hi > 1e16:
            break

    for _ in range(250):
        mid = 0.5 * (lo + hi)
        if allocation(mid).sum() > B:
            lo = mid
        else:
            hi = mid

    u = allocation(hi)
    # Numerical correction to exhaust the budget whenever feasible.
    residual = B - float(u.sum())
    if residual > 1e-10:
        order = np.argsort(-g)
        for idx in order:
            add = min(residual, cap - u[idx])
            if add > 0:
                u[idx] += add
                residual -= add
            if residual <= 1e-10:
                break

    gain = float(np.sum(g * np.power(u, alpha)))
    return u, gain


def build_resilience_efficiency_frontier(
    spec: StudySpec = BASELINE_SPEC,
    budgets: np.ndarray = RESILIENCE_BUDGET_GRID,
    alpha: float = RESILIENCE_ALPHA,
    cap: float = RESILIENCE_POLICY_CAP,
) -> pd.DataFrame:
    """Compute, export, and plot the resilience-efficiency Pareto frontier."""
    priority_file = _find_resilience_priority_file(spec)
    metrics_file = _find_seed_metrics_file(spec)

    priority = pd.read_csv(priority_file)
    metrics = pd.read_csv(metrics_file)

    required_priority = {"resource", "marginal_benefit_pp"}
    missing = required_priority.difference(priority.columns)
    if missing:
        raise ValueError(
            f"{priority_file} is missing required columns: {sorted(missing)}"
        )
    if "worstcase_loss_pct" not in metrics.columns:
        raise ValueError(
            f"{metrics_file} must contain 'worstcase_loss_pct'."
        )

    benefit = (
        priority.groupby("resource", as_index=True)["marginal_benefit_pp"]
        .mean()
        .reindex(RESOURCES)
    )
    if benefit.isna().any():
        missing_resources = benefit[benefit.isna()].index.tolist()
        raise ValueError(
            "Missing marginal benefits for resources: " + ", ".join(missing_resources)
        )

    baseline_loss = float(metrics["worstcase_loss_pct"].mean())
    g = benefit.to_numpy(float)

    frontier_rows = []
    allocation_rows = []
    for budget in np.asarray(budgets, dtype=float):
        allocation, total_gain = _capped_concave_allocation(
            g, budget=float(budget), alpha=alpha, cap=cap
        )
        optimized_loss = baseline_loss - total_gain
        relative_reduction = (
            100.0 * total_gain / baseline_loss if baseline_loss > 1e-12 else np.nan
        )
        frontier_rows.append(
            {
                "sample": spec.name,
                "budget": float(budget),
                "baseline_worstcase_loss_pct": baseline_loss,
                "optimized_worstcase_loss_pct": optimized_loss,
                "total_resilience_gain_pp": total_gain,
                "relative_loss_reduction_pct": relative_reduction,
                "alpha": alpha,
                "policy_cap": cap,
            }
        )
        for resource, value in zip(RESOURCES, allocation):
            allocation_rows.append(
                {
                    "sample": spec.name,
                    "budget": float(budget),
                    "resource": resource,
                    "marginal_benefit_pp": float(benefit.loc[resource]),
                    "optimal_allocation": float(value),
                    "allocation_share_pct": 100.0 * float(value),
                    "alpha": alpha,
                    "policy_cap": cap,
                }
            )

    frontier = pd.DataFrame(frontier_rows)
    allocations = pd.DataFrame(allocation_rows)

    output_data_dir = MAIN_FIGURE_DATA_DIR / spec.name
    output_data_dir.mkdir(parents=True, exist_ok=True)
    frontier_file = output_data_dir / "F15_resilience_efficiency_frontier.csv"
    allocation_file = output_data_dir / "F15_resilience_efficiency_allocations.csv"
    save_dataframe(frontier, frontier_file)
    save_dataframe(allocations, allocation_file)

    # Compact six-point table used in the manuscript.
    table_budgets = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    selected = []
    for target in table_budgets:
        idx = (frontier["budget"] - target).abs().idxmin()
        selected.append(frontier.loc[idx])
    table = pd.DataFrame(selected).drop_duplicates(subset=["budget"])
    save_dataframe(
        table,
        APPENDIX_TABLES_DIR / "Table_resilience_efficiency_frontier.csv",
    )

    # Publication figure from the exported CSV.
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(
        frontier["budget"],
        frontier["optimized_worstcase_loss_pct"],
        linewidth=2.2,
    )
    ax.scatter(
        [frontier.iloc[0]["budget"], frontier.iloc[-1]["budget"]],
        [
            frontier.iloc[0]["optimized_worstcase_loss_pct"],
            frontier.iloc[-1]["optimized_worstcase_loss_pct"],
        ],
        s=42,
        zorder=3,
    )
    ax.annotate(
        "No-intervention\nbaseline",
        xy=(frontier.iloc[0]["budget"], frontier.iloc[0]["optimized_worstcase_loss_pct"]),
        xytext=(10, -2),
        textcoords="offset points",
        ha="left",
        va="center",
        fontweight="semibold",
    )
    ax.annotate(
        "Maximum-budget\nsaturation",
        xy=(frontier.iloc[-1]["budget"], frontier.iloc[-1]["optimized_worstcase_loss_pct"]),
        xytext=(-8, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontweight="semibold",
    )
    ax.set_xlabel(r"Resilience intervention budget ($B$)")
    ax.set_ylabel(r"Optimized worst-case transition-capacity loss (\%$_{\mathrm{extloss}}$)")
    ax.set_title("The Resilience-Efficiency Pareto Frontier", fontweight="semibold")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.7)
    fig.tight_layout()

    MAIN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        MAIN_FIGURES_DIR / "F15_resilience_efficiency_pareto_frontier.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        MAIN_FIGURES_DIR / "F15_resilience_efficiency_pareto_frontier.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"[done] Pareto frontier data saved: {frontier_file}")
    return frontier


def regenerate_resilience_frontier_from_csv(
    spec: StudySpec = BASELINE_SPEC,
) -> None:
    csv_file = MAIN_FIGURE_DATA_DIR / spec.name / "F15_resilience_efficiency_frontier.csv"
    if not csv_file.exists():
        return
    frontier = pd.read_csv(csv_file)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(frontier["budget"], frontier["optimized_worstcase_loss_pct"], linewidth=2.2)
    ax.scatter(
        [frontier.iloc[0]["budget"], frontier.iloc[-1]["budget"]],
        [frontier.iloc[0]["optimized_worstcase_loss_pct"], frontier.iloc[-1]["optimized_worstcase_loss_pct"]],
        s=42,
        zorder=3,
    )
    ax.set_xlabel(r"Resilience intervention budget ($B$)")
    ax.set_ylabel(r"Optimized worst-case transition-capacity loss (\%$_{\mathrm{extloss}}$)")
    ax.set_title("The Resilience-Efficiency Pareto Frontier", fontweight="semibold")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(MAIN_FIGURES_DIR / "F15_resilience_efficiency_pareto_frontier.png", dpi=600, bbox_inches="tight")
    fig.savefig(MAIN_FIGURES_DIR / "F15_resilience_efficiency_pareto_frontier.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 8. Complete study runner
# =============================================================================

def run_study(
    v8,
    spec: StudySpec,
    nordic_file: Path,
    allow_calibrated_nordic: bool,
    seeds: Sequence[int],
    rebuild_matrices: bool,
    force_compute: bool,
) -> Dict[str, pd.DataFrame]:
    print(f"[info] Running study: {spec.name}")
    paths, A_RC, A_TR, X, x_source = prepare_study_matrices(
        v8=v8,
        spec=spec,
        nordic_file=nordic_file,
        allow_calibrated_nordic=allow_calibrated_nordic,
        rebuild=rebuild_matrices,
    )

    # Integrated V8 core.  Verify all filenames required by V8 before the
    # expensive Monte Carlo loop starts, so compatibility failures are caught
    # immediately rather than after the computations.
    required_v8_inputs = [
        paths.data / "A_RC_supplier_resource.csv",
        paths.data / "A_TR_resource_technology.csv",
        paths.data / "X_ET_economy_technology.csv",
        paths.data / "matrix_metadata.csv",
    ]
    missing_v8_inputs = [p for p in required_v8_inputs if not p.exists()]
    if missing_v8_inputs:
        raise FileNotFoundError(
            "V8 compatibility inputs are missing: "
            + ", ".join(str(p) for p in missing_v8_inputs)
        )

    v8.run_experiments(paths, list(seeds), force_compute=force_compute)
    v8.make_figures(paths)

    # Reviewer extensions.
    metrics = run_metric_panel(
        v8, spec, paths, A_RC, A_TR, X, seeds, force_compute
    )
    intensity = run_shock_intensity_sensitivity(
        v8, spec, A_RC, A_TR, X, seeds, force_compute
    )
    resource = run_resource_robustness(
        v8, spec, A_RC, A_TR, X, seeds, force_compute
    )

    performance = univariate_performance(metrics)
    incremental = incremental_r2(metrics)
    resource_summary = summarize_resource_robustness(resource)

    save_dataframe(
        performance,
        spec.output_dir / "Tables" / "metric_comparison_performance.csv",
    )
    save_dataframe(
        incremental,
        spec.output_dir / "Tables" / "metric_incremental_r2.csv",
    )
    save_dataframe(
        resource_summary,
        spec.output_dir / "Tables" / "resource_robustness_summary.csv",
    )

    print(f"[done] {spec.name} completed in {spec.output_dir}")
    return {
        "metrics": metrics,
        "intensity": intensity,
        "resource": resource,
        "performance": performance,
        "incremental": incremental,
    }


def write_comparative_tables(
    baseline: Dict[str, pd.DataFrame],
    extended: Dict[str, pd.DataFrame],
) -> None:
    perf = pd.concat(
        [baseline["performance"], extended["performance"]],
        ignore_index=True,
    )
    incr = pd.concat(
        [baseline["incremental"], extended["incremental"]],
        ignore_index=True,
    )
    save_dataframe(perf, TABLE_DIR / "Table_R1_metric_comparison.csv")
    save_dataframe(incr, TABLE_DIR / "Table_R2_incremental_explanatory_power.csv")

    intensity_summary = (
        pd.concat([baseline["intensity"], extended["intensity"]], ignore_index=True)
        .groupby(["sample", "intensity", "scenario"], as_index=False)
        .agg(
            loss_mean=("loss_pct", "mean"),
            loss_sd=("loss_pct", "std"),
        )
    )
    save_dataframe(
        intensity_summary,
        TABLE_DIR / "Table_R3_shock_intensity_sensitivity.csv",
    )

    resource_summary = summarize_resource_robustness(extended["resource"])
    save_dataframe(
        resource_summary,
        TABLE_DIR / "Table_R4_resource_robustness.csv",
    )


def write_manifest(
    v8_script: Path,
    nordic_file: Path,
    n_seeds: int,
    allow_calibrated_nordic: bool,
    studies: Sequence[str],
) -> None:
    input_files = [
        BASE_DATA_DIR / "A_RC_supplier_resource.csv",
        BASE_DATA_DIR / "A_TR_resource_technology.csv",
        BASE_DATA_DIR / "X_ET_economy_technology.csv",
        v8_script,
    ]
    if nordic_file.exists():
        input_files.append(nordic_file)

    manifest = {
        "pipeline": "geo_transition_network_v10.py",
        "integrates": str(v8_script),
        "studies": list(studies),
        "baseline_economies": BASE_ECONOMIES,
        "extended_economies": EXTENDED_ECONOMIES,
        "resources": RESOURCES,
        "technologies": TECHNOLOGIES,
        "shock_intensities": SHOCK_INTENSITIES,
        "n_seeds": n_seeds,
        "seed_sequence": list(range(DEFAULT_SEED_START, DEFAULT_SEED_START + n_seeds)),
        "allow_calibrated_nordic": allow_calibrated_nordic,
        "nordic_file": str(nordic_file),
        "network_metrics": METRIC_COLUMNS,
        "input_sha256": {
            str(path): sha256_file(path) for path in input_files if path.exists()
        },
        "notes": [
            "Original Results_Geo is never overwritten.",
            "Every reviewer figure is generated from a dedicated CSV file.",
            "The Nordic calibrated fallback is for preliminary testing only.",
            "Use harmonised observed Nordic portfolio data for the final revision.",
        ],
    }
    (REVIEW_RESULTS_DIR / "run_manifest_v10.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )



# =============================================================================
# 9. Final manuscript package and complete figure-data archive
# =============================================================================

IMAGE_EXTENSIONS = {".png", ".pdf", ".svg"}
TABLE_EXTENSIONS = {".csv", ".tex", ".xlsx"}

# Maximums requested for the revised paper. The package may contain fewer items
# when an optional V8 output is unavailable, but never more.
MAX_MAIN_FIGURES = 15
MAX_MAIN_TABLES = 7


@dataclass(frozen=True)
class MainFigureSpec:
    number: int
    title: str
    section: str
    source_group: str
    keywords: Tuple[str, ...]
    preferred_stems: Tuple[str, ...] = ()
    data_candidates: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MainTableSpec:
    number: int
    title: str
    section: str
    source_group: str
    candidates: Tuple[str, ...]


# Editorial ordering used in the revised manuscript.
MAIN_FIGURE_SPECS: Tuple[MainFigureSpec, ...] = (
    MainFigureSpec(1, "Supplier-resource concentration matrix", "Data and network construction",
                   "baseline", ("supplier", "resource", "matrix"), ("a_rc",),
                   ("A_RC_supplier_resource.csv",)),
    MainFigureSpec(2, "Technology-resource intensity matrix", "Data and network construction",
                   "baseline", ("technology", "resource", "intensity"), ("a_tr",),
                   ("A_TR_resource_technology.csv",)),
    MainFigureSpec(3, "European transition-technology portfolios", "Data and network construction",
                   "baseline", ("transition", "technology", "portfolio"), ("x_et", "portfolio"),
                   ("X_ET_economy_technology.csv",)),
    MainFigureSpec(4, "Hidden upstream concentration at matched technology concentration",
                   "Hidden vulnerability", "baseline",
                   ("hidden", "upstream", "concentration"), ("matched", "hidden"),
                   ("matched_hhi_hidden_concentration.csv",)),
    MainFigureSpec(5, "Baseline structural vulnerability in the original sample",
                   "Hidden vulnerability", "baseline",
                   ("baseline", "structural", "vulnerability"), ("baseline",),
                   ("seed_level_metrics.csv",)),
    MainFigureSpec(6, "Transition-capacity losses under geopolitical shocks",
                   "Counterfactual shocks", "baseline",
                   ("counterfactual", "geopolitical", "shock"), ("scenario", "counterfactual"),
                   ("scenario_losses_seed_level.csv", "Table2_counterfactual_losses.csv")),
    MainFigureSpec(7, "Resource-channel contributions to transition vulnerability",
                   "Policy implications", "baseline",
                   ("resource", "channel", "contribution"), ("resource_contribution",),
                   ("resource_contribution_seed_level.csv", "Table_resource_contribution.csv")),
    MainFigureSpec(8, "Resilience-priority ranking by critical resource",
                   "Policy implications", "baseline",
                   ("resilience", "priority", "ranking"), ("resilience_priority",),
                   ("resilience_priority_seed_level.csv", "Table3_resilience_priority.csv")),
    MainFigureSpec(9, "HHI and spectral concentration in the original and extended samples",
                   "Metric validation", "reviewer",
                   ("hhi", "spectral", "concentration"), ("FigR1_HHI_vs_spectral_concentration",),
                   ("FigR1_R2_metric_comparison.csv",)),
    MainFigureSpec(10, "Predictive comparison of conventional and network metrics",
                   "Metric validation", "reviewer",
                   ("metric", "predictive", "comparison"), ("FigR2_metric_predictive_comparison",),
                   ("FigR1_R2_metric_comparison.csv",)),
    MainFigureSpec(11, "Shock-intensity sensitivity in the extended Nordic sample",
                   "Robustness", "reviewer",
                   ("shock", "intensity", "sensitivity"), ("FigR4_shock_intensity_sensitivity",),
                   ("FigR4_shock_intensity_sensitivity.csv",)),
    MainFigureSpec(12, "Baseline vulnerability in the extended Nordic sample",
                   "Robustness", "reviewer",
                   ("extended", "sample", "baseline", "vulnerability"),
                   ("FigR5_extended_sample_baseline_vulnerability",),
                   ("FigR5_extended_sample_vulnerability.csv",)),
    MainFigureSpec(13, "Cross-metric vulnerability rankings in the extended sample",
                   "Robustness", "reviewer",
                   ("cross", "metric", "rank"), ("FigR3_cross_metric_country_rankings",),
                   ("FigR3_country_rankings.csv",)),
    MainFigureSpec(14, "Leave-one-resource-out robustness",
                   "Robustness", "reviewer",
                   ("leave", "one", "resource", "robustness"),
                   ("FigR6_leave_one_resource_out_robustness",),
                   ("FigR6_resource_robustness.csv",)),
)

MAIN_TABLE_SPECS: Tuple[MainTableSpec, ...] = (
    MainTableSpec(1, "Baseline structural vulnerability across European economies",
                  "Hidden vulnerability", "baseline",
                  ("Table1_baseline_metrics", "baseline_metrics")),
    MainTableSpec(2, "Transition-capacity losses under counterfactual geopolitical shocks",
                  "Counterfactual shocks", "baseline",
                  ("Table2_counterfactual_losses", "counterfactual_losses")),
    MainTableSpec(3, "Comparative performance of conventional and network vulnerability metrics",
                  "Metric validation", "comparative",
                  ("Table_R1_metric_comparison", "metric_comparison_performance")),
    MainTableSpec(4, "Incremental explanatory power beyond technology HHI",
                  "Metric validation", "comparative",
                  ("Table_R2_incremental_explanatory_power", "metric_incremental_r2")),
    MainTableSpec(5, "Shock-intensity sensitivity",
                  "Robustness", "comparative",
                  ("Table_R3_shock_intensity_sensitivity",)),
    MainTableSpec(6, "Resource-subset and leave-one-resource-out robustness",
                  "Robustness", "comparative",
                  ("Table_R4_resource_robustness", "resource_robustness_summary")),
    MainTableSpec(7, "Resource-level resilience priority ranking",
                  "Policy implications", "baseline",
                  ("Table3_resilience_priority", "resilience_priority")),
)


def _slug(text_value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", text_value).strip("_")
    return value[:100]


def _all_files(root: Path, extensions: Optional[set] = None) -> List[Path]:
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file()]
    if extensions is not None:
        files = [p for p in files if p.suffix.lower() in extensions]
    return sorted(files)


def _normalised_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", " ", path.stem.lower()).strip()


def _score_file(path: Path, keywords: Sequence[str], preferred_stems: Sequence[str]) -> int:
    name = _normalised_name(path)
    stem_compact = re.sub(r"[^a-z0-9]+", "", path.stem.lower())
    score = 0
    for preferred in preferred_stems:
        compact = re.sub(r"[^a-z0-9]+", "", preferred.lower())
        if compact and compact in stem_compact:
            score += 50
    for keyword in keywords:
        words = re.sub(r"[^a-z0-9]+", " ", keyword.lower()).split()
        if all(word in name for word in words):
            score += 8 + len(words)
    if path.suffix.lower() == ".pdf":
        score += 2
    if path.suffix.lower() == ".png":
        score += 1
    return score


def _source_roots(group: str) -> List[Path]:
    if group == "baseline":
        return [BASELINE_DIR / "Figures", BASELINE_DIR / "Tables",
                BASELINE_DIR / "Results", BASELINE_DIR / "Data"]
    if group == "extended":
        return [EXTENDED_DIR / "Figures", EXTENDED_DIR / "Tables",
                EXTENDED_DIR / "Results", EXTENDED_DIR / "Data"]
    if group == "reviewer":
        return [FIGURE_DIR, FIGURE_DATA_DIR]
    if group == "comparative":
        return [TABLE_DIR, COMPARATIVE_DIR]
    return [REVIEW_RESULTS_DIR]


def _best_matching_file(
    roots: Sequence[Path],
    extensions: set,
    keywords: Sequence[str],
    preferred_stems: Sequence[str] = (),
) -> Optional[Path]:
    candidates: List[Tuple[int, Path]] = []
    for root in roots:
        for path in _all_files(root, extensions):
            score = _score_file(path, keywords, preferred_stems)
            if score > 0:
                candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].suffix.lower() == ".pdf"), reverse=True)
    return candidates[0][1]


def _find_named_file(roots: Sequence[Path], candidate_names: Sequence[str],
                     extensions: Optional[set] = None) -> Optional[Path]:
    files: List[Path] = []
    for root in roots:
        files.extend(_all_files(root, extensions))
    lowered = [(p, p.stem.lower(), p.name.lower()) for p in files]
    for candidate in candidate_names:
        c = Path(candidate).stem.lower()
        for path, stem, name in lowered:
            if stem == c or c in stem or c in name:
                return path
    return None


def _copy_with_companions(source: Path, destination_stem: Path) -> List[Path]:
    copied: List[Path] = []
    for extension in [".png", ".pdf", ".svg"]:
        candidate = source.with_suffix(extension)
        if candidate.exists():
            destination = destination_stem.with_suffix(extension)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            copied.append(destination)
    if not copied and source.exists():
        destination = destination_stem.with_suffix(source.suffix.lower())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _matrix_to_long_csv(matrix_file: Path, output_file: Path,
                        row_name: str, column_name: str, value_name: str) -> Path:
    df = pd.read_csv(matrix_file, index_col=0)
    long = (
        df.rename_axis(row_name)
        .reset_index()
        .melt(id_vars=row_name, var_name=column_name, value_name=value_name)
    )
    save_dataframe(long, output_file)
    return output_file


def _derive_v8_figure_data(spec: StudySpec) -> Dict[str, Path]:
    """Create stable, publication-facing CSVs for all core V8 figures.

    V8 already stores seed-level results. V10 converts those files into compact
    figure-level datasets, while preserving the raw seed-level CSVs separately.
    """
    data_dir = spec.output_dir / "Data"
    results_dir = spec.output_dir / "Results"
    output_dir = (MAIN_FIGURE_DATA_DIR / spec.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    derived: Dict[str, Path] = {}

    arc = data_dir / "A_RC_supplier_resource.csv"
    atr = data_dir / "A_TR_resource_technology.csv"
    xet = data_dir / "X_ET_economy_technology.csv"
    if arc.exists():
        derived["supplier_resource"] = _matrix_to_long_csv(
            arc, output_dir / "F01_supplier_resource_matrix.csv",
            "resource", "supplier", "production_share"
        )
    if atr.exists():
        derived["technology_resource"] = _matrix_to_long_csv(
            atr, output_dir / "F02_technology_resource_matrix.csv",
            "technology", "resource", "material_intensity"
        )
    if xet.exists():
        derived["portfolio"] = _matrix_to_long_csv(
            xet, output_dir / "F03_transition_technology_portfolios.csv",
            "economy", "technology", "portfolio_weight"
        )

    # Candidate raw files used by the validated V8 script.
    aliases = {
        "hidden": ("matched_hhi_hidden_concentration", "matched_hhi", "hidden_concentration"),
        "metrics": ("seed_level_metrics", "baseline_metrics_seed", "baseline_metrics"),
        "scenario": ("scenario_losses_seed_level", "counterfactual_losses", "scenario_losses"),
        "resource_contribution": ("resource_contribution_seed_level", "resource_contribution"),
        "resilience_priority": ("resilience_priority_seed_level", "resilience_priority"),
        "resource_sensitivity": ("sensitivity_resource_intensity_seed_level",
                                 "sensitivity_resource_intensity"),
        "shock_composition": ("worstcase_shock_composition_seed_level",
                              "shock_composition_seed_level"),
    }
    raw_found: Dict[str, Path] = {}
    for key, names in aliases.items():
        found = _find_named_file([results_dir, spec.output_dir / "Tables"], names, {".csv"})
        if found:
            raw_found[key] = found
            shutil.copy2(found, output_dir / f"raw_{found.name}")

    if "hidden" in raw_found:
        out = output_dir / "F04_hidden_upstream_concentration.csv"
        shutil.copy2(raw_found["hidden"], out)
        derived["hidden"] = out

    if "metrics" in raw_found:
        df = pd.read_csv(raw_found["metrics"])
        group_cols = [c for c in ["economy"] if c in df.columns]
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != "seed"]
        if group_cols and numeric:
            summary = df.groupby(group_cols)[numeric].agg(["mean", "std"]).reset_index()
            summary.columns = [
                "_".join([str(v) for v in col if str(v)]) if isinstance(col, tuple) else str(col)
                for col in summary.columns
            ]
            out = output_dir / "F05_baseline_structural_vulnerability.csv"
            save_dataframe(summary, out)
            derived["metrics"] = out

    if "scenario" in raw_found:
        df = pd.read_csv(raw_found["scenario"])
        group_cols = [c for c in ["scenario", "economy"] if c in df.columns]
        value = next((c for c in ["loss_pct", "loss", "transition_loss_pct"] if c in df.columns), None)
        if len(group_cols) == 2 and value:
            summary = df.groupby(group_cols, as_index=False)[value].agg(["mean", "std"]).reset_index()
            summary.columns = [c if isinstance(c, str) else "_".join(filter(None, c)) for c in summary.columns]
            out = output_dir / "F06_counterfactual_shock_losses.csv"
            save_dataframe(summary, out)
            derived["scenario"] = out

    for key, number, stem in [
        ("resource_contribution", 7, "resource_channel_contribution"),
        ("resilience_priority", 8, "resilience_priority"),
    ]:
        if key in raw_found:
            df = pd.read_csv(raw_found[key])
            resource_col = next((c for c in ["resource", "Resource"] if c in df.columns), None)
            numeric_candidates = [c for c in df.select_dtypes(include=[np.number]).columns if c != "seed"]
            if resource_col and numeric_candidates:
                value = numeric_candidates[-1]
                summary = df.groupby(resource_col, as_index=False)[value].agg(["mean", "std"]).reset_index()
                summary.columns = [c if isinstance(c, str) else "_".join(filter(None, c)) for c in summary.columns]
                out = output_dir / f"F{number:02d}_{stem}.csv"
                save_dataframe(summary, out)
            else:
                out = output_dir / f"F{number:02d}_{stem}.csv"
                shutil.copy2(raw_found[key], out)
            derived[key] = out

    # Complete machine-readable archive: every V8 CSV, not only the selected figures.
    archive_dir = APPENDIX_FIGURE_DATA_DIR / spec.name / "All_V8_CSV"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for csv_file in _all_files(spec.output_dir, {".csv"}):
        relative = csv_file.relative_to(spec.output_dir)
        destination = archive_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_file, destination)

    return derived


def _copy_nordic_figure_data() -> None:
    """Copy every Nordic/reviewer figure dataset into a stable archive."""
    target = MAIN_FIGURE_DATA_DIR / "extended_nordic_8"
    target.mkdir(parents=True, exist_ok=True)
    for csv_file in _all_files(FIGURE_DATA_DIR, {".csv"}):
        shutil.copy2(csv_file, target / csv_file.name)

    raw_target = APPENDIX_FIGURE_DATA_DIR / "extended_nordic_8" / "All_Extended_CSV"
    raw_target.mkdir(parents=True, exist_ok=True)
    for csv_file in _all_files(EXTENDED_DIR, {".csv"}):
        relative = csv_file.relative_to(EXTENDED_DIR)
        destination = raw_target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_file, destination)


def build_final_manuscript_package() -> None:
    """Select, number, and package the publication-facing results.

    The function is deliberately non-destructive: it copies files into the final
    package and moves no raw output. A manifest records the selected source file,
    the corresponding figure-data CSV, and any missing optional result.
    """
    for directory in [
        MAIN_FIGURES_DIR, MAIN_FIGURE_DATA_DIR, MAIN_TABLES_DIR,
        APPENDIX_FIGURES_DIR, APPENDIX_FIGURE_DATA_DIR, APPENDIX_TABLES_DIR,
        REPRO_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # Rebuild curated folders to prevent stale manuscript outputs.
    for directory in [MAIN_FIGURES_DIR, MAIN_TABLES_DIR]:
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()

    baseline_data = _derive_v8_figure_data(BASELINE_SPEC)
    _derive_v8_figure_data(EXTENDED_SPEC)
    _copy_nordic_figure_data()

    figure_manifest: List[Dict[str, object]] = []
    selected_sources: set = set()

    for spec in MAIN_FIGURE_SPECS[:MAX_MAIN_FIGURES]:
        roots = _source_roots(spec.source_group)
        source = _best_matching_file(
            roots, IMAGE_EXTENSIONS, spec.keywords, spec.preferred_stems
        )
        destination_stem = MAIN_FIGURES_DIR / (
            f"F{spec.number:02d}_{_slug(spec.title)}"
        )
        copied = _copy_with_companions(source, destination_stem) if source else []
        if source:
            selected_sources.add(source.resolve())

        data_source = _find_named_file(
            roots + [FIGURE_DATA_DIR, MAIN_FIGURE_DATA_DIR],
            spec.data_candidates,
            {".csv"},
        )
        # Prefer V10-derived baseline figure data when available.
        if spec.number <= 8:
            derived_key = {
                1: "supplier_resource", 2: "technology_resource", 3: "portfolio",
                4: "hidden", 5: "metrics", 6: "scenario",
                7: "resource_contribution", 8: "resilience_priority",
            }.get(spec.number)
            if derived_key and derived_key in baseline_data:
                data_source = baseline_data[derived_key]

        packaged_data = None
        if data_source and data_source.exists():
            packaged_data = MAIN_FIGURE_DATA_DIR / (
                f"F{spec.number:02d}_{_slug(spec.title)}.csv"
            )
            shutil.copy2(data_source, packaged_data)

        figure_manifest.append(
            {
                "figure_number": spec.number,
                "title": spec.title,
                "section": spec.section,
                "source_group": spec.source_group,
                "source_figure": str(source) if source else "",
                "packaged_files": "|".join(str(p) for p in copied),
                "source_data": str(data_source) if data_source else "",
                "packaged_data": str(packaged_data) if packaged_data else "",
                "status": "ready" if copied and packaged_data else (
                    "figure_only" if copied else "missing_figure"
                ),
            }
        )

    # All unselected figures are retained in the appendix.
    figure_roots = [
        BASELINE_DIR / "Figures", EXTENDED_DIR / "Figures", FIGURE_DIR
    ]
    for root in figure_roots:
        for image in _all_files(root, IMAGE_EXTENSIONS):
            if image.resolve() in selected_sources:
                continue
            relative_group = root.parent.name if root.parent else "Other"
            destination = APPENDIX_FIGURES_DIR / relative_group / image.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination)

    pd.DataFrame(figure_manifest).to_csv(
        MANUSCRIPT_DIR / "main_figures_manifest.csv", index=False
    )

    table_manifest: List[Dict[str, object]] = []
    selected_tables: set = set()
    for spec in MAIN_TABLE_SPECS[:MAX_MAIN_TABLES]:
        roots = _source_roots(spec.source_group)
        source = _find_named_file(roots, spec.candidates, TABLE_EXTENSIONS)
        copied_files: List[str] = []
        if source:
            # Copy all available companion formats with the same stem.
            for extension in [".csv", ".tex", ".xlsx"]:
                companion = source.with_suffix(extension)
                if companion.exists():
                    destination = MAIN_TABLES_DIR / (
                        f"T{spec.number:02d}_{_slug(spec.title)}{extension}"
                    )
                    shutil.copy2(companion, destination)
                    copied_files.append(str(destination))
                    selected_tables.add(companion.resolve())
        table_manifest.append(
            {
                "table_number": spec.number,
                "title": spec.title,
                "section": spec.section,
                "source_group": spec.source_group,
                "source_table": str(source) if source else "",
                "packaged_files": "|".join(copied_files),
                "status": "ready" if copied_files else "missing",
            }
        )

    # Archive all non-selected tables.
    for root in [
        BASELINE_DIR / "Tables", EXTENDED_DIR / "Tables", TABLE_DIR
    ]:
        for table in _all_files(root, TABLE_EXTENSIONS):
            if table.resolve() in selected_tables:
                continue
            destination = APPENDIX_TABLES_DIR / root.parent.name / table.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(table, destination)

    pd.DataFrame(table_manifest).to_csv(
        MANUSCRIPT_DIR / "main_tables_manifest.csv", index=False
    )

    # Publication map, README, and inventory.
    publication_map = {
        "main_figures_maximum": MAX_MAIN_FIGURES,
        "main_figures_selected": len(MAIN_FIGURE_SPECS),
        "main_tables_maximum": MAX_MAIN_TABLES,
        "main_tables_selected": len(MAIN_TABLE_SPECS),
        "main_figure_order": [
            {"number": s.number, "title": s.title, "section": s.section}
            for s in MAIN_FIGURE_SPECS
        ],
        "main_table_order": [
            {"number": s.number, "title": s.title, "section": s.section}
            for s in MAIN_TABLE_SPECS
        ],
        "editorial_note": (
            "Figures 1-8 reproduce the scientific core of V8. Figures 9-14 "
            "contain the reviewer-requested metric, Nordic, shock-intensity, "
            "and resource-robustness extensions. All remaining outputs are "
            "retained in the appendix folders."
        ),
    }
    (MANUSCRIPT_DIR / "publication_map.json").write_text(
        json.dumps(publication_map, indent=2), encoding="utf-8"
    )

    readme = """# V10 publication package

## Main manuscript
- `Main_Figures`: numbered PNG/PDF/SVG files selected for the revised paper.
- `Main_Figure_Data`: one CSV per selected figure, including the V8 core and
  the Nordic/reviewer extensions.
- `Main_Tables`: at most seven numbered tables.
- `main_figures_manifest.csv` and `main_tables_manifest.csv`: source-to-output audit trail.

## Appendix
All non-selected figures, tables, and raw figure CSVs are retained under
`04_Appendix`; nothing is deleted from the numerical study folders.

## Rebuilding figures without recomputation
Run:
`python geo_transition_network_v10.py --figures-only --package-only`

The reviewer figures are recreated from `Figure_Data`. Core V8 figure datasets
are preserved in `00_Manuscript/Main_Figure_Data` and can be used by any
standalone plotting script without rerunning the Monte Carlo experiments.
"""
    (MANUSCRIPT_DIR / "README.md").write_text(readme, encoding="utf-8")

    inventory_rows = []
    for path in _all_files(REVIEW_RESULTS_DIR):
        inventory_rows.append(
            {
                "relative_path": str(path.relative_to(REVIEW_RESULTS_DIR)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(inventory_rows).to_csv(
        REPRO_DIR / "complete_output_inventory.csv", index=False
    )

    print(f"[done] Main manuscript package: {MANUSCRIPT_DIR}")
    print(f"[done] Main figures selected: {len(MAIN_FIGURE_SPECS)} / {MAX_MAIN_FIGURES}")
    print(f"[done] Main tables selected: {len(MAIN_TABLE_SPECS)} / {MAX_MAIN_TABLES}")
    print(f"[done] Remaining outputs archived in: {APPENDIX_DIR}")


# =============================================================================
# 10. Command-line interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V10 final revision pipeline for geopolitical transition vulnerability."
    )
    parser.add_argument(
        "--v8-script",
        type=Path,
        default=Path("geo_transition_network_v8.py"),
        help="Path to the validated V8 script.",
    )
    parser.add_argument(
        "--study",
        choices=["baseline", "extended", "both"],
        default="both",
        help="Study sample to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of deterministic robustness seeds.",
    )
    parser.add_argument(
        "--nordic-file",
        type=Path,
        default=REVIEW_DATA_DIR / "nordic_technology_portfolios.csv",
        help="Observed/harmonised Nordic technology portfolio file.",
    )
    parser.add_argument(
        "--allow-calibrated-nordic",
        action="store_true",
        help="Allow explicitly flagged calibrated Nordic fallback rows.",
    )
    parser.add_argument(
        "--rebuild-matrices",
        action="store_true",
        help="Recreate study-specific matrix directories.",
    )
    parser.add_argument(
        "--force-compute",
        action="store_true",
        help="Recompute all results even when cached CSV files exist.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate reviewer figures from Figure_Data CSV files only.",
    )
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="Build the final manuscript/appendix package from existing outputs only.",
    )
    parser.add_argument(
        "--create-nordic-template",
        action="store_true",
        help="Create a long-format Nordic data template and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.create_nordic_template:
        create_nordic_template(args.nordic_file)
        return

    if args.figures_only:
        make_all_figures()
        regenerate_resilience_frontier_from_csv(BASELINE_SPEC)
        build_final_manuscript_package()
        return

    if args.package_only:
        build_final_manuscript_package()
        return

    if args.seeds < 5:
        raise ValueError("--seeds must be at least 5.")

    v8 = import_v8(args.v8_script)
    seeds = list(range(DEFAULT_SEED_START, DEFAULT_SEED_START + args.seeds))

    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    studies_run: List[str] = []

    if args.study in {"baseline", "both"}:
        results["baseline"] = run_study(
            v8=v8,
            spec=BASELINE_SPEC,
            nordic_file=args.nordic_file,
            allow_calibrated_nordic=args.allow_calibrated_nordic,
            seeds=seeds,
            rebuild_matrices=args.rebuild_matrices,
            force_compute=args.force_compute,
        )
        studies_run.append("baseline_5")

    if args.study in {"extended", "both"}:
        results["extended"] = run_study(
            v8=v8,
            spec=EXTENDED_SPEC,
            nordic_file=args.nordic_file,
            allow_calibrated_nordic=args.allow_calibrated_nordic,
            seeds=seeds,
            rebuild_matrices=args.rebuild_matrices,
            force_compute=args.force_compute,
        )
        studies_run.append("extended_nordic_8")

    if "baseline" in results:
        build_resilience_efficiency_frontier(BASELINE_SPEC)

    if args.study == "both":
        write_comparative_tables(results["baseline"], results["extended"])
        prepare_all_figure_data(
            baseline_panel=results["baseline"]["metrics"],
            extended_panel=results["extended"]["metrics"],
            baseline_intensity=results["baseline"]["intensity"],
            extended_intensity=results["extended"]["intensity"],
            extended_resource=results["extended"]["resource"],
        )
        make_all_figures()

    write_manifest(
        v8_script=args.v8_script,
        nordic_file=args.nordic_file,
        n_seeds=args.seeds,
        allow_calibrated_nordic=args.allow_calibrated_nordic,
        studies=studies_run,
    )
    build_final_manuscript_package()

    print("[done] V10 final revision pipeline completed.")
    print(f"[done] Results: {REVIEW_RESULTS_DIR}")
    print(f"[done] Reviewer figures: {FIGURE_DIR}")
    print(f"[done] Figure CSV data: {FIGURE_DATA_DIR}")
    print(f"[done] Comparative tables: {TABLE_DIR}")


if __name__ == "__main__":
    main()
