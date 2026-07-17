#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geopolitical Contagion in the Energy Transition
------------------------------------------------
End-to-end reproducible pipeline:

1. Creates/uses cached data in Data_geo/
2. Builds the multilayer transition network:
   supplier countries -> critical minerals -> clean-energy technologies -> EU economies
3. Computes theoretical quantities:
   K_e, L_e, V_e, Q_e, chi_e, HHI, scenario losses, worst-case shocks
4. Runs robustness over 15 random seeds
5. Saves all numerical results as CSV in Results_Geo/
6. Generates up to 7 publication-ready figures in Results_Geo/Figures/

Important:
- UN Comtrade requires a free API key for full automated downloads.
  Set it as an environment variable before running:
      export COMTRADE_API_KEY="YOUR_KEY"
  or on Windows PowerShell:
      setx COMTRADE_API_KEY "YOUR_KEY"

- IEA Critical Minerals data may require a free IEA account for downloading.
  If you have the file, place it in:
      Data_geo/IEA_CriticalMinerals_TechnologyRequirements.csv
  with columns: technology, resource, intensity

- If a source cannot be downloaded, the script uses transparent fallback matrices
  saved in Data_geo/ with a data_source column. These fallbacks are intended
  for code testing and reproducibility, not as a substitute for final empirical data.

Author: generated for the Energy Policy paper project
"""

from __future__ import annotations

import os
import time
import json
import math
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except Exception:
    NETWORKX_AVAILABLE = False


# =============================================================================
# 0. Project configuration
# =============================================================================

DATA_DIR = Path("Data_geo")
RESULTS_DIR = Path("Results_Geo")
FIG_DIR = RESULTS_DIR / "Figures"
TABLE_DIR = RESULTS_DIR / "Tables"

for d in [DATA_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(1, 16))
DELTA = 0.25
RHO = -2.0
ETA = 0.50
EPS = 1e-9

EU_ECONOMIES = {
    "France": {"iso3": "FRA", "m49": "251"},
    "Germany": {"iso3": "DEU", "m49": "276"},
    "Spain": {"iso3": "ESP", "m49": "724"},
    "Italy": {"iso3": "ITA", "m49": "380"},
    "Netherlands": {"iso3": "NLD", "m49": "528"},
}

SUPPLIERS = {
    "China": {"iso3": "CHN", "m49": "156"},
    "Australia": {"iso3": "AUS", "m49": "036"},
    "Chile": {"iso3": "CHL", "m49": "152"},
    "DR Congo": {"iso3": "COD", "m49": "180"},
    "Indonesia": {"iso3": "IDN", "m49": "360"},
    "Russia": {"iso3": "RUS", "m49": "643"},
    "United States": {"iso3": "USA", "m49": "842"},
    "Morocco": {"iso3": "MAR", "m49": "504"},
    "South Africa": {"iso3": "ZAF", "m49": "710"},
    "Brazil": {"iso3": "BRA", "m49": "076"},
}

RESOURCES = ["Lithium", "Cobalt", "Nickel", "Graphite", "Rare earths", "Copper"]
TECHNOLOGIES = ["EV batteries", "Wind", "Solar PV", "Electrolysers", "Power grids"]
ECONOMIES = list(EU_ECONOMIES.keys())
SUPPLIER_NAMES = list(SUPPLIERS.keys())

# HS code bundles are deliberately transparent and editable.
# They are proxies for traded products related to each resource.
HS_CODES = {
    "Lithium": ["283691", "282520"],
    "Cobalt": ["260500", "810520"],
    "Nickel": ["260400", "750210"],
    "Graphite": ["250410"],
    "Rare earths": ["280530", "284610"],
    "Copper": ["260300", "740311"],
}

SCENARIOS = {
    "China rare-earth/graphite shock": {"China": DELTA},
    "Indonesia nickel shock": {"Indonesia": DELTA},
    "Lithium shock Chile+Australia": {"Chile": DELTA / 2, "Australia": DELTA / 2},
    "Dual shock China+Russia": {"China": DELTA / 2, "Russia": DELTA / 2},
}


# =============================================================================
# 1. Utility functions
# =============================================================================

def save_metadata() -> None:
    metadata = {
        "created_by": "geo_transition_network.py",
        "delta": DELTA,
        "rho": RHO,
        "eta": ETA,
        "seeds": SEEDS,
        "economies": ECONOMIES,
        "resources": RESOURCES,
        "technologies": TECHNOLOGIES,
        "suppliers": SUPPLIER_NAMES,
        "notes": [
            "UN Comtrade downloads require COMTRADE_API_KEY.",
            "IEA Critical Minerals technology-requirement data may need manual placement in Data_geo.",
            "Fallback matrices are saved and flagged when public API downloads are unavailable.",
        ],
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def normalize_rows(df: pd.DataFrame, value_cols: Optional[List[str]] = None) -> pd.DataFrame:
    out = df.copy()
    cols = value_cols if value_cols is not None else out.columns.tolist()
    row_sums = out[cols].sum(axis=1).replace(0, np.nan)
    out[cols] = out[cols].div(row_sums, axis=0).fillna(0.0)
    return out


def dirichlet_perturb_matrix(mat: np.ndarray, seed: int, concentration: float = 200.0) -> np.ndarray:
    """
    Row-wise Dirichlet perturbation preserving row sums at 1.
    Rows with zero mass remain zero.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        p = np.clip(mat[i].astype(float), 0, None)
        s = p.sum()
        if s <= EPS:
            out[i] = p
        else:
            p = p / s
            alpha = np.maximum(p * concentration, 1e-4)
            out[i] = rng.dirichlet(alpha)
    return out


def country_vector_from_scenario(scenario: Dict[str, float]) -> np.ndarray:
    z = np.zeros(len(SUPPLIER_NAMES), dtype=float)
    for name, val in scenario.items():
        if name in SUPPLIER_NAMES:
            z[SUPPLIER_NAMES.index(name)] = float(val)
    return z


def ensure_nonnegative_prob_matrix(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [c for c in out.columns if c != index_col]
    out[numeric_cols] = out[numeric_cols].clip(lower=0).fillna(0.0)
    row_sums = out[numeric_cols].sum(axis=1)
    for idx, s in row_sums.items():
        if s <= EPS:
            out.loc[idx, numeric_cols] = 1.0 / len(numeric_cols)
    out[numeric_cols] = out[numeric_cols].div(out[numeric_cols].sum(axis=1), axis=0)
    return out


# =============================================================================
# 2. Data acquisition and cache
# =============================================================================

def download_wgi(force: bool = False) -> pd.DataFrame:
    """
    Downloads World Bank WGI political stability estimates.
    Public CSV URL from World Bank Data360.
    """
    out_path = DATA_DIR / "WGI_Political_Stability.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    url = "https://data360files.worldbank.org/data360-data/data/WB_WGI/WB_WGI_PV_EST_WIDEF.csv"
    try:
        df = pd.read_csv(url)
        df.to_csv(out_path, index=False)
        return df
    except Exception as exc:
        warnings.warn(f"WGI download failed: {exc}. Using fallback geopolitical risk.")
        fallback = pd.DataFrame({
            "country": SUPPLIER_NAMES,
            "iso3": [SUPPLIERS[c]["iso3"] for c in SUPPLIER_NAMES],
            "political_stability_estimate": [0.0, 1.0, 0.4, -1.5, -0.2, -1.0, 0.6, -0.3, -0.1, -0.2],
            "data_source": "fallback_curated"
        })
        fallback.to_csv(out_path, index=False)
        return fallback


def parse_wgi_risk(wgi_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Robust parser for WGI PV estimate file. If the World Bank schema changes,
    this function falls back to the simple fallback schema.
    """
    if {"country", "iso3", "political_stability_estimate"}.issubset(wgi_raw.columns):
        df = wgi_raw.copy()
        x = df["political_stability_estimate"].astype(float)
        # Convert stability estimate into risk in [0,1]: lower stability -> higher risk
        risk = (x.max() - x) / (x.max() - x.min() + EPS)
        return pd.DataFrame({"supplier": df["country"], "iso3": df["iso3"], "risk": risk, "data_source": df.get("data_source", "WGI")})

    # Try to infer columns from Data360 wide file.
    cols_lower = {c.lower(): c for c in wgi_raw.columns}
    country_col = None
    iso_col = None
    for c in wgi_raw.columns:
        lc = c.lower()
        if lc in ["economy", "country", "country name"]:
            country_col = c
        if lc in ["iso3", "country code", "economy code"]:
            iso_col = c

    # Extract latest numeric year-like column.
    year_cols = []
    for c in wgi_raw.columns:
        try:
            if 1990 <= int(str(c)[:4]) <= 2030:
                year_cols.append(c)
        except Exception:
            pass

    if not year_cols or iso_col is None:
        warnings.warn("Could not parse WGI schema. Using fallback risk.")
        return parse_wgi_risk(pd.DataFrame({
            "country": SUPPLIER_NAMES,
            "iso3": [SUPPLIERS[c]["iso3"] for c in SUPPLIER_NAMES],
            "political_stability_estimate": [0.0, 1.0, 0.4, -1.5, -0.2, -1.0, 0.6, -0.3, -0.1, -0.2],
            "data_source": "fallback_curated"
        }))

    latest_col = sorted(year_cols)[-1]
    temp = wgi_raw[[iso_col, latest_col]].copy()
    temp.columns = ["iso3", "stability"]
    temp = temp.dropna()
    iso_to_name = {v["iso3"]: k for k, v in SUPPLIERS.items()}
    temp = temp[temp["iso3"].isin(iso_to_name.keys())].copy()
    temp["supplier"] = temp["iso3"].map(iso_to_name)
    x = temp["stability"].astype(float)
    temp["risk"] = (x.max() - x) / (x.max() - x.min() + EPS)
    temp["data_source"] = "World Bank WGI"
    return temp[["supplier", "iso3", "risk", "data_source"]]


def download_un_comtrade(force: bool = False, year: int = 2023) -> pd.DataFrame:
    """
    Attempts to download imports of mineral-related HS bundles by the five EU economies.
    Requires COMTRADE_API_KEY. If unavailable or fails, returns cached/fallback data.
    """
    out_path = DATA_DIR / f"UN_Comtrade_critical_minerals_{year}.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    api_key = os.getenv("COMTRADE_API_KEY", "").strip()
    if not api_key:
        warnings.warn("COMTRADE_API_KEY not set. Using fallback supplier-resource matrix.")
        fallback = fallback_supplier_resource_matrix(long_format=True)
        fallback.to_csv(out_path, index=False)
        return fallback

    reporter_codes = ",".join([EU_ECONOMIES[e]["m49"] for e in ECONOMIES])
    rows = []
    headers = {"Ocp-Apim-Subscription-Key": api_key}

    # UN Comtrade API versions change over time. This endpoint works for many v1 subscriptions.
    base_url = "https://comtradeapi.un.org/data/v1/get/C/A/HS"

    for resource, codes in HS_CODES.items():
        for code in codes:
            params = {
                "cmdCode": code,
                "flowCode": "M",
                "reporterCode": reporter_codes,
                "partnerCode": ",".join([SUPPLIERS[s]["m49"] for s in SUPPLIER_NAMES]),
                "period": str(year),
                "includeDesc": "true",
            }
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=60)
                if r.status_code != 200:
                    warnings.warn(f"Comtrade status {r.status_code} for {resource} HS {code}: {r.text[:120]}")
                    continue
                payload = r.json()
                data = payload.get("data", [])
                for item in data:
                    partner = item.get("partnerDesc") or item.get("partnerISO") or item.get("partnerCode")
                    partner_iso = item.get("partnerISO")
                    value = item.get("primaryValue") or item.get("cifvalue") or item.get("fobvalue") or 0
                    rows.append({
                        "resource": resource,
                        "hs_code": code,
                        "partner": partner,
                        "partner_iso": partner_iso,
                        "trade_value_usd": float(value) if value is not None else 0.0,
                        "year": year,
                        "data_source": "UN Comtrade API",
                    })
                time.sleep(0.4)
            except Exception as exc:
                warnings.warn(f"Comtrade request failed for {resource} HS {code}: {exc}")

    if not rows:
        warnings.warn("Comtrade returned no data. Using fallback supplier-resource matrix.")
        fallback = fallback_supplier_resource_matrix(long_format=True)
        fallback.to_csv(out_path, index=False)
        return fallback

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def fallback_supplier_resource_matrix(long_format: bool = False) -> pd.DataFrame:
    """
    Transparent fallback supplier shares based on common global supply-chain patterns.
    Use only for testing if Comtrade API is not available.
    Rows sum to one.
    """
    data = pd.DataFrame({
        "resource": RESOURCES,
        "China":       [0.15, 0.10, 0.08, 0.65, 0.80, 0.10],
        "Australia":   [0.35, 0.03, 0.05, 0.02, 0.05, 0.05],
        "Chile":       [0.30, 0.00, 0.00, 0.00, 0.00, 0.25],
        "DR Congo":    [0.00, 0.65, 0.00, 0.00, 0.00, 0.05],
        "Indonesia":   [0.00, 0.02, 0.45, 0.00, 0.00, 0.03],
        "Russia":      [0.02, 0.05, 0.15, 0.02, 0.02, 0.05],
        "United States":[0.05, 0.03, 0.04, 0.05, 0.05, 0.07],
        "Morocco":     [0.00, 0.00, 0.00, 0.00, 0.02, 0.00],
        "South Africa":[0.00, 0.04, 0.10, 0.00, 0.01, 0.02],
        "Brazil":      [0.13, 0.08, 0.13, 0.26, 0.05, 0.38],
    })
    data = ensure_nonnegative_prob_matrix(data, "resource")
    if long_format:
        return data.melt(id_vars="resource", var_name="supplier", value_name="share").assign(
            trade_value_usd=lambda x: x["share"],
            data_source="fallback_curated_supplier_shares"
        )
    return data


def build_A_RC(comtrade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns resource x supplier matrix A_RC.
    If Comtrade data are long with trade values, compute supplier shares by resource.
    """
    out_path = RESULTS_DIR / "A_RC_supplier_resource.csv"
    if {"resource", "supplier", "share"}.issubset(comtrade_df.columns):
        pivot = comtrade_df.pivot_table(index="resource", columns="supplier", values="share", aggfunc="sum").reindex(index=RESOURCES, columns=SUPPLIER_NAMES).fillna(0)
    else:
        # Map partner ISO or desc to suppliers
        iso_to_name = {v["iso3"]: k for k, v in SUPPLIERS.items()}
        df = comtrade_df.copy()
        if "partner_iso" in df.columns:
            df["supplier"] = df["partner_iso"].map(iso_to_name)
        if "supplier" not in df.columns or df["supplier"].isna().all():
            # try partner name matching
            def match_partner(x):
                x = str(x).lower()
                for s in SUPPLIER_NAMES:
                    if s.lower() in x:
                        return s
                return None
            df["supplier"] = df.get("partner", "").apply(match_partner)
        df = df[df["supplier"].isin(SUPPLIER_NAMES)].copy()
        if df.empty:
            return build_A_RC(fallback_supplier_resource_matrix(long_format=True))
        pivot = df.pivot_table(index="resource", columns="supplier", values="trade_value_usd", aggfunc="sum").reindex(index=RESOURCES, columns=SUPPLIER_NAMES).fillna(0)
    mat = pivot.to_numpy(dtype=float)
    mat = dirichlet_perturb_matrix(mat, seed=999, concentration=1e9)  # normalize deterministic
    result = pd.DataFrame(mat, index=RESOURCES, columns=SUPPLIER_NAMES)
    result.insert(0, "resource", RESOURCES)
    result.to_csv(out_path, index=False)
    return result


def load_or_create_A_TR() -> pd.DataFrame:
    """
    Technology-resource requirement matrix. If IEA-derived file exists, use it.
    Required local columns: technology, resource, intensity.
    Otherwise fallback matrix is used.
    """
    local = DATA_DIR / "IEA_CriticalMinerals_TechnologyRequirements.csv"
    out_path = RESULTS_DIR / "A_TR_technology_resource.csv"
    if local.exists():
        df = pd.read_csv(local)
        required = {"technology", "resource", "intensity"}
        if required.issubset(df.columns):
            pivot = df.pivot_table(index="technology", columns="resource", values="intensity", aggfunc="sum").reindex(index=TECHNOLOGIES, columns=RESOURCES).fillna(0)
            mat = dirichlet_perturb_matrix(pivot.to_numpy(dtype=float), seed=1000, concentration=1e9)
            result = pd.DataFrame(mat, index=TECHNOLOGIES, columns=RESOURCES)
            result.insert(0, "technology", TECHNOLOGIES)
            result["data_source"] = "local_IEA_calibrated"
            result.to_csv(out_path, index=False)
            return result
        warnings.warn(f"{local} exists but lacks required columns {required}; using fallback A_TR.")

    data = pd.DataFrame({
        "technology": TECHNOLOGIES,
        "Lithium":     [0.30, 0.00, 0.00, 0.00, 0.02],
        "Cobalt":      [0.15, 0.00, 0.00, 0.00, 0.00],
        "Nickel":      [0.25, 0.02, 0.00, 0.03, 0.03],
        "Graphite":    [0.20, 0.00, 0.00, 0.00, 0.00],
        "Rare earths": [0.02, 0.45, 0.00, 0.02, 0.00],
        "Copper":      [0.08, 0.53, 1.00, 0.95, 0.95],
    })
    data = ensure_nonnegative_prob_matrix(data, "technology")
    data["data_source"] = "fallback_engineering_template"
    data.to_csv(out_path, index=False)
    return data


def load_or_create_X() -> pd.DataFrame:
    """
    Economy-technology portfolio weights. If local file exists, use it.
    Required local columns: economy, technology, weight.
    Otherwise fallback stylized EU portfolios are used.
    """
    local = DATA_DIR / "Eurostat_IRENA_technology_portfolios.csv"
    out_path = RESULTS_DIR / "X_economy_technology_portfolios.csv"
    if local.exists():
        df = pd.read_csv(local)
        required = {"economy", "technology", "weight"}
        if required.issubset(df.columns):
            pivot = df.pivot_table(index="economy", columns="technology", values="weight", aggfunc="sum").reindex(index=ECONOMIES, columns=TECHNOLOGIES).fillna(0)
            mat = dirichlet_perturb_matrix(pivot.to_numpy(dtype=float), seed=1001, concentration=1e9)
            result = pd.DataFrame(mat, index=ECONOMIES, columns=TECHNOLOGIES)
            result.insert(0, "economy", ECONOMIES)
            result["data_source"] = "local_Eurostat_IRENA_calibrated"
            result.to_csv(out_path, index=False)
            return result
        warnings.warn(f"{local} exists but lacks required columns {required}; using fallback X.")

    data = pd.DataFrame({
        "economy": ECONOMIES,
        "EV batteries": [0.22, 0.35, 0.22, 0.25, 0.20],
        "Wind":         [0.22, 0.25, 0.25, 0.20, 0.35],
        "Solar PV":     [0.25, 0.20, 0.35, 0.32, 0.15],
        "Electrolysers":[0.16, 0.10, 0.10, 0.10, 0.15],
        "Power grids":  [0.15, 0.10, 0.08, 0.13, 0.15],
    })
    data = ensure_nonnegative_prob_matrix(data, "economy")
    data["data_source"] = "fallback_stylized_portfolios"
    data.to_csv(out_path, index=False)
    return data


def prepare_data(force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    save_metadata()

    wgi_raw = download_wgi(force=force_download)
    risk = parse_wgi_risk(wgi_raw)
    risk = risk.set_index("supplier").reindex(SUPPLIER_NAMES).reset_index()
    # Fill missing risk with neutral 0.5
    risk["risk"] = risk["risk"].fillna(0.5)
    risk.to_csv(RESULTS_DIR / "supplier_geopolitical_risk.csv", index=False)

    comtrade = download_un_comtrade(force=force_download)
    A_RC = build_A_RC(comtrade)
    A_TR = load_or_create_A_TR()
    X = load_or_create_X()

    sources = pd.DataFrame([
        {"component": "A_RC", "source_used": "UN Comtrade if COMTRADE_API_KEY available; otherwise fallback flagged in Data_geo/Results_Geo"},
        {"component": "A_TR", "source_used": "Local IEA-derived file if provided; otherwise fallback engineering template"},
        {"component": "X", "source_used": "Local Eurostat/IRENA portfolio file if provided; otherwise fallback stylized portfolios"},
        {"component": "risk", "source_used": "World Bank WGI political stability estimate, converted to risk in [0,1]"},
    ])
    sources.to_csv(TABLE_DIR / "Table1_data_sources.csv", index=False)

    return A_RC, A_TR, X, risk


# =============================================================================
# 3. Theory computations using NumPy/Torch
# =============================================================================

def extract_numeric_matrices(A_RC_df: pd.DataFrame, A_TR_df: pd.DataFrame, X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_RC = A_RC_df.set_index("resource").reindex(RESOURCES)[SUPPLIER_NAMES].to_numpy(dtype=float)
    A_TR = A_TR_df.set_index("technology").reindex(TECHNOLOGIES)[RESOURCES].to_numpy(dtype=float)
    X = X_df.set_index("economy").reindex(ECONOMIES)[TECHNOLOGIES].to_numpy(dtype=float)
    return A_RC, A_TR, X


def ces_capacity_numpy(A_RC: np.ndarray, A_TR: np.ndarray, x: np.ndarray, z: np.ndarray,
                       rho: float = RHO, eta: float = ETA) -> float:
    """
    K_e(x,z) with baseline Sbar=1 for all resources.
    A_RC: R x C
    A_TR: T x R
    x: T
    z: C
    """
    S = 1.0 - A_RC @ z
    S = np.maximum(S, 1e-6)
    T_cap = (A_TR @ (S ** rho)) ** (1.0 / rho)
    K = (x @ (T_cap ** eta)) ** (1.0 / eta)
    return float(K)


def loss_numpy(A_RC: np.ndarray, A_TR: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    return ces_capacity_numpy(A_RC, A_TR, x, np.zeros_like(z)) - ces_capacity_numpy(A_RC, A_TR, x, z)


def compute_Q_chi(A_RC: np.ndarray, A_TR: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
    Q = np.diag(x) @ A_TR @ A_RC
    vals = np.linalg.eigvalsh(Q @ Q.T)
    chi = float(np.max(vals))
    return Q, chi


def compute_gamma_finite_diff(A_RC: np.ndarray, A_TR: np.ndarray, x: np.ndarray, h: float = 1e-5) -> Tuple[np.ndarray, float]:
    C = A_RC.shape[1]
    g = np.zeros(C)
    z0 = np.zeros(C)
    L0 = loss_numpy(A_RC, A_TR, x, z0)
    for c in range(C):
        zp = z0.copy()
        zp[c] = h
        g[c] = (loss_numpy(A_RC, A_TR, x, zp) - L0) / h
    return g, float(np.linalg.norm(g))


def worst_case_torch(A_RC_np: np.ndarray, A_TR_np: np.ndarray, x_np: np.ndarray,
                     delta: float = DELTA, steps: int = 1200, lr: float = 0.05,
                     seed: int = 0) -> Tuple[np.ndarray, float]:
    """
    Maximizes transition loss over z >=0, sum z <= delta.
    Parameterizes z = delta * softmax(logits), which uses the full budget.
    This is appropriate because loss is monotone in z.
    """
    if not TORCH_AVAILABLE:
        # Fallback: evaluate all single-country extreme shocks and return best
        best_z, best_loss = None, -1.0
        for c in range(A_RC_np.shape[1]):
            z = np.zeros(A_RC_np.shape[1])
            z[c] = delta
            val = loss_numpy(A_RC_np, A_TR_np, x_np, z)
            if val > best_loss:
                best_loss, best_z = val, z
        return best_z, float(best_loss)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    A_RC = torch.tensor(A_RC_np, dtype=torch.float64, device=device)
    A_TR = torch.tensor(A_TR_np, dtype=torch.float64, device=device)
    x = torch.tensor(x_np, dtype=torch.float64, device=device)

    logits = torch.zeros(A_RC_np.shape[1], dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    def K_torch(z):
        S = 1.0 - A_RC @ z
        S = torch.clamp(S, min=1e-6)
        T_cap = torch.pow(A_TR @ torch.pow(S, RHO), 1.0 / RHO)
        return torch.pow(torch.sum(x * torch.pow(T_cap, ETA)), 1.0 / ETA)

    K0 = K_torch(torch.zeros(A_RC_np.shape[1], dtype=torch.float64, device=device)).detach()

    best_loss = -1e9
    best_z = None

    for _ in range(steps):
        opt.zero_grad()
        z = delta * torch.softmax(logits, dim=0)
        loss = K0 - K_torch(z)
        objective = -loss
        objective.backward()
        opt.step()
        if float(loss.detach().cpu()) > best_loss:
            best_loss = float(loss.detach().cpu())
            best_z = z.detach().cpu().numpy()

    return best_z, float(best_loss)


def compute_resilience_marginal_benefits(A_RC: np.ndarray, A_TR: np.ndarray, X: np.ndarray,
                                         delta: float = DELTA, eps_u: float = 0.05) -> pd.DataFrame:
    """
    Approximate marginal benefit of resource-level resilience.
    Applies attenuation psi_r(u)=exp(-u_r) locally by modifying A_RC rows.
    """
    rows = []
    base_losses = {}
    for e_idx, e in enumerate(ECONOMIES):
        _, base_v = worst_case_torch(A_RC, A_TR, X[e_idx], delta=delta, seed=1234 + e_idx, steps=600)
        base_losses[e] = base_v
        for r_idx, r in enumerate(RESOURCES):
            A_mod = A_RC.copy()
            A_mod[r_idx, :] = np.exp(-eps_u) * A_mod[r_idx, :]
            _, v_mod = worst_case_torch(A_mod, A_TR, X[e_idx], delta=delta, seed=4321 + e_idx + r_idx, steps=600)
            rows.append({
                "economy": e,
                "resource": r,
                "base_vulnerability": base_v,
                "vulnerability_after_small_resilience": v_mod,
                "marginal_benefit": (base_v - v_mod) / eps_u
            })
    return pd.DataFrame(rows)


def run_all_computations(force_compute: bool = False) -> None:
    metrics_path = RESULTS_DIR / "results_all_seeds_metrics.csv"
    if metrics_path.exists() and not force_compute:
        print(f"[cache] Results already exist in {RESULTS_DIR}. Use --force-compute to recompute.")
        return

    A_RC_df = pd.read_csv(RESULTS_DIR / "A_RC_supplier_resource.csv")
    A_TR_df = pd.read_csv(RESULTS_DIR / "A_TR_technology_resource.csv")
    X_df = pd.read_csv(RESULTS_DIR / "X_economy_technology_portfolios.csv")
    A_RC_base, A_TR_base, X_base = extract_numeric_matrices(A_RC_df, A_TR_df, X_df)

    metrics_rows = []
    scenario_rows = []
    worst_z_rows = []
    resilience_rows = []

    for seed in SEEDS:
        A_RC = dirichlet_perturb_matrix(A_RC_base, seed=10000 + seed, concentration=250)
        A_TR = dirichlet_perturb_matrix(A_TR_base, seed=20000 + seed, concentration=150)
        X = dirichlet_perturb_matrix(X_base, seed=30000 + seed, concentration=200)

        # Save seed-specific matrices for reproducibility
        pd.DataFrame(A_RC, index=RESOURCES, columns=SUPPLIER_NAMES).to_csv(RESULTS_DIR / f"A_RC_seed_{seed:02d}.csv")
        pd.DataFrame(A_TR, index=TECHNOLOGIES, columns=RESOURCES).to_csv(RESULTS_DIR / f"A_TR_seed_{seed:02d}.csv")
        pd.DataFrame(X, index=ECONOMIES, columns=TECHNOLOGIES).to_csv(RESULTS_DIR / f"X_seed_{seed:02d}.csv")

        for e_idx, econ in enumerate(ECONOMIES):
            x = X[e_idx]
            Q, chi = compute_Q_chi(A_RC, A_TR, x)
            grad, gamma = compute_gamma_finite_diff(A_RC, A_TR, x)
            hhi = float(np.sum(x ** 2))
            K0 = ces_capacity_numpy(A_RC, A_TR, x, np.zeros(len(SUPPLIER_NAMES)))

            z_wc, v_wc = worst_case_torch(A_RC, A_TR, x, delta=DELTA, seed=40000 + 100 * seed + e_idx)

            metrics_rows.append({
                "seed": seed,
                "economy": econ,
                "K0": K0,
                "HHI_technology": hhi,
                "chi_spectral": chi,
                "gamma_local": gamma,
                "V_worstcase": v_wc,
                "device": "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else ("cpu_torch" if TORCH_AVAILABLE else "numpy_fallback")
            })

            for c_idx, supplier in enumerate(SUPPLIER_NAMES):
                worst_z_rows.append({
                    "seed": seed,
                    "economy": econ,
                    "supplier": supplier,
                    "worstcase_z": z_wc[c_idx]
                })

            for scen_name, scen in SCENARIOS.items():
                z = country_vector_from_scenario(scen)
                scenario_rows.append({
                    "seed": seed,
                    "economy": econ,
                    "scenario": scen_name,
                    "loss": loss_numpy(A_RC, A_TR, x, z),
                    "relative_loss_pct": 100 * loss_numpy(A_RC, A_TR, x, z) / max(K0, EPS)
                })

        # marginal resilience benefits once per seed
        res_seed = compute_resilience_marginal_benefits(A_RC, A_TR, X, delta=DELTA, eps_u=0.05)
        res_seed["seed"] = seed
        resilience_rows.append(res_seed)

    metrics = pd.DataFrame(metrics_rows)
    scenarios = pd.DataFrame(scenario_rows)
    worst_z = pd.DataFrame(worst_z_rows)
    resilience = pd.concat(resilience_rows, ignore_index=True)

    metrics.to_csv(RESULTS_DIR / "results_all_seeds_metrics.csv", index=False)
    scenarios.to_csv(RESULTS_DIR / "scenario_losses_by_seed.csv", index=False)
    worst_z.to_csv(RESULTS_DIR / "worstcase_shocks_by_seed.csv", index=False)
    resilience.to_csv(RESULTS_DIR / "resilience_marginal_benefits_by_seed.csv", index=False)

    # Tables <=4
    metrics_summary = metrics.groupby("economy").agg(
        K0_mean=("K0", "mean"),
        HHI_mean=("HHI_technology", "mean"),
        chi_mean=("chi_spectral", "mean"),
        chi_sd=("chi_spectral", "std"),
        gamma_mean=("gamma_local", "mean"),
        V_mean=("V_worstcase", "mean"),
        V_sd=("V_worstcase", "std"),
    ).reset_index()
    metrics_summary.to_csv(TABLE_DIR / "Table2_baseline_vulnerability_metrics.csv", index=False)

    scenario_summary = scenarios.groupby(["economy", "scenario"]).agg(
        loss_mean=("loss", "mean"),
        loss_sd=("loss", "std"),
        relative_loss_pct_mean=("relative_loss_pct", "mean"),
        relative_loss_pct_sd=("relative_loss_pct", "std"),
    ).reset_index()
    scenario_summary.to_csv(TABLE_DIR / "Table3_counterfactual_scenarios.csv", index=False)

    resilience_summary = resilience.groupby("resource").agg(
        marginal_benefit_mean=("marginal_benefit", "mean"),
        marginal_benefit_sd=("marginal_benefit", "std"),
    ).reset_index().sort_values("marginal_benefit_mean", ascending=False)
    resilience_summary.to_csv(TABLE_DIR / "Table4_resilience_priority_ranking.csv", index=False)

    print(f"[done] Computations saved in {RESULTS_DIR}")


# =============================================================================
# 4. Figures
# =============================================================================

def set_plot_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, name: str):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1_supplier_resource_heatmap():
    df = pd.read_csv(RESULTS_DIR / "A_RC_supplier_resource.csv").set_index("resource")[SUPPLIER_NAMES]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(df.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(SUPPLIER_NAMES)))
    ax.set_xticklabels(SUPPLIER_NAMES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(RESOURCES)))
    ax.set_yticklabels(RESOURCES)
    ax.set_title("Supplier concentration by critical mineral")
    ax.set_xlabel("Supplier country")
    ax.set_ylabel("Critical mineral")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Supplier share")
    save_fig(fig, "Fig1_supplier_resource_heatmap")


def fig2_multilayer_network():
    A_RC = pd.read_csv(RESULTS_DIR / "A_RC_supplier_resource.csv").set_index("resource")[SUPPLIER_NAMES]
    A_TR = pd.read_csv(RESULTS_DIR / "A_TR_technology_resource.csv").set_index("technology")[RESOURCES]
    X = pd.read_csv(RESULTS_DIR / "X_economy_technology_portfolios.csv").set_index("economy")[TECHNOLOGIES]

    if not NETWORKX_AVAILABLE:
        warnings.warn("networkx not installed; skipping network figure.")
        return

    G = nx.DiGraph()
    # add nodes with layers
    for s in SUPPLIER_NAMES:
        G.add_node(f"C:{s}", layer=0, label=s)
    for r in RESOURCES:
        G.add_node(f"R:{r}", layer=1, label=r)
    for t in TECHNOLOGIES:
        G.add_node(f"T:{t}", layer=2, label=t)
    for e in ECONOMIES:
        G.add_node(f"E:{e}", layer=3, label=e)

    # top edges only for readability
    for r in RESOURCES:
        top = A_RC.loc[r].sort_values(ascending=False).head(2)
        for s, w in top.items():
            if w > 0:
                G.add_edge(f"C:{s}", f"R:{r}", weight=w)
    for t in TECHNOLOGIES:
        top = A_TR.loc[t].sort_values(ascending=False).head(2)
        for r, w in top.items():
            if w > 0:
                G.add_edge(f"R:{r}", f"T:{t}", weight=w)
    for e in ECONOMIES:
        top = X.loc[e].sort_values(ascending=False).head(2)
        for t, w in top.items():
            if w > 0:
                G.add_edge(f"T:{t}", f"E:{e}", weight=w)

    pos = {}
    layer_y = {0: 0, 1: 1, 2: 2, 3: 3}
    for layer in range(4):
        nodes = [n for n, d in G.nodes(data=True) if d["layer"] == layer]
        xs = np.linspace(0, 1, len(nodes)) if len(nodes) > 1 else np.array([0.5])
        for x, n in zip(xs, nodes):
            pos[n] = (x, 3 - layer_y[layer])

    fig, ax = plt.subplots(figsize=(12, 7))
    widths = [0.5 + 3.0 * G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, width=widths, alpha=0.45, arrowstyle="-|>")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500)
    labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)
    ax.set_title("Multilayer transition network: suppliers, minerals, technologies, economies")
    ax.axis("off")
    save_fig(fig, "Fig2_multilayer_network")


def fig3_spectral_vulnerability_bars():
    df = pd.read_csv(RESULTS_DIR / "results_all_seeds_metrics.csv")
    summary = df.groupby("economy").agg(chi_mean=("chi_spectral", "mean"), chi_sd=("chi_spectral", "std")).reindex(ECONOMIES)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(summary.index, summary["chi_mean"], yerr=summary["chi_sd"], capsize=4)
    ax.set_title("Spectral upstream concentration across European economies")
    ax.set_ylabel(r"$\chi_e = \lambda_{\max}(Q_eQ_e')$")
    ax.set_xlabel("Economy")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, "Fig3_spectral_concentration_robustness")


def fig4_hidden_vulnerability_scatter():
    df = pd.read_csv(RESULTS_DIR / "results_all_seeds_metrics.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    for econ in ECONOMIES:
        sub = df[df["economy"] == econ]
        ax.scatter(sub["HHI_technology"], sub["chi_spectral"], label=econ, alpha=0.75)
    ax.set_title("Visible diversification versus hidden upstream concentration")
    ax.set_xlabel("Technology concentration (HHI)")
    ax.set_ylabel("Spectral upstream concentration")
    ax.legend(frameon=False)
    save_fig(fig, "Fig4_diversification_vs_hidden_concentration")


def fig5_scenario_loss_heatmap():
    df = pd.read_csv(RESULTS_DIR / "scenario_losses_by_seed.csv")
    pivot = df.groupby(["economy", "scenario"])["relative_loss_pct"].mean().unstack().reindex(ECONOMIES)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ECONOMIES)))
    ax.set_yticklabels(ECONOMIES)
    ax.set_title("Transition-capacity losses under geopolitical shock scenarios")
    ax.set_ylabel("Economy")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean relative loss (%)")
    save_fig(fig, "Fig5_counterfactual_scenario_losses")


def fig6_worstcase_shock_composition():
    df = pd.read_csv(RESULTS_DIR / "worstcase_shocks_by_seed.csv")
    pivot = df.groupby(["economy", "supplier"])["worstcase_z"].mean().unstack().reindex(ECONOMIES)[SUPPLIER_NAMES]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(ECONOMIES))
    x = np.arange(len(ECONOMIES))
    for supplier in SUPPLIER_NAMES:
        vals = pivot[supplier].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=supplier)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(ECONOMIES, rotation=30)
    ax.set_ylabel("Worst-case shock allocation")
    ax.set_title("Composition of worst-case geopolitical disturbances")
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper right")
    save_fig(fig, "Fig6_worstcase_shock_composition")


def fig7_resilience_priority():
    df = pd.read_csv(RESULTS_DIR / "resilience_marginal_benefits_by_seed.csv")
    summary = df.groupby("resource").agg(mean=("marginal_benefit", "mean"), sd=("marginal_benefit", "std")).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(summary.index, summary["mean"], yerr=summary["sd"], capsize=4)
    ax.set_title("Resilience priority ranking by critical mineral")
    ax.set_ylabel("Marginal vulnerability reduction")
    ax.set_xlabel("Critical mineral")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, "Fig7_resilience_priority_ranking")


def generate_figures():
    set_plot_style()
    fig1_supplier_resource_heatmap()
    fig2_multilayer_network()
    fig3_spectral_vulnerability_bars()
    fig4_hidden_vulnerability_scatter()
    fig5_scenario_loss_heatmap()
    fig6_worstcase_shock_composition()
    fig7_resilience_priority()
    print(f"[done] Figures saved in {FIG_DIR}")


# =============================================================================
# 5. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Geopolitical transition vulnerability pipeline")
    parser.add_argument("--force-download", action="store_true", help="Re-download raw public data where possible")
    parser.add_argument("--force-compute", action="store_true", help="Recompute results even if CSV files exist")
    parser.add_argument("--download-only", action="store_true", help="Only download/cache data")
    parser.add_argument("--compute-only", action="store_true", help="Skip downloads and use cached Data_geo/Results_Geo inputs")
    parser.add_argument("--figures-only", action="store_true", help="Only regenerate figures from Results_Geo CSV files")
    args = parser.parse_args()

    if args.figures_only:
        generate_figures()
        return

    if not args.compute_only:
        prepare_data(force_download=args.force_download)

    if args.download_only:
        print(f"[done] Data prepared in {DATA_DIR} and matrices in {RESULTS_DIR}")
        return

    run_all_computations(force_compute=args.force_compute)
    generate_figures()


if __name__ == "__main__":
    main()
