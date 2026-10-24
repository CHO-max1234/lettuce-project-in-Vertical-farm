"""Frozen slope/embedding checks used by supplementary figure reproduction.

The arithmetic is copied from the original research helper without modification.
Callers pass sorted, already cutoff-filtered plant histories; no fitting occurs.
"""
from __future__ import annotations
import math
import re
import numpy as np
import pandas as pd
def safe_num(value: object) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def window_slope(history: pd.DataFrame, col: str, cutoff: pd.Timestamp, days: int) -> float:
    if col not in history.columns or history.empty or pd.isna(cutoff):
        return np.nan
    h = history[history['history_date'].ge(cutoff - pd.Timedelta(days=days))].copy()
    h = h[pd.to_numeric(h[col], errors='coerce').notna()]
    if h['history_date'].nunique() < 2:
        return np.nan
    first = h.iloc[0]
    last = h.iloc[-1]
    gap = (last['history_date'] - first['history_date']).days
    if gap <= 0:
        return np.nan
    return float((safe_num(last[col]) - safe_num(first[col])) / gap)

def _embedding_columns(history: pd.DataFrame) -> list[str]:
    return sorted([c for c in history.columns if re.fullmatch('emb_\\d{3}', str(c))])

def _embedding_vector(row: pd.Series, emb_cols: list[str]) -> np.ndarray | None:
    if not emb_cols:
        return None
    values = pd.to_numeric(row[emb_cols], errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    return values

def _cosine_change(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    if vec_a is None or vec_b is None:
        return np.nan
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    cosine = float(np.dot(vec_a, vec_b) / denom)
    return float(1.0 - cosine)

def _l2_change(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    if vec_a is None or vec_b is None:
        return np.nan
    return float(np.linalg.norm(vec_b - vec_a))

def _embedding_window_change(history: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> tuple[float, float, float]:
    emb_cols = _embedding_columns(history)
    if not emb_cols or history.empty or pd.isna(cutoff):
        return (np.nan, np.nan, np.nan)
    h = history[history['history_date'].ge(cutoff - pd.Timedelta(days=days))].dropna(subset=emb_cols, how='any')
    if h['history_date'].nunique() < 2:
        return (np.nan, np.nan, np.nan)
    first = h.iloc[0]
    last = h.iloc[-1]
    va = _embedding_vector(first, emb_cols)
    vb = _embedding_vector(last, emb_cols)
    gap = (last['history_date'] - first['history_date']).days
    l2 = _l2_change(va, vb)
    return (_cosine_change(va, vb), l2, l2 / gap if pd.notna(l2) and gap > 0 else np.nan)
