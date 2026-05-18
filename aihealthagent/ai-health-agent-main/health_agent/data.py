from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSummary:
    n: int
    bmi_mean: float
    bmi_median: float
    overweight_rate: float
    obese_rate: float
    age_bmi_corr: float
    bmi_bins: dict[str, int]


def _norm_col(c: str) -> str:
    return "".join(ch.lower() for ch in c.strip() if ch.isalnum())


def load_dataset(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(str(path))

    # Expected by README: `bmi` sheet. Fall back gracefully.
    try:
        df = pd.read_excel(path, sheet_name="bmi")
    except Exception:
        df = pd.read_excel(path)

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Normalize common column naming variants.
    rename: dict[str, str] = {}
    for c in df.columns:
        nc = _norm_col(c)
        if nc in {"age", "ages", "ageyears"}:
            rename[c] = "Age"
        elif nc in {"height", "heightm", "heightmeters", "heightmeter"}:
            rename[c] = "Height"
        elif nc in {"weight", "weightkg", "weightkgs"}:
            rename[c] = "Weight"
        elif nc in {"bmi"}:
            rename[c] = "BMI"
        elif nc in {"bmiclass", "bmicategory", "bmicat", "class"}:
            rename[c] = "BMI Class"
    if rename:
        df = df.rename(columns=rename)

    # Coerce numeric columns.
    for col in ("Age", "Height", "Weight", "BMI"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[c for c in ("Age", "Height", "Weight", "BMI") if c in df.columns])
    return df


def calculate_bmi(height_m: float, weight_kg: float) -> float:
    h = float(height_m)
    w = float(weight_kg)
    if h <= 0:
        raise ValueError("Height must be > 0.")
    if w <= 0:
        raise ValueError("Weight must be > 0.")
    return w / (h * h)


def bmi_category(bmi: float) -> str:
    b = float(bmi)
    if b < 18.5:
        return "Underweight"
    if b < 25:
        return "Normal"
    if b < 30:
        return "Overweight"
    if b < 35:
        return "Obesity I"
    if b < 40:
        return "Obesity II"
    return "Obesity III"


def summarize_dataset(df: pd.DataFrame) -> DatasetSummary:
    n = int(len(df))
    bmi = df["BMI"].astype(float)
    overweight_rate = float((bmi >= 25).mean()) if n else 0.0
    obese_rate = float((bmi >= 30).mean()) if n else 0.0
    age_bmi_corr = float(df["Age"].astype(float).corr(bmi)) if n else 0.0
    bins = {
        "Underweight (<18.5)": int((bmi < 18.5).sum()),
        "Normal (18.5–24.9)": int(((bmi >= 18.5) & (bmi < 25)).sum()),
        "Overweight (25–29.9)": int(((bmi >= 25) & (bmi < 30)).sum()),
        "Obese (>=30)": int((bmi >= 30).sum()),
    }
    return DatasetSummary(
        n=n,
        bmi_mean=float(bmi.mean()) if n else float("nan"),
        bmi_median=float(bmi.median()) if n else float("nan"),
        overweight_rate=overweight_rate,
        obese_rate=obese_rate,
        age_bmi_corr=age_bmi_corr if np.isfinite(age_bmi_corr) else 0.0,
        bmi_bins=bins,
    )


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"Age", "Height", "Weight", "BMI"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Height"] = pd.to_numeric(out["Height"], errors="coerce")
    out["Weight"] = pd.to_numeric(out["Weight"], errors="coerce")
    out["BMI"] = pd.to_numeric(out["BMI"], errors="coerce")

    # Binary target: obesity = BMI>=30. If explicit BMI Class exists, still use BMI for consistency.
    out["target_obese"] = (out["BMI"] >= 30).astype(int)
    out = out.dropna(subset=["Age", "Height", "Weight", "BMI", "target_obese"])
    return out[["Age", "Height", "Weight", "BMI", "target_obese"]]


def as_feature_dict(*, age: int, height_m: float, weight_kg: float, bmi: float) -> dict[str, Any]:
    return {"Age": float(age), "Height": float(height_m), "Weight": float(weight_kg), "BMI": float(bmi)}


@dataclass(frozen=True)
class DiabetesDatasetSummary:
    n: int
    positive_rate: float
    glucose_mean: float
    bmi_mean: float
    age_mean: float


def load_diabetes_dataset(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def summarize_diabetes_dataset(df: pd.DataFrame) -> DiabetesDatasetSummary:
    n = int(len(df))
    if n == 0 or "Outcome" not in df.columns:
        return DiabetesDatasetSummary(n=0, positive_rate=0.0, glucose_mean=0.0, bmi_mean=0.0, age_mean=0.0)
    
    outcome = df["Outcome"].astype(int)
    return DiabetesDatasetSummary(
        n=n,
        positive_rate=float(outcome.mean()),
        glucose_mean=float(df["Glucose"].mean()) if "Glucose" in df.columns else float("nan"),
        bmi_mean=float(df["BMI"].mean()) if "BMI" in df.columns else float("nan"),
        age_mean=float(df["Age"].mean()) if "Age" in df.columns else float("nan"),
    )


def to_diabetes_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Diabetes dataset missing required columns: {sorted(missing)}")
    
    out = df.copy()
    out["target_diabetes"] = out["Outcome"].astype(int)
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "target_diabetes"]
    return out[features]
