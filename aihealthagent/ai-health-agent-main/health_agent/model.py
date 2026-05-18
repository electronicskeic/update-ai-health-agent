from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RiskPrediction:
    obesity_probability: float
    risk_level: str


@dataclass(frozen=True)
class ModelQuality:
    roc_auc_mean: float
    roc_auc_std: float


def train_obesity_model(df_model: pd.DataFrame) -> Pipeline:
    X = df_model[["Age", "Height", "Weight", "BMI"]].astype(float).to_numpy()
    y = df_model["target_obese"].astype(int).to_numpy()

    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    pipe.fit(X, y)
    return pipe


def predict_obesity_risk(model: Pipeline, *, age: int, height_m: float, weight_kg: float, bmi: float) -> RiskPrediction:
    X = np.array([[float(age), float(height_m), float(weight_kg), float(bmi)]], dtype=float)
    p = float(model.predict_proba(X)[0, 1])

    if p < 0.33:
        lvl = "Low"
    elif p < 0.66:
        lvl = "Moderate"
    else:
        lvl = "High"
    return RiskPrediction(obesity_probability=p, risk_level=lvl)


def evaluate_model(df_model: pd.DataFrame, *, folds: int = 5, seed: int = 42) -> ModelQuality:
    X = df_model[["Age", "Height", "Weight", "BMI"]].astype(float).to_numpy()
    y = df_model["target_obese"].astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    aucs: list[float] = []
    for train_idx, test_idx in cv.split(X, y):
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[test_idx], proba)))

    return ModelQuality(roc_auc_mean=float(np.mean(aucs)), roc_auc_std=float(np.std(aucs)))


@dataclass(frozen=True)
class DiabetesRiskPrediction:
    diabetes_probability: float
    risk_level: str


def train_diabetes_model(df_model: pd.DataFrame) -> Pipeline:
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    X = df_model[features].astype(float).to_numpy()
    y = df_model["target_diabetes"].astype(int).to_numpy()

    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    pipe.fit(X, y)
    return pipe


def predict_diabetes_risk(model: Pipeline, *, pregnancies: int, glucose: float, bp: float, skin: float, insulin: float, bmi: float, dpf: float, age: int) -> DiabetesRiskPrediction:
    X = np.array([[float(pregnancies), float(glucose), float(bp), float(skin), float(insulin), float(bmi), float(dpf), float(age)]], dtype=float)
    p = float(model.predict_proba(X)[0, 1])

    if p < 0.33:
        lvl = "Low"
    elif p < 0.66:
        lvl = "Moderate"
    else:
        lvl = "High"
    return DiabetesRiskPrediction(diabetes_probability=p, risk_level=lvl)


def evaluate_diabetes_model(df_model: pd.DataFrame, *, folds: int = 5, seed: int = 42) -> ModelQuality:
    features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    X = df_model[features].astype(float).to_numpy()
    y = df_model["target_diabetes"].astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    aucs: list[float] = []
    for train_idx, test_idx in cv.split(X, y):
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[test_idx], proba)))

    return ModelQuality(roc_auc_mean=float(np.mean(aucs)), roc_auc_std=float(np.std(aucs)))
