from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_dataset(rows: int = 2600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(69, 8.5, rows), 45, 92)
    education_years = np.clip(rng.normal(12.8, 3.2, rows), 0, 22)
    family_history = rng.binomial(1, 0.28, rows)
    apoe4 = rng.binomial(1, 0.24 + 0.12 * family_history, rows)
    hypertension = rng.binomial(1, sigmoid((age - 63) / 10), rows)
    diabetes = rng.binomial(1, 0.24 + 0.08 * hypertension, rows)

    latent = (
        0.065 * (age - 66)
        - 0.08 * (education_years - 12)
        + 0.42 * family_history
        + 0.7 * apoe4
        + 0.28 * hypertension
        + 0.26 * diabetes
        + rng.normal(0, 0.85, rows)
    )

    impairment = sigmoid(latent)
    mmse = np.clip(30 - 8.4 * impairment + rng.normal(0, 1.2, rows), 8, 30)
    moca = np.clip(30 - 10.2 * impairment + rng.normal(0, 1.5, rows), 5, 30)
    cdr = np.select(
        [impairment < 0.34, impairment < 0.64, impairment < 0.82],
        [0.0, 0.5, 1.0],
        default=2.0,
    )
    cdr = np.clip(cdr + rng.choice([0, 0, 0, 0.5], rows, p=[0.74, 0.1, 0.1, 0.06]), 0, 3)

    memory_recall = np.clip(10 - 6.4 * impairment + rng.normal(0, 1.1, rows), 0, 10)
    orientation = np.clip(10 - 3.4 * impairment + rng.normal(0, 0.9, rows), 0, 10)
    daily_function = np.clip(10 - 4.9 * impairment + rng.normal(0, 1.0, rows), 0, 10)
    mood_change = np.clip(1.5 + 5.6 * impairment + rng.normal(0, 1.1, rows), 0, 10)
    sleep_quality = np.clip(8.6 - 3.5 * impairment + rng.normal(0, 1.25, rows), 0, 10)
    wandering = rng.binomial(1, np.clip(0.02 + 0.5 * impairment**2, 0, 0.75), rows)
    medication_adherence = np.clip(9.2 - 4.8 * impairment + rng.normal(0, 1.1, rows), 0, 10)
    amyloid_beta = np.clip(1050 - 380 * impairment + rng.normal(0, 95, rows), 280, 1350)
    tau = np.clip(210 + 360 * impairment + rng.normal(0, 80, rows), 80, 900)

    risk_score = (
        0.09 * (age - 65)
        - 0.1 * (education_years - 12)
        + 0.75 * family_history
        + 1.05 * apoe4
        - 0.38 * (mmse - 26)
        - 0.34 * (moca - 25)
        + 1.35 * cdr
        - 0.48 * (memory_recall - 7)
        - 0.24 * (orientation - 8)
        - 0.34 * (daily_function - 7)
        + 0.24 * mood_change
        - 0.13 * (sleep_quality - 6)
        + 0.82 * wandering
        - 0.24 * (medication_adherence - 7)
        + 0.35 * hypertension
        + 0.28 * diabetes
        - 0.0042 * (amyloid_beta - 850)
        + 0.004 * (tau - 320)
        + rng.normal(0, 1.15, rows)
    )

    low_threshold = np.quantile(risk_score, 0.43)
    high_threshold = np.quantile(risk_score, 0.77)
    risk_class = np.where(risk_score < low_threshold, 0, np.where(risk_score < high_threshold, 1, 2))

    data = pd.DataFrame(
        {
            "age": age.round(1),
            "education_years": education_years.round(1),
            "family_history": family_history,
            "apoe4": apoe4,
            "mmse": mmse.round(1),
            "moca": moca.round(1),
            "cdr": cdr,
            "memory_recall": memory_recall.round(1),
            "orientation": orientation.round(1),
            "daily_function": daily_function.round(1),
            "mood_change": mood_change.round(1),
            "sleep_quality": sleep_quality.round(1),
            "wandering": wandering,
            "medication_adherence": medication_adherence.round(1),
            "hypertension": hypertension,
            "diabetes": diabetes,
            "amyloid_beta": amyloid_beta.round(1),
            "tau": tau.round(1),
            "risk_class": risk_class,
        }
    )
    return data[FEATURE_NAMES + ["risk_class"]]
