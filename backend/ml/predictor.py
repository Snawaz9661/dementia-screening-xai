from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .features import CLASS_LABELS, DEFAULTS, DIRECTIONS, FEATURE_LABELS, FEATURE_NAMES, FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "model.json"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact not found. Run backend/ml/train.py first.")
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def sanitize_payload(payload: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in FEATURE_NAMES:
        raw = payload.get(name, DEFAULTS[name])
        if raw is None or raw == "":
            raw = DEFAULTS[name]
        try:
            values[name] = float(raw)
        except (TypeError, ValueError):
            values[name] = float(DEFAULTS[name])
    return values


class DementiaRiskPredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = _load_model()
        self.means = np.array(self.model["means"], dtype=float)
        self.stds = np.array(self.model["stds"], dtype=float)
        self.weights = np.array(self.model["weights"], dtype=float)
        self.bias = np.array(self.model["bias"], dtype=float)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        values = sanitize_payload(payload)
        x = np.array([values[name] for name in FEATURE_NAMES], dtype=float)
        z = (x - self.means) / self.stds
        probabilities = _softmax(z @ self.weights + self.bias)
        predicted_index = int(np.argmax(probabilities))
        high_risk_score = float(round(probabilities[2] * 100, 1))

        contributions = self._explain(z, predicted_index)
        summary = self._plain_language_summary(predicted_index, contributions)

        return {
            "riskLabel": CLASS_LABELS[predicted_index],
            "riskClass": predicted_index,
            "riskScore": high_risk_score,
            "probabilities": {
                CLASS_LABELS[index]: round(float(probabilities[index]), 4)
                for index in range(len(CLASS_LABELS))
            },
            "topFactors": contributions[:6],
            "summary": summary,
            "input": values,
            "disclaimer": (
                "This screening output is not a diagnosis. It should be reviewed with a qualified clinician, "
                "especially when risk is moderate or high."
            ),
        }

    def _explain(self, z: np.ndarray, predicted_index: int) -> list[dict[str, Any]]:
        baseline = int(np.argmin(np.abs(self.bias - np.median(self.bias))))
        contrast = self.weights[:, predicted_index] - self.weights[:, baseline]
        raw_contributions = z * contrast
        order = np.argsort(np.abs(raw_contributions))[::-1]

        explanations = []
        for feature_index in order:
            name = FEATURE_NAMES[int(feature_index)]
            contribution = float(raw_contributions[int(feature_index)])
            direction = "increased" if contribution > 0 else "reduced"
            explanations.append(
                {
                    "feature": name,
                    "label": FEATURE_LABELS[name],
                    "value": round(float(z[feature_index] * self.stds[feature_index] + self.means[feature_index]), 2),
                    "contribution": round(contribution, 3),
                    "direction": direction,
                    "text": self._factor_text(name, contribution),
                }
            )
        return explanations

    def _factor_text(self, feature: str, contribution: float) -> str:
        label = FEATURE_LABELS[feature]
        expected = DIRECTIONS[feature]
        if contribution > 0:
            if expected == "lower":
                return f"{label} is lower than expected and pushed the risk estimate upward."
            return f"{label} is elevated or present and pushed the risk estimate upward."
        if expected == "lower":
            return f"{label} is relatively preserved and reduced the risk estimate."
        return f"{label} is lower or absent and reduced the risk estimate."

    def _plain_language_summary(self, predicted_index: int, factors: list[dict[str, Any]]) -> str:
        strongest = factors[0]["label"] if factors else "the screening profile"
        if predicted_index == 0:
            return f"The profile is currently closest to the low-risk group. {strongest} had the largest influence."
        if predicted_index == 1:
            return f"The profile falls in a moderate-risk zone where follow-up screening is recommended. {strongest} was the strongest signal."
        return f"The profile shows several indicators associated with elevated dementia risk. {strongest} was the strongest signal."


def feature_info() -> list[dict[str, Any]]:
    return FEATURES
