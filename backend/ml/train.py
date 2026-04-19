from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data_generator import generate_dataset
from .features import CLASS_LABELS, FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"


def one_hot(labels: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((labels.size, classes))
    matrix[np.arange(labels.size), labels] = 1.0
    return matrix


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def split_indices(total: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for label in np.unique(y_true):
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in np.unique(y_true):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float | str]]:
    rows = []
    for index, label in enumerate(CLASS_LABELS):
        tp = np.sum((y_true == index) & (y_pred == index))
        fp = np.sum((y_true != index) & (y_pred == index))
        fn = np.sum((y_true == index) & (y_pred != index))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "class": label,
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "support": int(np.sum(y_true == index)),
            }
        )
    return rows


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    return [
        [int(np.sum((y_true == row) & (y_pred == col))) for col in range(len(CLASS_LABELS))]
        for row in range(len(CLASS_LABELS))
    ]


def train() -> dict[str, object]:
    ARTIFACTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)

    dataset = generate_dataset()
    dataset.to_csv(DATA / "synthetic_dementia_screening.csv", index=False)

    x = dataset[FEATURE_NAMES].to_numpy(dtype=float)
    y = dataset["risk_class"].to_numpy(dtype=int)
    train_idx, val_idx, test_idx = split_indices(len(dataset))

    means = x[train_idx].mean(axis=0)
    stds = x[train_idx].std(axis=0)
    stds[stds == 0] = 1.0

    x_train = (x[train_idx] - means) / stds
    x_val = (x[val_idx] - means) / stds
    x_test = (x[test_idx] - means) / stds
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    rng = np.random.default_rng(12)
    weights = rng.normal(0, 0.04, (len(FEATURE_NAMES), len(CLASS_LABELS)))
    bias = np.zeros(len(CLASS_LABELS))
    y_encoded = one_hot(y_train, len(CLASS_LABELS))

    learning_rate = 0.075
    regularization = 0.0015
    for _ in range(2400):
        probabilities = softmax(x_train @ weights + bias)
        error = probabilities - y_encoded
        weights -= learning_rate * ((x_train.T @ error) / len(x_train) + regularization * weights)
        bias -= learning_rate * error.mean(axis=0)

    val_probabilities = softmax(x_val @ weights + bias)
    test_probabilities = softmax(x_test @ weights + bias)
    val_pred = np.argmax(val_probabilities, axis=1)
    test_pred = np.argmax(test_probabilities, axis=1)

    metrics = {
        "validation": {
            "accuracy": round(float(np.mean(val_pred == y_val)), 4),
            "balanced_accuracy": round(balanced_accuracy(y_val, val_pred), 4),
            "macro_f1": round(macro_f1(y_val, val_pred), 4),
        },
        "test": {
            "accuracy": round(float(np.mean(test_pred == y_test)), 4),
            "balanced_accuracy": round(balanced_accuracy(y_test, test_pred), 4),
            "macro_f1": round(macro_f1(y_test, test_pred), 4),
            "confusion_matrix": confusion_matrix(y_test, test_pred),
            "classification_report": classification_report(y_test, test_pred),
        },
    }

    model = {
        "feature_names": FEATURE_NAMES,
        "class_labels": CLASS_LABELS,
        "means": means.round(8).tolist(),
        "stds": stds.round(8).tolist(),
        "weights": weights.round(8).tolist(),
        "bias": bias.round(8).tolist(),
        "metrics": metrics,
        "training_note": "Synthetic clinically inspired data. Replace with approved clinical data before real-world use.",
    }

    (ARTIFACTS / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
