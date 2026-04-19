from __future__ import annotations

import os
from pathlib import Path

from .ml.train import train
from .server import run

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.json"


def main() -> None:
    if not MODEL_PATH.exists():
        print("Model artifact not found. Training model before startup...")
        train()
    run(host=os.environ.get("HOST", "0.0.0.0"))


if __name__ == "__main__":
    main()
