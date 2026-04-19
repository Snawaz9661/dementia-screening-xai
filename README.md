# Explainable AI Dementia Screening Application

This project is a full local prototype for early-stage dementia screening using behavioral and clinical indicators. It includes a dependency-light backend, a NumPy-based ML model, explainable prediction output, a browser frontend, and PDF report generation.

## What is included

- Adaptive screening form for patient, caregiver, or clinician entry
- Clinical, behavioral, demographic, history, and biomarker inputs
- Synthetic-data training pipeline for a three-class risk model
- Explainable per-person factor contributions
- PDF screening report with result, top factors, input summary, and next steps
- Static frontend served by the backend

## Important clinical note

This is a screening and research prototype, not a diagnostic medical device. Replace the synthetic training data with approved clinical datasets and complete clinical validation before real-world use.

## Run locally

Use the bundled Python runtime shown by Codex, or any Python environment with `numpy`, `pandas`, and `reportlab`.

```powershell
& 'C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m backend.ml.train
& 'C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m backend.server
```

Then open:

```text
http://127.0.0.1:8000
```

## Deploy online

This repo includes deployment files for Render, Railway, and Docker:

- `render.yaml`
- `Procfile`
- `Dockerfile`
- `requirements.txt`
- `runtime.txt`

Recommended Render settings:

```text
Build Command: pip install -r requirements.txt && python -m backend.ml.train
Start Command: python -m backend.start
```

See `DEPLOYMENT.md` for the full step-by-step guide.

## API

- `GET /api/health`
- `GET /api/feature-info`
- `POST /api/predict`
- `POST /api/report`

Example payload:

```json
{
  "respondent": "Caregiver",
  "age": 74,
  "education_years": 10,
  "family_history": 1,
  "apoe4": 1,
  "mmse": 22,
  "moca": 19,
  "cdr": 0.5,
  "memory_recall": 3,
  "orientation": 7,
  "daily_function": 5,
  "mood_change": 6,
  "sleep_quality": 4,
  "wandering": 1,
  "medication_adherence": 5,
  "hypertension": 1,
  "diabetes": 0,
  "amyloid_beta": 620,
  "tau": 540
}
```
