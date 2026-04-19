# Online Deployment Guide

This application is ready to deploy as a single Python web service. The backend serves both the API and the frontend.

## Recommended: Render

1. Create a GitHub repository and upload this project.
2. Go to Render and create a new Blueprint or Web Service.
3. Select the GitHub repository.
4. Render can use `render.yaml` automatically.
5. Deploy.

The deployed app URL will look like:

```text
https://dementia-screening-xai.onrender.com
```

### Manual Render settings

If you create a manual Web Service instead of using `render.yaml`:

```text
Environment: Python
Build Command: pip install -r requirements.txt && python -m backend.ml.train
Start Command: python -m backend.start
```

Add this environment variable:

```text
HOST=0.0.0.0
```

Render provides the `PORT` variable automatically.

## Railway

1. Create a new Railway project from a GitHub repository.
2. Railway will detect the Python app.
3. Use this start command:

```text
python -m backend.start
```

The `Procfile` is included for platforms that read it automatically.

## Docker

The included `Dockerfile` can be used on any container platform.

```powershell
docker build -t dementia-screening-xai .
docker run -p 8000:8000 dementia-screening-xai
```

## Important note

The current model is trained from synthetic clinically inspired data during deployment. This is suitable for a prototype, academic demo, or research presentation. For real clinical deployment, replace the synthetic dataset with approved clinical data and complete validation.
