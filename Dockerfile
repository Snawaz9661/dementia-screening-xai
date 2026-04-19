FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m backend.ml.train

ENV HOST=0.0.0.0
EXPOSE 8000

CMD ["python", "-m", "backend.start"]
