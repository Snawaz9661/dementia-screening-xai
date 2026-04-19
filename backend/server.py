from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .ml.predictor import DementiaRiskPredictor, feature_info
from .report import build_pdf_report

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
REPORTS = ROOT / "reports"


class AppState:
    predictor: DementiaRiskPredictor | None = None


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict | list) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class DementiaScreeningHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            return json_response(self, 200, {"status": "ok", "service": "dementia-screening"})
        if path == "/api/feature-info":
            return json_response(self, 200, feature_info())
        if path.startswith("/reports/"):
            return self._serve_report(path)
        return self._serve_static(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/predict", "/api/report"}:
            return json_response(self, 404, {"error": "Not found"})

        try:
            payload = self._read_json()
            predictor = self._predictor()
            result = predictor.predict(payload)
            if parsed.path == "/api/report":
                report_path = build_pdf_report(result, payload.get("respondent", "Not specified"))
                result["reportUrl"] = f"/reports/{report_path.name}"
            return json_response(self, 200, result)
        except FileNotFoundError as exc:
            return json_response(self, 503, {"error": str(exc), "hint": "Run training before starting the app."})
        except Exception as exc:
            return json_response(self, 400, {"error": str(exc)})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw or "{}")

    def _predictor(self) -> DementiaRiskPredictor:
        if AppState.predictor is None:
            AppState.predictor = DementiaRiskPredictor()
        return AppState.predictor

    def _serve_report(self, path: str) -> None:
        filename = Path(unquote(path)).name
        report = REPORTS / filename
        if not report.exists() or report.suffix.lower() != ".pdf":
            return json_response(self, 404, {"error": "Report not found"})
        data = report.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            target = FRONTEND / "index.html"
        else:
            target = FRONTEND / unquote(path.lstrip("/"))
        try:
            target = target.resolve()
            if FRONTEND.resolve() not in target.parents and target != FRONTEND.resolve():
                raise ValueError("Invalid path")
            if not target.exists() or target.is_dir():
                target = FRONTEND / "index.html"
            data = target.read_bytes()
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            json_response(self, 404, {"error": "Static file not found"})


def run(host: str | None = None, port: int | None = None) -> None:
    host = host or os.environ.get("HOST", "127.0.0.1")
    port = port or int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), DementiaScreeningHandler)
    print(f"Serving dementia screening app at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
