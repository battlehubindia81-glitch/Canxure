
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.real_data_api import create_api


BASE_DIR = "/content/drive/MyDrive/Canxure_backup"
PROJECT_DIR = "/content/Canxure"

SERVICE = create_api(
    base_dir=BASE_DIR,
    project_dir=PROJECT_DIR,
    checkpoint_name="epoch_015.pt",
)


def _json_response(handler, payload, status=200):
    body = json.dumps(payload).encode("utf-8")

    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header(
        "Access-Control-Allow-Methods",
        "GET, POST, OPTIONS",
    )
    handler.end_headers()
    handler.wfile.write(body)


class CanxureAPIHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Keep server output clean during notebook tests.
        return

    def do_OPTIONS(self):
        _json_response(self, {"status": "ok"})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/health":
                _json_response(
                    self,
                    SERVICE.health(),
                )
                return

            if path == "/samples":
                _json_response(
                    self,
                    SERVICE.list_samples(),
                )
                return

            if path == "/drugs":
                _json_response(
                    self,
                    SERVICE.list_drugs(),
                )
                return

            if path == "/predict":
                sample_id = query.get("sample_id", [None])[0]
                drug_id = query.get("drug_id", [None])[0]

                if not sample_id or not drug_id:
                    _json_response(
                        self,
                        {
                            "error": (
                                "sample_id and drug_id are required"
                            )
                        },
                        status=400,
                    )
                    return

                result = SERVICE.predict_pair(
                    sample_id=sample_id,
                    drug_id=drug_id,
                )

                _json_response(self, result)
                return

            _json_response(
                self,
                {"error": "Not found"},
                status=404,
            )

        except ValueError as exc:
            _json_response(
                self,
                {"error": str(exc)},
                status=400,
            )
        except Exception as exc:
            _json_response(
                self,
                {"error": "Internal server error"},
                status=500,
            )

    def do_POST(self):
        parsed = urlparse(self.path)

        try:
            if parsed.path != "/predict":
                _json_response(
                    self,
                    {"error": "Not found"},
                    status=404,
                )
                return

            length = int(
                self.headers.get("Content-Length", "0")
            )

            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))

            sample_id = payload.get("sample_id")
            drug_ids = payload.get("drug_ids")

            if not sample_id:
                _json_response(
                    self,
                    {"error": "sample_id is required"},
                    status=400,
                )
                return

            if not isinstance(drug_ids, list) or not drug_ids:
                _json_response(
                    self,
                    {"error": "drug_ids must be a non-empty list"},
                    status=400,
                )
                return

            results = SERVICE.predict_drugs(
                sample_id=sample_id,
                drug_ids=drug_ids,
            )

            _json_response(
                self,
                {
                    "sample_id": sample_id,
                    "results": results,
                    "count": len(results),
                    "checkpoint_epoch": 15,
                },
            )

        except ValueError as exc:
            _json_response(
                self,
                {"error": str(exc)},
                status=400,
            )
        except Exception:
            _json_response(
                self,
                {"error": "Internal server error"},
                status=500,
            )


def create_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer(
        (host, port),
        CanxureAPIHandler,
    )

if __name__ == "__main__":
    server = create_server()
    print(
        "Canxure API server starting on http://127.0.0.1:8000",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCanxure API server stopping...", flush=True)
    finally:
        server.server_close()

