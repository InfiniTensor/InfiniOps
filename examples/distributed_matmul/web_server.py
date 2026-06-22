#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "examples" / "distributed_matmul"
RUNNER = DEMO_DIR / "run_remote.sh"
SOURCE = DEMO_DIR / "distributed_matmul.cc"
INDEX = DEMO_DIR / "showcase.html"

PLATFORMS = {"nvidia", "metax", "iluvatar", "moore", "cambricon", "ascend"}


def json_response(handler, status, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    server_version = "InfiniOpsDemo/1.0"

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/showcase.html"):
            data = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/source":
            text = SOURCE.read_text(encoding="utf-8")
            json_response(self, 200, {"source": text})
            return
        json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            json_response(self, 404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            json_response(self, 400, {"error": "invalid json"})
            return

        platform = str(payload.get("platform", "nvidia"))
        if platform not in PLATFORMS:
            json_response(self, 400, {"error": "invalid platform"})
            return

        def clean_int(name, default, low, high):
            try:
                value = int(payload.get(name, default))
            except (TypeError, ValueError):
                value = default
            return max(low, min(high, value))

        env = os.environ.copy()
        env.update({
            "NP": str(clean_int("np", 2, 1, 8)),
            "ROWS": str(clean_int("rows", 1024, 1, 8192)),
            "K": str(clean_int("k", 2048, 1, 8192)),
            "N": str(clean_int("n", 1024, 1, 8192)),
        })
        if "remoteRoot" in payload and payload["remoteRoot"]:
            env["REMOTE_ROOT"] = str(payload["remoteRoot"])

        cmd = [str(RUNNER), platform]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=int(os.environ.get("DEMO_RUN_TIMEOUT", "900")),
            )
            output = proc.stdout
            lines = [line for line in output.splitlines() if "max_error=" in line or "global_shape=" in line or "failed:" in line]
            json_response(self, 200, {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "command": " ".join([f"{k}={env[k]}" for k in ("NP", "ROWS", "K", "N")]) + " " + " ".join(cmd),
                "summary": lines[-3:],
                "output": output,
            })
        except subprocess.TimeoutExpired as exc:
            json_response(self, 200, {
                "ok": False,
                "returncode": None,
                "command": " ".join(cmd),
                "summary": ["run timed out"],
                "output": (exc.stdout or "") + "\n[timeout]",
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving distributed matmul demo at http://{args.host}:{args.port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
