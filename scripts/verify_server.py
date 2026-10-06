"""Start a hidden local Streamlit server, check HTTP health, then stop our process."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Choose an available ephemeral port so a user's existing server is untouched.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_path = ROOT / "results/streamlit_server.log"
    url = f"http://127.0.0.1:{port}"
    print(f"Checking Streamlit on {url}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
             "--server.address", "127.0.0.1", "--server.port", str(port),
             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        try:
            start = time.monotonic()
            ready = False
            session = requests.Session()
            session.trust_env = False  # Local health must not be routed through a proxy.
            while time.monotonic() - start < 40:
                if process.poll() is not None:
                    break
                try:
                    health = session.get(url + "/_stcore/health", timeout=2)
                    page = session.get(url, timeout=2)
                    if health.status_code == 200 and health.text == "ok" and page.status_code == 200:
                        ready = True
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.4)
            if not ready:
                raise RuntimeError(f"Streamlit failed to become healthy. Read {log_path}.")
            record = {"health_status": health.status_code, "health_body": health.text,
                      "page_status": page.status_code, "startup_seconds": time.monotonic() - start,
                      "port": port, "scope": "HTTP startup only; app interactions covered by pytest AppTest"}
            (ROOT / "results/server_verification.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(json.dumps(record, indent=2), flush=True)
            return 0
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
