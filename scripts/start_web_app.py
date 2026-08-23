"""Start the local web application and open it in the default browser."""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn

from web.server import app

HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}"


def server_is_ready() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=0.8):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def open_browser_when_ready() -> None:
    for _ in range(40):
        if server_is_ready():
            webbrowser.open(URL)
            return
        time.sleep(0.25)


if server_is_ready():
    webbrowser.open(URL)
else:
    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
