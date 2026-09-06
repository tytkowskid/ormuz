"""
Live agent view
---------------
A small Tornado server that streams the multi-agent trading pipeline to a
browser over a WebSocket, so you can watch the agents talk to each other
in real time.

Run it (from the project root):

    python server/live_server.py

then open http://127.0.0.1:8765 and press "Run".

The broker server (server/broker-server.py, port 5000) still needs to be
running if you want the trader's buy_stock / sell_stock calls to succeed.
Tornado is already a dependency, so this needs no extra installs.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
from pathlib import Path

import tornado.httpclient
import tornado.ioloop
import tornado.web
import tornado.websocket

# Allow `from src...` when run directly as `python server/live_server.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import streaming  # noqa: E402

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("LIVE_PORT", "8765"))
BROKER_URL = os.environ.get("BROKER_URL", "http://127.0.0.1:5000").rstrip("/")
AGENT_PORTFOLIO = "agent"          # src/tools.py trades through this portfolio_id
INITIAL_CASH = 100_000.0

_run_lock = threading.Lock()
_state = {"running": False}


# ---------------------------------------------------------------------------
# PIPELINE RUNNER
# ---------------------------------------------------------------------------

def _reset_agent_portfolio() -> None:
    """Wipe the agent portfolio's ledger so the Metrics tab reflects this run."""
    try:
        import requests

        requests.post(
            f"{BROKER_URL}/reset/{AGENT_PORTFOLIO}",
            json={"cash": INITIAL_CASH},
            timeout=5,
        )
        streaming.emit("metrics_reset", portfolio_id=AGENT_PORTFOLIO)
    except Exception as exc:  # noqa: BLE001 - broker may just not be running
        streaming.emit("error", agent="pipeline",
                       error=f"could not reset broker portfolio: {exc}")


def _run_pipeline_thread(ticker: str, start_date: str, num_days: int, model_name: str) -> None:
    from src.agents.orchestrator import run_pipeline

    _reset_agent_portfolio()
    try:
        asyncio.run(run_pipeline(ticker, start_date, num_days, model_name=model_name))
    except Exception as exc:  # noqa: BLE001 - surface everything to the UI
        streaming.emit("error", agent="pipeline", error=repr(exc))
    finally:
        streaming.emit("pipeline_stopped")
        with _run_lock:
            _state["running"] = False


# ---------------------------------------------------------------------------
# HTTP HANDLERS
# ---------------------------------------------------------------------------

class IndexHandler(tornado.web.RequestHandler):
    def get(self) -> None:
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.write((HERE / "live.html").read_text(encoding="utf-8"))


class RunHandler(tornado.web.RequestHandler):
    def post(self) -> None:
        try:
            body = json.loads(self.request.body or b"{}")
        except json.JSONDecodeError:
            body = {}

        ticker = (body.get("ticker") or "TSLA").strip().upper()
        start_date = (body.get("start_date") or "2019-06-01").strip()
        num_days = int(body.get("num_days") or 3)
        model_name = (body.get("model_name") or "openai:gpt-4.1-mini").strip()

        with _run_lock:
            if _state["running"]:
                self.set_status(409)
                self.write({"error": "a run is already in progress"})
                return
            _state["running"] = True

        streaming.clear_history()
        streaming.emit("run_requested", ticker=ticker, start_date=start_date,
                       num_days=num_days, model=model_name)

        threading.Thread(
            target=_run_pipeline_thread,
            args=(ticker, start_date, num_days, model_name),
            daemon=True,
        ).start()

        self.write({"status": "started"})


class MetricsHandler(tornado.web.RequestHandler):
    """Proxy the broker's /metrics/<portfolio_id> (same origin -> no CORS)."""

    async def get(self) -> None:
        pid = self.get_argument("portfolio_id", AGENT_PORTFOLIO)
        self.set_header("Content-Type", "application/json")
        try:
            resp = await tornado.httpclient.AsyncHTTPClient().fetch(
                f"{BROKER_URL}/metrics/{pid}", request_timeout=5, raise_error=False,
            )
            self.set_status(resp.code if resp.code < 600 else 502)
            self.write(resp.body or b"{}")
        except Exception as exc:  # noqa: BLE001
            self.set_status(502)
            self.write({"error": f"broker unreachable at {BROKER_URL}: {exc}"})


class EventSocket(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin: str) -> bool:  # noqa: ARG002 - local dev tool
        return True

    def open(self, *args, **kwargs) -> None:  # noqa: A003
        self.queue = streaming.subscribe()
        self.pump = tornado.ioloop.PeriodicCallback(self._drain, 100)
        self.pump.start()

    def _drain(self) -> None:
        try:
            while True:
                ev = self.queue.get_nowait()
                self.write_message(json.dumps(ev))
        except queue.Empty:
            pass
        except tornado.websocket.WebSocketClosedError:
            self.pump.stop()
        except Exception:
            pass

    def on_close(self) -> None:
        self.pump.stop()
        streaming.unsubscribe(self.queue)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def make_app() -> tornado.web.Application:
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/run", RunHandler),
        (r"/metrics", MetricsHandler),
        (r"/ws", EventSocket),
    ])


if __name__ == "__main__":
    make_app().listen(PORT)
    print(f"Live agent view:  http://127.0.0.1:{PORT}")
    tornado.ioloop.IOLoop.current().start()
