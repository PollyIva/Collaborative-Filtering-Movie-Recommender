#!/usr/bin/env python3
"""
experiments/run.py — drive the shipped app in headless Chrome and run
experiments/measure.js against it.

The point of routing this through a real browser rather than a Python
reimplementation is fidelity: the CF code that runs here is the exact
data.js + script.js that ships, so no result can come from a lookalike.

    python experiments/run.py                  # defaults
    python experiments/run.py --users 50       # quicker smoke run

Writes experiments/results.json and prints a summary.
"""

import argparse
import functools
import http.server
import json
import pathlib
import socket
import subprocess
import sys
import threading
import time

import websocket  # websocket-client

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve():
    port = free_port()
    handler = functools.partial(Quiet, directory=str(ROOT))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def wait_for_target(port, timeout=45):
    """Poll the DevTools HTTP endpoint until the page target is attached."""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as r:
                for t in json.load(r):
                    if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                        return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.4)
    raise RuntimeError("Chrome DevTools target never appeared")


class CDP:
    def __init__(self, url):
        self.ws = websocket.create_connection(url, timeout=1800)
        self.next_id = 0

    def call(self, method, params=None, on_event=None):
        self.next_id += 1
        mid = self.next_id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            if on_event and "method" in msg:
                on_event(msg)

    def evaluate(self, expression, await_promise=False, on_event=None):
        r = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
            on_event=on_event,
        )
        if r.get("exceptionDetails"):
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:2000])
        return r["result"].get("value")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=250)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rel", type=int, default=4)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--cost-users", type=int, default=40)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    if not pathlib.Path(CHROME).exists():
        sys.exit(f"Chrome not found at {CHROME}")

    httpd, http_port = serve()
    dbg_port = free_port()
    profile = HERE / ".chrome-profile"

    url = f"http://127.0.0.1:{http_port}/index.html"
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={dbg_port}",
            "--remote-allow-origins=*",
            # do not let headless throttle the timers the chunked precompute relies on
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--disable-features=CalculateNativeWinOcclusion",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        ws_url = wait_for_target(dbg_port)
        cdp = CDP(ws_url)
        cdp.call("Runtime.enable")

        print("waiting for the app to finish loading + precompute…", flush=True)
        deadline = time.time() + 300
        while time.time() < deadline:
            if cdp.evaluate("typeof isReady !== 'undefined' && isReady === true"):
                break
            time.sleep(1)
        else:
            sys.exit("app never became ready")
        print("app ready\n", flush=True)

        measure = (HERE / "measure.js").read_text(encoding="utf-8")
        cfg = {
            "k": args.k,
            "relThreshold": args.rel,
            "users": args.users,
            "holdoutFrac": args.holdout,
            "seed": args.seed,
            "costUsers": args.cost_users,
        }
        expr = measure + "\nrunMeasurements(" + json.dumps(cfg) + ")"

        def on_event(msg):
            if msg["method"] == "Runtime.consoleAPICalled":
                text = " ".join(
                    str(a.get("value", a.get("description", "")))
                    for a in msg["params"].get("args", [])
                )
                if text.startswith("[measure]"):
                    print(text, flush=True)

        t0 = time.time()
        res = cdp.evaluate(expr, await_promise=True, on_event=on_event)
        print(f"\nfinished in {time.time() - t0:.0f}s", flush=True)

        res["wallClockSeconds"] = round(time.time() - t0, 1)
        pathlib.Path(args.out).write_text(
            json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        d, p, a, n, c = (
            res["dataset"],
            res["pairSpace"],
            res["agreement"],
            res["ndcg"],
            res["cost"],
        )
        print(f"\nwrote {args.out}")
        print(f"  dataset      {d['users']}u {d['movies']}i {d['ratings']}r  {d['emptyPct']:.2f}% empty")
        print(f"  user pairs   {p['user']['pairs']:,}  clearing floor {p['user']['clearing']:,}")
        print(f"  item pairs   {p['item']['pairs']:,}  clearing floor {p['item']['clearing']:,}")
        print(f"  overlap/user {a['meanSharedPerUser']:.4f} vs {a['expectedSharedPerUser']:.4f} by chance")
        print(f"  cost         precompute {c['precomputeMs']:.0f}ms  ub {c['userQueryMsMedian']:.2f}ms  ib {c['itemQueryMsMedian']:.2f}ms")
        print(f"  NDCG@{args.k}       ub {n['mean']['userBased']:.4f}  ib {n['mean']['itemBased']:.4f}"
              f"  ib-unguarded {n['mean']['itemBasedUnguarded']:.4f}  random {n['mean']['random']:.4f}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        httpd.shutdown()


if __name__ == "__main__":
    main()
