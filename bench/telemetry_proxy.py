#!/usr/bin/env python3
"""Transparent proxy between an agent and llama-server (or vLLM) that logs
per-request llama.cpp timings (vLLM: token counts from `usage`) to JSONL. This
records how prefill/decode speed and TTFT change as the agent's context grows
during real agentic runs.

  python bench/telemetry_proxy.py --listen 8081 --upstream http://127.0.0.1:8080 --log calls.jsonl
"""
import argparse
import http.client
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()


def record(path, req_body, status, timings, usage, wall_s, ttft_s, streamed):
    t = timings or {}
    row = {
        "ts": time.time(), "status": status, "stream": streamed, "wall_s": wall_s,
        "ttft_s": ttft_s,
        "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
        "prompt_ms": t.get("prompt_ms"), "prompt_tps": t.get("prompt_per_second"),
        "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
        "predicted_tps": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
        "usage": usage,
        "n_messages": len(req_body.get("messages", [])) if isinstance(req_body, dict) else None,
    }
    if t.get("prompt_n") is None and usage:  # vLLM: no timings; usage has the token counts
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        if usage.get("prompt_tokens") is not None:
            row["prompt_n"], row["cache_n"] = usage["prompt_tokens"] - cached, cached
        row["predicted_n"] = usage.get("completion_tokens")
    if row["prompt_n"] is not None:
        row["context_tokens"] = (row["prompt_n"] or 0) + (row["cache_n"] or 0)
    elif usage:
        row["context_tokens"] = usage.get("prompt_tokens")
    with LOCK, open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


class Handler(BaseHTTPRequestHandler):
    upstream = None
    log_path = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _forward(self, method):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            req_json = json.loads(body) if body else {}
        except ValueError:
            req_json = {}
        u = urllib.parse.urlparse(self.upstream)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=7200)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in {"host", "content-length", "connection"}}
        headers["Content-Length"] = str(len(body))
        t0 = time.perf_counter()
        conn.request(method, self.path, body=body, headers=headers)
        resp = conn.getresponse()
        streamed = "text/event-stream" in (resp.getheader("Content-Type") or "")
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in {"transfer-encoding", "content-length", "connection"}:
                self.send_header(k, v)
        timings = usage = ttft = None
        is_chat = self.path.rstrip("/").endswith("/chat/completions")
        if streamed:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            client_alive = True  # keep reading upstream for timings if the client hangs up early
            while True:
                line = resp.readline()
                if not line:
                    break
                if ttft is None and line.startswith(b"data:") and b'"delta"' in line:
                    ttft = time.perf_counter() - t0
                if line.startswith(b"data:") and (b'"timings"' in line or b'"usage"' in line):
                    try:
                        obj = json.loads(line[5:])
                        timings = obj.get("timings") or timings
                        usage = obj.get("usage") or usage
                    except ValueError:
                        pass
                client_alive = self._write(b"%x\r\n%s\r\n" % (len(line), line)) if client_alive else False
            if client_alive:
                self._write(b"0\r\n\r\n")
        else:
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            try:
                obj = json.loads(data)
                timings, usage = obj.get("timings"), obj.get("usage")
            except ValueError:
                pass
            if timings and timings.get("prompt_ms") is not None:
                ttft = timings["prompt_ms"] / 1000  # prefill time ~= TTFT for non-streamed calls
        wall = time.perf_counter() - t0
        conn.close()
        if is_chat:
            record(self.log_path, req_json, resp.status, timings, usage, wall, ttft, streamed)

    def _write(self, data):
        try:
            self.wfile.write(data)
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def do_POST(self):
        self._forward("POST")

    def do_GET(self):
        self._forward("GET")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8081)
    ap.add_argument("--upstream", default="http://127.0.0.1:8080")
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    Handler.upstream, Handler.log_path = a.upstream, a.log
    ThreadingHTTPServer(("127.0.0.1", a.listen), Handler).serve_forever()


if __name__ == "__main__":
    main()
