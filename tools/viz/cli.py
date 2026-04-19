"""CLI entrypoint for the trade visualizer."""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from tools.viz.app import build_app
from tools.viz.parser import load_log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viz",
        description="Interactive visualizer for IMC Prosperity 4 trading logs "
        "(Rust backtester or Prosperity website submissions).",
    )
    parser.add_argument(
        "log_path",
        help="Path to a log directory (e.g. round1/logs/local-20260418-203355 "
        "or round1/logs/263711) or a single .log / .json file.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the browser.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    log_path = Path(args.log_path)
    try:
        log = load_log(log_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    _print_summary(log)

    app = build_app(log)

    url = f"http://{args.host}:{args.port}"
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    print(f"Serving at {url}  (Ctrl-C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def _print_summary(log) -> None:
    bits = [
        f"format={log.source}",
        f"run_id={log.run_id}",
        f"products={log.products}",
        f"days={log.days}",
        f"trades={log.trade_count} (own={log.own_trade_count})",
    ]
    if log.source == "platform":
        if log.meta.get("profit") is not None:
            bits.append(f"profit={log.meta['profit']:.2f}")
        if log.meta.get("status"):
            bits.append(f"status={log.meta['status']}")
    print("loaded: " + "  |  ".join(bits))


if __name__ == "__main__":
    raise SystemExit(main())
