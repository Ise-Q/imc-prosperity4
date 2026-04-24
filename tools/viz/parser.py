"""Log file parsing for IMC Prosperity 4 trade visualizer.

Handles two formats:
  - "local": single `backtest.log` emitted by the Rust backtester. Three sections
    separated by exact header lines: "Sandbox logs:", "Activities log:",
    "Trade History:".
  - "platform": `<id>.json` + `<id>.log` pair from the Prosperity website.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

LOCAL_HEADERS = ("Sandbox logs:", "Activities log:", "Trade History:")

_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


@dataclass
class LoadedLog:
    activities: pd.DataFrame
    trades: pd.DataFrame
    sandbox: pd.DataFrame
    products: list[str]
    days: list[int]
    source: Literal["local", "platform"]
    run_id: str
    meta: dict = field(default_factory=dict)

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def own_trade_count(self) -> int:
        if self.trades.empty:
            return 0
        return int(self.trades["is_own"].sum())


def load_log(log_path: str | Path) -> LoadedLog:
    path = Path(log_path)
    if path.is_dir():
        return _load_from_dir(path)
    if path.is_file():
        return _load_from_file(path)
    raise FileNotFoundError(f"{path} does not exist")


def _load_from_dir(path: Path) -> LoadedLog:
    backtest = path / "backtest.log"
    if backtest.is_file():
        return _parse_local(backtest, run_id=path.name)

    json_files = sorted(path.glob("*.json"))
    log_files = sorted(path.glob("*.log"))
    if json_files:
        json_path = json_files[0]
        log_path = log_files[0] if log_files else None
        return _parse_platform(json_path, log_path, run_id=path.name)

    raise FileNotFoundError(
        f"No recognized log files in {path} "
        "(expected backtest.log or <id>.json)"
    )


def _load_from_file(path: Path) -> LoadedLog:
    if path.suffix == ".json":
        sibling_log = path.with_suffix(".log")
        log_path = sibling_log if sibling_log.is_file() else None
        return _parse_platform(path, log_path, run_id=path.parent.name)

    text_head = path.read_text(errors="replace")[:100]
    if "Sandbox logs:" in text_head or "Activities log:" in text_head:
        return _parse_local(path, run_id=path.parent.name)

    if path.suffix == ".log":
        try:
            with path.open() as f:
                d = json.load(f)
            if "activitiesLog" in d or "tradeHistory" in d:
                sibling_json = path.with_suffix(".json")
                json_path = sibling_json if sibling_json.is_file() else None
                return _parse_platform(json_path, path, run_id=path.parent.name)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not determine log format for {path}")


def _parse_local(path: Path, run_id: str) -> LoadedLog:
    text = path.read_text()
    sections = _split_local_sections(text)

    sandbox = _parse_local_sandbox(sections.get("Sandbox logs:", ""))
    activities = _parse_activities_csv(sections.get("Activities log:", ""))
    trades = _parse_local_trades(sections.get("Trade History:", ""))

    activities = _add_global_ts(activities)
    trades = _enrich_trades(trades, activities)

    return LoadedLog(
        activities=activities,
        trades=trades,
        sandbox=sandbox,
        products=sorted(activities["product"].unique().tolist()),
        days=sorted(activities["day"].unique().tolist()),
        source="local",
        run_id=run_id,
        meta={"file_size": path.stat().st_size, "path": str(path)},
    )


def _split_local_sections(text: str) -> dict[str, str]:
    positions: list[tuple[int, str]] = []
    for header in LOCAL_HEADERS:
        idx = text.find(header + "\n")
        if idx == -1:
            idx = text.find(header)
        if idx != -1:
            positions.append((idx, header))
    positions.sort()

    out: dict[str, str] = {}
    for i, (start, header) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body_start = start + len(header)
        body = text[body_start:end].lstrip("\n")
        out[header] = body.rstrip()
    return out


def _parse_local_sandbox(body: str) -> pd.DataFrame:
    body = body.strip()
    if not body:
        return pd.DataFrame(columns=["timestamp", "sandboxLog", "lambdaLog"])

    decoder = json.JSONDecoder()
    records: list[dict] = []
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in " \n\r\t":
            i += 1
        if i >= n:
            break
        try:
            obj, offset = decoder.raw_decode(body, i)
        except json.JSONDecodeError:
            break
        records.append(obj)
        i = offset
    if not records:
        return pd.DataFrame(columns=["timestamp", "sandboxLog", "lambdaLog"])
    df = pd.DataFrame(records)
    keep = [c for c in ("timestamp", "sandboxLog", "lambdaLog") if c in df.columns]
    return df[keep]


def _parse_local_trades(body: str) -> pd.DataFrame:
    body = body.strip()
    if not body:
        return _empty_trades()
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", body)
    records = json.loads(cleaned)
    if not records:
        return _empty_trades()
    return pd.DataFrame(records)


def _parse_platform(json_path: Path | None, log_path: Path | None, run_id: str) -> LoadedLog:
    if json_path is None and log_path is None:
        raise ValueError("platform format requires at least a .json or .log file")

    results: dict = {}
    execution: dict = {}
    if json_path is not None:
        with json_path.open() as f:
            results = json.load(f)
    if log_path is not None:
        with log_path.open() as f:
            execution = json.load(f)

    activities_csv = results.get("activitiesLog") or execution.get("activitiesLog") or ""
    activities = _parse_activities_csv(activities_csv)
    activities = _add_global_ts(activities)

    trade_history = execution.get("tradeHistory", []) or []
    if trade_history:
        trades = pd.DataFrame(trade_history)
    else:
        trades = _empty_trades()
    trades = _enrich_trades(trades, activities)

    sandbox_records = execution.get("logs") or []
    if sandbox_records and isinstance(sandbox_records, list):
        sandbox = pd.DataFrame(sandbox_records)
        keep = [c for c in ("timestamp", "sandboxLog", "lambdaLog") if c in sandbox.columns]
        sandbox = sandbox[keep] if keep else pd.DataFrame(columns=["timestamp", "sandboxLog", "lambdaLog"])
    else:
        sandbox = pd.DataFrame(columns=["timestamp", "sandboxLog", "lambdaLog"])

    meta = {
        "round": results.get("round"),
        "status": results.get("status"),
        "profit": results.get("profit"),
        "submissionId": execution.get("submissionId"),
    }

    return LoadedLog(
        activities=activities,
        trades=trades,
        sandbox=sandbox,
        products=sorted(activities["product"].unique().tolist()) if not activities.empty else [],
        days=sorted(activities["day"].unique().tolist()) if not activities.empty else [],
        source="platform",
        run_id=run_id,
        meta=meta,
    )


def _parse_activities_csv(csv_text: str) -> pd.DataFrame:
    if not csv_text.strip():
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(csv_text), sep=";")
    df["day"] = df["day"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)
    df["product"] = df["product"].astype(str)

    numeric_cols = [c for c in df.columns if c not in ("day", "timestamp", "product")]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _add_global_ts(activities: pd.DataFrame) -> pd.DataFrame:
    """Timestamps from the backtester are already globally unique across days —
    each day occupies a consecutive range (e.g. day -2: 0..999900, day -1:
    1000000..1999900, day 0: 2000000..2999900). So global_ts == timestamp.
    """
    if activities.empty:
        return activities
    activities = activities.copy()
    activities["global_ts"] = activities["timestamp"].astype(int)
    return activities


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp", "buyer", "seller", "symbol", "currency",
            "price", "quantity", "side", "signed_qty", "is_own",
            "day", "global_ts",
        ]
    )


def _enrich_trades(trades: pd.DataFrame, activities: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return _empty_trades()

    for col in ("buyer", "seller"):
        if col not in trades.columns:
            trades[col] = ""
        trades[col] = trades[col].fillna("").astype(str)

    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0).astype(int)
    trades["timestamp"] = pd.to_numeric(trades["timestamp"], errors="coerce").fillna(0).astype(int)

    def _side(r):
        if r["buyer"] == "SUBMISSION":
            return "BUY"
        if r["seller"] == "SUBMISSION":
            return "SELL"
        return "MARKET"

    trades["side"] = trades.apply(_side, axis=1)
    trades["signed_qty"] = np.where(
        trades["side"] == "BUY", trades["quantity"],
        np.where(trades["side"] == "SELL", -trades["quantity"], 0),
    )
    trades["is_own"] = trades["side"] != "MARKET"

    trades = trades.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    trades["global_ts"] = trades["timestamp"].astype(int)

    if "day" not in trades.columns:
        trades["day"] = _assign_trade_days(trades["timestamp"], activities)

    return trades


def _assign_trade_days(trade_ts: pd.Series, activities: pd.DataFrame) -> pd.Series:
    """Map trade timestamps to the day whose activities range contains them."""
    if activities.empty:
        return pd.Series(0, index=trade_ts.index, dtype=int)

    bounds = (
        activities.groupby("day")["timestamp"]
        .agg(["min", "max"])
        .sort_values("min")
        .reset_index()
    )
    edges = bounds["min"].tolist() + [bounds["max"].iloc[-1] + 1]
    labels = bounds["day"].tolist()
    assigned = pd.cut(
        trade_ts,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return assigned.astype(int)
