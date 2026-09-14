#!/usr/bin/env python3
"""Aggregate real-robot and simulation evidence into a reviewable knowledge ledger.

The ledger deliberately stores aggregate facts and references, not raw voice, text,
faces, or camera frames.  It is an evidence store for policy review; it does not
automatically overwrite the real robot policy.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
ARCHIVES = REPO_ROOT / "04-Archives"
PI_LOGS = ARCHIVES / "conversation-logs" / "pi"
SIM_RUNS = ARCHIVES / "sim-runs"
SIM_DATA = PROJECT_ROOT / "sim_data"
REAL_COMMANDS = PROJECT_ROOT / "config" / "real_commands.json"
KNOWLEDGE_DIR = ARCHIVES / "knowledge"
DB_PATH = KNOWLEDGE_DIR / "knowledge.db"
SUMMARY_PATH = KNOWLEDGE_DIR / "knowledge_summary.json"

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL CHECK(source IN ('real', 'sim')),
    kind TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'observed'
        CHECK(status IN ('observed', 'validated', 'proposed', 'promoted', 'rejected')),
    evidence TEXT,
    observed_at TEXT NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_facts_source_kind ON facts(source, kind);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def add_fact(
    conn: sqlite3.Connection,
    *,
    source: str,
    kind: str,
    fact_key: str,
    value: Any,
    confidence: float,
    status: str = "observed",
    evidence: str = "",
) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    fingerprint = hashlib.sha256(
        f"{source}|{kind}|{fact_key}|{payload}|{evidence}".encode("utf-8")
    ).hexdigest()
    conn.execute(
        """INSERT OR IGNORE INTO facts
        (source, kind, fact_key, value_json, confidence, status, evidence, observed_at, fingerprint)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (source, kind, fact_key, payload, max(0.0, min(1.0, confidence)), status, evidence, now(), fingerprint),
    )


def jsonl_files() -> Iterable[Path]:
    if not PI_LOGS.exists():
        return ()
    return PI_LOGS.rglob("events.jsonl")


def collect_real(conn: sqlite3.Connection) -> dict[str, Any]:
    event_counts: collections.Counter[str] = collections.Counter()
    outcome_counts: collections.Counter[str] = collections.Counter()
    command_counts: collections.Counter[str] = collections.Counter()
    total = 0
    for path in jsonl_files():
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            total += 1
            kind = str(event.get("type") or event.get("event") or "unknown")[:80]
            event_counts[kind] += 1
            outcome = event.get("outcome") or event.get("status")
            if outcome:
                outcome_counts[str(outcome)[:80]] += 1
            command = event.get("command") or event.get("text")
            # Keep only a digest.  Raw conversation and names stay in the existing archive.
            if isinstance(command, str) and command.strip():
                normalized = re.sub(r"\s+", " ", command).strip().lower()
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
                command_counts[digest] += 1

    for kind, count in event_counts.items():
        add_fact(conn, source="real", kind="event_count", fact_key=kind,
                 value={"count": count}, confidence=min(0.99, 0.5 + count / 100.0),
                 evidence=str(PI_LOGS))
    for outcome, count in outcome_counts.items():
        add_fact(conn, source="real", kind="outcome_count", fact_key=outcome,
                 value={"count": count}, confidence=min(0.99, 0.5 + count / 100.0),
                 evidence=str(PI_LOGS))
    for digest, count in command_counts.items():
        add_fact(conn, source="real", kind="command_digest", fact_key=digest,
                 value={"count": count}, confidence=min(0.95, 0.5 + count / 50.0),
                 evidence=str(REAL_COMMANDS))

    persons = {}
    db_path = ARCHIVES / "person-dataset" / "persons" / "persons.db"
    if db_path.exists():
        try:
            db = sqlite3.connect(db_path)
            persons["people"] = db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            persons["boxes"] = db.execute("SELECT COUNT(*) FROM person_boxes").fetchone()[0]
            db.close()
        except sqlite3.Error:
            persons = {}
    if persons:
        add_fact(conn, source="real", kind="person_dataset", fact_key="aggregate",
                 value=persons, confidence=0.8, evidence=str(db_path))
    return {"events": total, "event_kinds": dict(event_counts),
            "outcomes": dict(outcome_counts), "command_digests": len(command_counts),
            "person_dataset": persons}


def improve_files() -> list[Path]:
    files = set(SIM_DATA.glob("improve-*.json")) if SIM_DATA.exists() else set()
    if SIM_RUNS.exists():
        files.update(SIM_RUNS.rglob("improve-*.json"))
    return sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0)


def collect_sim(conn: sqlite3.Connection) -> dict[str, Any]:
    runs = 0
    latest: dict[str, Any] | None = None
    causes: collections.Counter[str] = collections.Counter()
    for path in improve_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        runs += 1
        best = data.get("best_ever") or data.get("best") or {}
        if not isinstance(best, dict):
            best = {}
        metrics = best.get("metrics") if isinstance(best.get("metrics"), dict) else best
        value = {
            "quality": metrics.get("quality"), "success": metrics.get("success"),
            "total": metrics.get("total"), "collision": metrics.get("collision"),
            "params": best.get("params", data.get("params")),
        }
        status = "proposed" if value.get("params") else "observed"
        add_fact(conn, source="sim", kind="policy_candidate", fact_key=str(path),
                 value=value, confidence=0.9 if value.get("collision", 1) == 0 else 0.5,
                 status=status, evidence=str(path))
        history = data.get("history") or []
        if isinstance(history, list):
            for row in history:
                if isinstance(row, dict):
                    for cause in row.get("causes", []) if isinstance(row.get("causes"), list) else []:
                        causes[str(cause)[:100]] += 1
        if path.exists() and (latest is None or path.stat().st_mtime > latest.get("mtime", 0)):
            latest = {"path": str(path), "mtime": path.stat().st_mtime, **value}
    for cause, count in causes.items():
        add_fact(conn, source="sim", kind="failure_cause", fact_key=cause,
                 value={"count": count}, confidence=min(0.95, 0.5 + count / 20.0),
                 evidence=str(SIM_DATA))
    if latest:
        latest = {k: v for k, v in latest.items() if k != "mtime"}
        add_fact(conn, source="sim", kind="latest_metrics", fact_key="latest",
                 value=latest, confidence=0.85, evidence=latest["path"])
    return {"runs": runs, "latest": latest, "failure_causes": dict(causes)}


def sync() -> dict[str, Any]:
    conn = connect()
    real = collect_real(conn)
    sim = collect_sim(conn)
    conn.commit()
    row = conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()
    summary = {
        "schema_version": 1,
        "generated_at": now(),
        "facts": row["n"],
        "real": real,
        "sim": sim,
        "policy": {
            "automatic_promotion": False,
            "note": "sim 후보는 검토 후에만 config/behavior_real.json으로 승격; 원본 개인정보는 원장에 복사하지 않음",
        },
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    conn.close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true", help="collect current real/sim evidence")
    parser.add_argument("--stats", action="store_true", help="print the generated summary")
    args = parser.parse_args()
    summary = sync() if args.sync or not args.stats else json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
