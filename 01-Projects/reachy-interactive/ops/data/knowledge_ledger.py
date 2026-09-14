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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import para  # PARA 기준 경로 - 상대 깊이 셈으로 가짜 보관고를 만든 사고 재발 방지

PROJECT_ROOT = Path(para.PROJECT)
ARCHIVES = Path(para.ARCHIVES)
PI_LOGS = Path(para.PI_LOGS)
SIM_RUNS = Path(para.SIM_RUNS)
SIM_DATA = PROJECT_ROOT / "sim_data"
REAL_COMMANDS = PROJECT_ROOT / "config" / "real_commands.json"
KNOWLEDGE_DIR = Path(para.KNOWLEDGE)
DIGESTS_DIR = Path(para.DIGESTS)
OPS_LOGS = Path(para.OPS_LOGS)
DB_PATH = KNOWLEDGE_DIR / "knowledge.db"
SUMMARY_PATH = KNOWLEDGE_DIR / "knowledge_summary.json"
MAIN_PATH = KNOWLEDGE_DIR / "KNOWLEDGE.md"

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
    last_seen TEXT,
    generation TEXT,
    git_rev TEXT,
    note TEXT,
    fingerprint TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_facts_source_kind ON facts(source, kind);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);

-- 증분 커서: 어디까지 읽었는지. 이게 있어야 매번 모든 로그를 열지
-- 않는다 (실측 174 세션 113MB 를 사이클마다 전부 읽고 있었다).
CREATE TABLE IF NOT EXISTS cursors (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    offset INTEGER NOT NULL DEFAULT 0,
    scanned_at TEXT NOT NULL
);

-- 주기 요약(중간층) 색인: 원시 구간 하나를 접은 결과 한 줄.
CREATE TABLE IF NOT EXISTS digests (
    period TEXT PRIMARY KEY,          -- 'YYYY-MM-DD' (일 단위)
    source TEXT NOT NULL,             -- real | sim | ops
    path TEXT NOT NULL,               -- digests/<period>/<source>.json
    covered INTEGER NOT NULL,         -- 접어 넣은 원시 단위 수
    created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _provenance() -> dict:
    """이 동기화가 어떤 코드 세대에서 나왔는지 - 알고리즘이 계속 바뀌므로
    출처 없는 사실은 나중에 교훈을 오염시킨다 (사용자 원칙)."""
    import subprocess
    rev = ''
    try:
        rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=str(PROJECT_ROOT), capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    gen = 'unknown'
    try:
        src = (PROJECT_ROOT / 'sim' / 'sim_director.py').read_text(
            encoding='utf-8')
        m = re.search(r"RUN_GENERATION\s*=\s*'([^']+)'", src)
        if m:
            gen = m.group(1)
    except Exception:
        pass
    return {'git_rev': rev, 'generation': gen}


PROV = None                     # sync 당 한 번 계산


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # 구판 DB 마이그레이션 (이미 있으면 조용히 넘어간다)
    for col, typ in (('last_seen', 'TEXT'), ('generation', 'TEXT'),
                     ('git_rev', 'TEXT'), ('note', 'TEXT')):
        try:
            conn.execute('ALTER TABLE facts ADD COLUMN %s %s' % (col, typ))
        except sqlite3.OperationalError:
            pass
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
    global PROV
    if PROV is None:
        PROV = _provenance()
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    fingerprint = hashlib.sha256(
        f"{source}|{kind}|{fact_key}|{payload}|{evidence}".encode("utf-8")
    ).hexdigest()
    ts = now()
    conn.execute(
        """INSERT OR IGNORE INTO facts
        (source, kind, fact_key, value_json, confidence, status, evidence,
         observed_at, last_seen, generation, git_rev, fingerprint)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (source, kind, fact_key, payload, max(0.0, min(1.0, confidence)),
         status, evidence, ts, ts, PROV['generation'], PROV['git_rev'],
         fingerprint),
    )
    # 같은 사실을 다시 보면 '마지막 목격'만 갱신 - 살아있는 사실과
    # 사라진 사실을 가른다 (관리 체계의 은퇴 판단 근거).
    conn.execute("UPDATE facts SET last_seen=? WHERE fingerprint=?",
                 (ts, fingerprint))


def upsert_fact(conn: sqlite3.Connection, *, source: str, kind: str,
                fact_key: str, value: Any, confidence: float = 0.8,
                status: str = "observed", evidence: str = "") -> None:
    """롤업 결과를 넣는다 - 같은 사실이면 값을 갱신(누적값이 자란다).

    add_fact 는 '이 값을 이때 봤다'는 관측 기록(값이 다르면 새 행)이고,
    이쪽은 '현재 종합값'이다. 메인 지식에는 종합값만 둔다 - 안 그러면
    사이클마다 행이 쌓여 원장이 로그가 된다 (실측: policy_candidate
    899행).
    """
    global PROV
    if PROV is None:
        PROV = _provenance()
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    fingerprint = hashlib.sha256(
        f"rollup|{source}|{kind}|{fact_key}".encode("utf-8")).hexdigest()
    ts = now()
    cur = conn.execute(
        "UPDATE facts SET value_json=?, confidence=?, evidence=?, "
        "last_seen=?, generation=?, git_rev=? WHERE fingerprint=?",
        (payload, max(0.0, min(1.0, confidence)), evidence, ts,
         PROV['generation'], PROV['git_rev'], fingerprint))
    if cur.rowcount == 0:
        conn.execute(
            """INSERT INTO facts
            (source, kind, fact_key, value_json, confidence, status, evidence,
             observed_at, last_seen, generation, git_rev, fingerprint)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (source, kind, fact_key, payload, confidence, status, evidence,
             ts, ts, PROV['generation'], PROV['git_rev'], fingerprint))


def jsonl_files() -> Iterable[Path]:
    if not PI_LOGS.exists():
        return ()
    return PI_LOGS.rglob("events.jsonl")


# --------------------------------------------------------------- 증분 커서
def read_new_lines(conn: sqlite3.Connection, path: Path,
                   full: bool = False) -> list[str]:
    """이 파일에서 '아직 안 읽은' 줄만 돌려준다.

    로그는 append-only 라 크기만 커진다 - 지난번 오프셋부터 읽으면
    같은 줄을 다시 파싱하지 않는다. 파일이 줄었거나(truncate/교체)
    full=True 면 처음부터 다시 읽는다.
    """
    try:
        stat = path.stat()
    except OSError:
        return []
    row = conn.execute("SELECT size, offset FROM cursors WHERE path=?",
                       (str(path),)).fetchone()
    start = 0
    if row and not full and stat.st_size >= row["size"]:
        start = int(row["offset"])
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(start)
            lines = fh.read().splitlines()
            end = fh.tell()
    except OSError:
        return []
    conn.execute(
        "INSERT INTO cursors (path, size, mtime, offset, scanned_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
        "size=excluded.size, mtime=excluded.mtime, offset=excluded.offset, "
        "scanned_at=excluded.scanned_at",
        (str(path), stat.st_size, stat.st_mtime, end, now()))
    return lines


def pending_sources(conn: sqlite3.Connection) -> dict:
    """아직 안 읽은 원시가 얼마나 남았는지 (운영 가시성용)."""
    total = new = 0
    for path in list(jsonl_files()):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total += 1
        row = conn.execute("SELECT offset FROM cursors WHERE path=?",
                           (str(path),)).fetchone()
        if not row or row["offset"] < size:
            new += 1
    return {"files": total, "with_new_data": new}


def collect_real(conn: sqlite3.Connection, full: bool = False) -> dict[str, Any]:
    event_counts: collections.Counter[str] = collections.Counter()
    outcome_counts: collections.Counter[str] = collections.Counter()
    command_counts: collections.Counter[str] = collections.Counter()
    total = 0
    grasp_labels: collections.Counter[str] = collections.Counter()
    for path in jsonl_files():
        lines = read_new_lines(conn, path, full=full)
        for line in lines:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            total += 1
            # TurnLogger 는 'kind' 로 쓴다 - type/event 만 보던 초판은
            # 실전 이벤트 376건을 전부 unknown 으로 뭉갰다.
            kind = str(event.get("kind") or event.get("type")
                       or event.get("event") or "unknown")[:80]
            event_counts[kind] += 1
            outcome = event.get("outcome") or event.get("status")
            if outcome:
                outcome_counts[str(outcome)[:80]] += 1
            if kind == "grasp_label":
                grasp_labels[str(event.get("label") or "unknown")[:20]] += 1
            command = event.get("command") or event.get("text")
            # Keep only a digest.  Raw conversation and names stay in the existing archive.
            if isinstance(command, str) and command.strip():
                normalized = re.sub(r"\s+", " ", command).strip().lower()
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
                command_counts[digest] += 1

    # 여기서는 사실을 넣지 않는다 - 이번 구간의 '델타'만 만든다.
    # 델타는 digest(중간층)로 저장되고, 메인 지식은 rollup 이 만든다.
    persons = {}
    # 3층 재편으로 원시는 raw/ 밑이다 - para 가 단일 출처 (하드코딩 금지)
    db_path = Path(para.PERSONS) / "persons.db"
    if db_path.exists():
        try:
            db = sqlite3.connect(db_path)
            persons["people"] = db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            persons["boxes"] = db.execute("SELECT COUNT(*) FROM person_boxes").fetchone()[0]
            db.close()
        except sqlite3.Error:
            persons = {}
    return {"events": total, "event_kinds": dict(event_counts),
            "outcomes": dict(outcome_counts), "command_digests": len(command_counts),
            "grasp_labels": dict(grasp_labels), "person_dataset": persons}


def improve_files() -> list[Path]:
    files = set(SIM_DATA.glob("improve-*.json")) if SIM_DATA.exists() else set()
    if SIM_RUNS.exists():
        files.update(SIM_RUNS.rglob("improve-*.json"))
    return sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0)


def collect_sim(conn: sqlite3.Connection, full: bool = False) -> dict[str, Any]:
    """이번 구간의 시뮬 델타. 이미 접은 run 파일은 다시 열지 않는다."""
    runs = 0
    skipped = 0
    latest: dict[str, Any] | None = None
    causes: collections.Counter[str] = collections.Counter()
    best_cand: dict[str, Any] | None = None
    for path in improve_files():
        # 커서: 이미 읽은 run 은 건너뛴다 (899개를 매번 파싱하던 것).
        # full 재구축이어도 커서는 남긴다 - 다음 동기화가 증분이 되게.
        try:
            stat = path.stat()
        except OSError:
            continue
        if True:
            row = conn.execute("SELECT size FROM cursors WHERE path=?",
                               (str(path),)).fetchone()
            if row and int(row["size"]) == stat.st_size and not full:
                skipped += 1
                continue
            conn.execute(
                "INSERT INTO cursors (path, size, mtime, offset, scanned_at) "
                "VALUES (?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                "size=excluded.size, mtime=excluded.mtime, "
                "scanned_at=excluded.scanned_at",
                (str(path), stat.st_size, stat.st_mtime, stat.st_size, now()))
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
        # 후보는 전부 저장하지 않는다 - 이 구간의 '최고' 하나만 남긴다
        # (전부 넣던 초판이 원장을 899행짜리 로그로 만들었다).
        def _score(v):
            return (v.get("success") or 0, v.get("quality") or 0,
                    -(v.get("collision") or 0))
        if value.get("params") and (best_cand is None
                                    or _score(value) > _score(best_cand)):
            best_cand = dict(value, path=str(path))
        history = data.get("history") or []
        if isinstance(history, list):
            for row in history:
                if not isinstance(row, dict):
                    continue
                cz = row.get("causes")
                # causes 는 {원인: 횟수} dict 다 - list 로 검사하던 초판
                # 버그로 failure_cause 사실이 한 건도 안 쌓였다.
                if isinstance(cz, dict):
                    for cause, cnt in cz.items():
                        causes[str(cause)[:100]] += int(cnt or 0)
                elif isinstance(cz, list):
                    for cause in cz:
                        causes[str(cause)[:100]] += 1
        if path.exists() and (latest is None or path.stat().st_mtime > latest.get("mtime", 0)):
            latest = {"path": str(path), "mtime": path.stat().st_mtime, **value}
    if latest:
        latest = {k: v for k, v in latest.items() if k != "mtime"}
    return {"runs": runs, "skipped_runs": skipped, "latest": latest,
            "failure_causes": dict(causes), "best_candidate": best_cand}


# --------------------------------------- 1층 수집: 말·프롬프트·페르소나
# 계측만 지식이 아니다 (사용자). 로봇이 '무엇을 말하도록 되어 있는가'
# (페르소나·프롬프트)와 '대화에서 무엇을 배웠는가'(즉답 노트, STT 교정,
# 지적 원장, 방향 결정)도 같은 원장에 모은다. 프롬프트는 내용 전문이
# 아니라 지문(해시)·크기·규칙 수를 남긴다 - 원문은 git 과 config 에
# 있고, 원장은 '어느 판이 언제 쓰였나'를 잇는 역할이다.

PROMPT_FILES = ("persona.txt", "motion_prompt.txt", "vision_prompt.txt")
CODE_PROMPTS = (("sim/sim_director.py", "DIRECTOR_PROMPT"),
                ("sim/sim_critic.py", "CRITIQUE_PROMPT"),
                ("sim/sim_critic.py", "ORCHESTRA_PROMPT"),
                ("sim/sim_critic.py", "DUEL_PROMPT"),
                ("sim/sim_pilot.py", "PILOT_PROMPT"))


def _digest_text(text: str) -> dict:
    body = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rules = [ln for ln in body if ln.startswith(("-", "*", "•"))]
    return {"sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
            "chars": len(text), "lines": len(body), "rules": len(rules)}


def collect_assets(conn: sqlite3.Connection) -> dict[str, Any]:
    """프롬프트·페르소나 자산의 현재 판을 사실로. 바뀌면 이력이 남는다."""
    out: dict[str, Any] = {"prompts": {}, "persona_sections": 0}
    for name in PROMPT_FILES:
        path = PROJECT_ROOT / "config" / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        d = _digest_text(text)
        out["prompts"][name] = d
        # upsert = 현재 판, add_fact = '이 판을 이때 봤다'는 이력 한 줄
        upsert_fact(conn, source="real", kind="prompt_asset", fact_key=name,
                    value=d, confidence=0.95, evidence=str(path))
        add_fact(conn, source="real", kind="prompt_version", fact_key=name,
                 value=d, confidence=0.9, evidence=str(path))
        if name == "persona.txt":
            out["persona_sections"] = sum(
                1 for ln in text.splitlines() if ln.strip().endswith(":"))
    for rel, const in CODE_PROMPTS:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        m = re.search(const + r'\s*=\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')',
                      path.read_text(encoding="utf-8", errors="replace"), re.S)
        if not m:
            continue
        d = _digest_text(m.group(1))
        out["prompts"][const] = d
        upsert_fact(conn, source="sim", kind="prompt_asset", fact_key=const,
                    value=d, confidence=0.95, evidence=rel)
        add_fact(conn, source="sim", kind="prompt_version", fact_key=const,
                 value=d, confidence=0.9, evidence=rel)
    return out


def collect_conversation(conn: sqlite3.Connection) -> dict[str, Any]:
    """대화에서 얻은 지식: 즉답 노트, STT 교정, 지적 원장, 방향 결정.

    원문 대화는 넣지 않는다 - 여기 담기는 것은 이미 '규칙으로 승격된'
    것들(패턴->답변, 오인식->교정어)과 집계뿐이다.
    """
    out: dict[str, Any] = {}
    cfg = PROJECT_ROOT / "config"

    notes = _load_json(cfg / "quick_notes.json") or {}
    rows = notes.get("notes") if isinstance(notes, dict) else notes
    if isinstance(rows, list):
        out["quick_notes"] = len(rows)
        by_kind: collections.Counter = collections.Counter()
        for n in rows:
            if not isinstance(n, dict):
                continue
            kind = str(n.get("kind") or "misc")[:40]
            by_kind[kind] += 1
            upsert_fact(conn, source="real", kind="dialog_rule",
                        fact_key="%s/%s" % (kind, (n.get("patterns") or [""])[0][:30]),
                        value={"kind": kind,
                               "patterns": (n.get("patterns") or [])[:6],
                               "reply": (n.get("reply") or "")[:200]},
                        confidence=0.85, status="promoted",
                        evidence="config/quick_notes.json")
        out["quick_notes_by_kind"] = dict(by_kind)

    prop = _load_json(cfg / "quick_notes.proposed.json")
    cand = ((prop.get("proposals") or prop.get("notes"))
            if isinstance(prop, dict) else prop)
    if isinstance(cand, list):
        out["dialog_candidates"] = len(cand)
        for n in cand[:15]:
            if not isinstance(n, dict):
                continue
            pat = (n.get("patterns") or [n.get("example")
                                         or n.get("norm")
                                         or n.get("text") or ""])[0]
            upsert_fact(conn, source="real", kind="dialog_candidate",
                        fact_key=str(pat)[:60],
                        value={"patterns": (n.get("patterns")
                                            or [n.get("example")])[:4],
                               "reply": (n.get("reply") or "")[:200],
                               "count": n.get("count")},
                        confidence=0.5, status="proposed",
                        evidence="config/quick_notes.proposed.json")

    stt = _load_json(cfg / "stt_corrections.json") or {}
    fixes = {k: v for k, v in stt.items() if k != "_comment"}
    counts = {k: len(v) for k, v in fixes.items() if isinstance(v, (dict, list))}
    if counts:
        out["stt_corrections"] = counts
        upsert_fact(conn, source="real", kind="stt_correction",
                    fact_key="aggregate", value=counts, confidence=0.9,
                    status="promoted", evidence="config/stt_corrections.json")

    fb = _load_json(cfg / "sim_feedback.json")
    if isinstance(fb, list):
        open_notes = [f for f in fb if isinstance(f, dict) and not f.get("done")]
        out["feedback"] = {"total": len(fb), "open": len(open_notes)}
        for f in open_notes[:20]:
            upsert_fact(conn, source="sim", kind="feedback_note",
                        fact_key=str(f.get("note"))[:80],
                        value={"note": f.get("note")}, confidence=0.6,
                        status="proposed", evidence="config/sim_feedback.json")

    log_md = PROJECT_ROOT / "docs" / "eval" / "direction-log.md"
    if log_md.exists():
        text = log_md.read_text(encoding="utf-8", errors="replace")
        decisions = re.findall(r"^- \*\*방향\*\*: (.+)$", text, re.M)
        out["directions"] = len(decisions)
        for d in decisions[-5:]:
            upsert_fact(conn, source="sim", kind="direction_decision",
                        fact_key=hashlib.sha256(d.encode()).hexdigest()[:12],
                        value={"why": d[:300]}, confidence=0.7,
                        evidence="docs/eval/direction-log.md")
    return out


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ------------------------------------------------- 2층: 주기 요약(지식화)
def write_digest(conn: sqlite3.Connection, source: str, delta: dict,
                 period: str | None = None, replace: bool = False) -> Path | None:
    """이번 구간의 델타를 하루치 요약 파일로 접는다.

    같은 날 여러 번 돌면 그날 요약에 합산된다 - 하루가 한 장이 되고,
    메인 지식은 이 장들만 본다 (원시 로그를 다시 열지 않는 이유).
    """
    period = period or time.strftime("%Y-%m-%d")
    out_dir = DIGESTS_DIR / period
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source}.json"
    base: dict = {}
    if path.exists() and not replace:
        try:
            base = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            base = {}
    merged = _merge_delta(base, delta)
    merged["period"] = period
    merged["source"] = source
    merged["updated_at"] = now()
    merged.setdefault("generation", (PROV or _provenance())["generation"])
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    covered = int(merged.get("events") or merged.get("runs") or 0)
    conn.execute(
        "INSERT INTO digests (period, source, path, covered, created_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(period) DO UPDATE SET "
        "path=excluded.path, covered=excluded.covered",
        (f"{period}|{source}", source, str(path), covered, now()))
    return path


def _merge_delta(base: dict, delta: dict) -> dict:
    """숫자는 더하고, 카운터 dict 는 키별로 더하고, 나머지는 최신으로."""
    out = dict(base)
    for k, v in delta.items():
        if isinstance(v, (int, float)) and isinstance(out.get(k), (int, float)):
            out[k] = out[k] + v
        elif k in ("person_dataset", "assets", "conversation"):
            out[k] = v or out.get(k)      # 현재 상태 - 합산하지 않는다
        elif isinstance(v, dict) and all(
                isinstance(x, (int, float)) for x in v.values()):
            merged = dict(out.get(k) or {})
            for kk, vv in v.items():
                merged[kk] = merged.get(kk, 0) + vv
            out[k] = merged
        elif v not in (None, {}, [], 0):
            out[k] = v
        else:
            out.setdefault(k, v)
    return out


# ------------------------------------------------------- 3층: 메인 지식
def rollup(conn: sqlite3.Connection) -> dict:
    """모든 일별 요약을 합쳐 메인 지식을 만든다.

    여기서만 facts 에 쓴다 (upsert - 종합값 갱신). 원시 로그는 건드리지
    않는다 - 요약 파일 수십 장만 읽으면 된다.
    """
    totals: dict[str, dict] = {"real": {}, "sim": {}}
    periods: list[str] = []
    if DIGESTS_DIR.exists():
        for day_dir in sorted(DIGESTS_DIR.iterdir()):
            if not day_dir.is_dir():
                continue
            periods.append(day_dir.name)
            for src in ("real", "sim"):
                f = day_dir / f"{src}.json"
                if not f.exists():
                    continue
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                d.pop("period", None); d.pop("source", None)
                d.pop("updated_at", None); d.pop("generation", None)
                totals[src] = _merge_delta(totals[src], d)

    for kind, count in (totals["real"].get("event_kinds") or {}).items():
        upsert_fact(conn, source="real", kind="event_count", fact_key=kind,
                    value={"count": count}, confidence=min(0.99, 0.5 + count / 100.0),
                    evidence="digests")
    for outcome, count in (totals["real"].get("outcomes") or {}).items():
        upsert_fact(conn, source="real", kind="outcome_count", fact_key=outcome,
                    value={"count": count}, confidence=min(0.99, 0.5 + count / 100.0),
                    evidence="digests")
    for label, count in (totals["real"].get("grasp_labels") or {}).items():
        upsert_fact(conn, source="real", kind="grasp_label", fact_key=label,
                    value={"count": count}, confidence=0.7, evidence="digests")
    if totals["real"].get("person_dataset"):
        upsert_fact(conn, source="real", kind="person_dataset",
                    fact_key="aggregate", value=totals["real"]["person_dataset"],
                    confidence=0.8, evidence="digests")
    for cause, count in (totals["sim"].get("failure_causes") or {}).items():
        upsert_fact(conn, source="sim", kind="failure_cause", fact_key=cause,
                    value={"count": count},
                    confidence=min(0.95, 0.5 + count / 20.0), evidence="digests")
    if totals["sim"].get("latest"):
        upsert_fact(conn, source="sim", kind="latest_metrics", fact_key="latest",
                    value=totals["sim"]["latest"], confidence=0.85,
                    evidence="digests")
    if totals["sim"].get("best_candidate"):
        upsert_fact(conn, source="sim", kind="policy_candidate",
                    fact_key="best", value=totals["sim"]["best_candidate"],
                    confidence=0.9, status="proposed", evidence="digests")
    conn.commit()
    return {"periods": len(periods), "range": [periods[0], periods[-1]]
            if periods else [], "real": totals["real"], "sim": totals["sim"]}


def _brief(latest: dict | None) -> str:
    """메인 한 장에는 지표만 - 파라미터 전문은 knowledge.db 에 있다."""
    if not isinstance(latest, dict):
        return ""
    keep = {k: latest.get(k) for k in ("quality", "success", "collision")
            if latest.get(k) is not None}
    src = Path(latest.get("path", "")).name
    return "%s (%s)" % (keep, src) if src else str(keep)


def write_main(rolled: dict, conn: sqlite3.Connection) -> Path:
    """사람과 모델이 '이것만 읽으면 되는' 한 장 (04-Archives/knowledge)."""
    real, sim = rolled.get("real", {}), rolled.get("sim", {})
    prompts = (real.get("assets") or {}).get("prompts") or {}
    talk = real.get("conversation") or {}
    rows = conn.execute(
        "SELECT source, kind, fact_key, value_json FROM facts "
        "WHERE status='proposed' ORDER BY id DESC LIMIT 8").fetchall()
    causes = sorted((sim.get("failure_causes") or {}).items(),
                    key=lambda kv: -kv[1])[:5]
    lines = [
        "# 메인 지식 (Reachy)", "",
        "실전·시뮬 원시 데이터를 일별 요약으로 접고, 그 요약들을 합친 종합본.",
        "**원시 로그를 열 필요 없이 이 장과 knowledge.db 만 보면 된다.**", "",
        "- 갱신: %s" % now(),
        "- 요약 구간: %s (%d일치)" % (" ~ ".join(rolled.get("range") or ["-"]),
                                     rolled.get("periods", 0)),
        "", "## 실전 (로봇)", "",
        "- 이벤트 %s건, 종류: %s" % (
            real.get("events", 0),
            ", ".join("%s %s" % kv for kv in sorted(
                (real.get("event_kinds") or {}).items(),
                key=lambda kv: -kv[1])[:6]) or "-"),
        "- 동작 결과: %s" % (real.get("outcomes") or "-"),
        "- 파지 라벨(사람이 말해 준 성패): %s" % (real.get("grasp_labels") or "아직 없음"),
        "- 사람 데이터셋: %s" % (real.get("person_dataset") or "-"),
        "", "## 시뮬", "",
        "- 학습 run %s건 접힘" % sim.get("runs", 0),
        "- 최신 지표: %s" % (_brief(sim.get("latest")) or "-"),
        "- 실패 원인 상위: %s" % (", ".join("%s %d" % c for c in causes) or "-"),
        "", "## 말과 규칙 (로봇이 무엇을 말하도록 되어 있나)", "",
        "- 페르소나: %s (%s자, 규칙 %s줄)" % (
            (prompts.get("persona.txt") or {}).get("sha", "-"),
            (prompts.get("persona.txt") or {}).get("chars", 0),
            (prompts.get("persona.txt") or {}).get("rules", 0)),
        "- 프롬프트 판본 %d종: %s" % (
            len(prompts), ", ".join("%s=%s" % (k, v.get("sha"))
                                    for k, v in sorted(prompts.items()))),
        "- 즉답 규칙(대화에서 승격): %s건, 검토 대기 후보 %s건" % (
            talk.get("quick_notes", 0), talk.get("dialog_candidates", 0)),
        "- STT 오인식 교정: %s" % (talk.get("stt_corrections") or "-"),
        "- 동작 지적 원장: %s" % (talk.get("feedback") or "-"),
        "- 기록된 방향 결정: %s건" % talk.get("directions", 0),
        "", "## 검토 대기 (proposed)", ""]
    for r in rows:
        lines.append("- [%s/%s] %s" % (r["source"], r["kind"],
                                       r["fact_key"][:70]))
    lines += ["", "---", "",
              "구조·운영법: `01-Projects/reachy-interactive/docs/architecture/"
              "knowledge.md`"]
    MAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAIN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MAIN_PATH


def sync(full: bool = False) -> dict[str, Any]:
    """1층(원시) -> 2층(일별 요약) -> 3층(메인 지식) 한 바퀴."""
    conn = connect()
    real = collect_real(conn, full=full)      # 델타
    sim = collect_sim(conn, full=full)        # 델타
    assets = collect_assets(conn)             # 프롬프트·페르소나 현재 판
    talk = collect_conversation(conn)         # 대화에서 얻은 규칙·지적
    real["assets"] = assets
    real["conversation"] = talk
    # full 재구축이면 그날 요약을 '덮어쓴다' - 합산하면 같은 원시를 두 번
    # 센다 (실측: 재구축 두 번에 runs 899 -> 1798).
    write_digest(conn, "real", real, replace=full)     # 2층에 접기
    write_digest(conn, "sim", sim, replace=full)
    conn.commit()
    rolled = rollup(conn)                     # 3층 종합
    write_main(rolled, conn)
    row = conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()
    summary = {
        "schema_version": 2,
        "generated_at": now(),
        "facts": row["n"],
        "delta": {"real": real, "sim": sim},
        "rolled": rolled,
        "pending_raw": pending_sources(conn),
        "policy": {
            "automatic_promotion": False,
            "note": "sim 후보는 검토 후에만 승격; 원문 음성·얼굴은 원장에 복사하지 않음",
        },
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    conn.commit()
    conn.close()
    return summary


# ---------------------------------------------------------------- 소비자들
# 축적만 하고 흐르는 곳이 없으면 원장은 장식이다. 세 소비자가 있다:
#   1) 감독 증거 요약 (digest_for_director) - 매 사이클 방향 결정에 주입
#   2) 게이트 단어 제안 (propose_gate_words) - 실전 null_moves 에서 어휘 발굴
#   3) 주간 다이제스트 (weekly_digest) - 사람 검토용 마크다운

def digest_for_director(top: int = 5) -> dict:
    """감독의 방향 결정에 넣을 압축 요약.

    메인 지식(facts)과 실수요 파일만 읽는다 - 원시 로그·run 을 열지
    않으므로 사이클마다 불러도 싸다. LLM 호출도 없어 offline 안전.
    """
    out: dict = {}
    try:
        rc = json.loads(REAL_COMMANDS.read_text(encoding="utf-8"))
        out["실전_수요_상위"] = [
            {"명령": c.get("text"), "횟수": c.get("count")}
            for c in (rc.get("commands") or [])[:top]]
        out["실전_실패분포"] = rc.get("failures")
    except Exception:
        pass
    try:
        conn = connect()
        rows = conn.execute(
            "SELECT fact_key, value_json FROM facts "
            "WHERE source='sim' AND kind='failure_cause' "
            "ORDER BY json_extract(value_json,'$.count') DESC LIMIT ?",
            (top,)).fetchall()
        out["시뮬_실패원인_누적"] = {
            r["fact_key"]: json.loads(r["value_json"]).get("count")
            for r in rows}
        n_prop = conn.execute(
            "SELECT COUNT(*) AS n FROM facts WHERE status='proposed'"
        ).fetchone()["n"]
        out["승격_대기_사실"] = n_prop
        lab = conn.execute(
            "SELECT fact_key, value_json FROM facts "
            "WHERE source='real' AND kind='grasp_label'").fetchall()
        if lab:
            out["실전_파지_라벨"] = {
                r["fact_key"]: json.loads(r["value_json"]).get("count")
                for r in lab}
        row = conn.execute(
            "SELECT COUNT(*) AS n, MIN(period) AS a, MAX(period) AS b "
            "FROM digests").fetchone()
        out["요약_구간"] = {"장수": row["n"], "처음": row["a"], "끝": row["b"]}
        # 지금 어떤 프롬프트 판으로 돌고 있는지 - 성과를 프롬프트 판에
        # 귀속시키려면 감독도 이걸 알아야 한다
        pr = conn.execute(
            "SELECT fact_key, value_json FROM facts WHERE kind='prompt_asset'"
        ).fetchall()
        if pr:
            out["프롬프트_판본"] = {
                r["fact_key"]: json.loads(r["value_json"]).get("sha")
                for r in pr}
        fbrow = conn.execute(
            "SELECT fact_key FROM facts WHERE kind='feedback_note' "
            "AND status='proposed' LIMIT 5").fetchall()
        if fbrow:
            out["미결_동작지적"] = [r["fact_key"] for r in fbrow]
        conn.close()
    except Exception:
        pass
    return out


_KO_TOKEN = re.compile(r"[\uac00-\ud7a3]{2,}")
_GATE_STOP = {"해줘", "해봐", "한번", "그리고", "이거", "저거", "그거",
              "있는", "으로", "좀", "말이야", "리치"}


def _voice_vocab() -> set:
    """voice_chat.py 의 게이트 어휘를 AST 로 읽는다 (임포트 없이).

    정규식은 튜플 안 주석의 닫는 괄호에서 끊겨 '물건' 같은 기존 어휘를
    놓쳤다 (실측) - 구문 트리는 주석과 무관하다.
    """
    import ast
    names = {"BODY_WORDS", "GESTURE_WORDS", "GESTURE_WEAK",
             "ACTION_STEMS", "NOT_MOTION", "GESTURE_STOPWORDS",
             "COMMAND_TAILS", "HEAD_WORDS"}
    vocab: set = set()
    try:
        tree = ast.parse((PROJECT_ROOT / "robot" / "voice_chat.py")
                         .read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = {t.id for t in node.targets
                       if isinstance(t, ast.Name)}
            if not (targets & names):
                continue
            for c in ast.walk(node.value):
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    vocab.add(c.value.replace(" ", ""))
    except Exception:
        pass
    return vocab


def propose_gate_words(min_count: int = 2) -> dict:
    """실전에서 동작으로 라우팅되지 못한 명령(null_moves 등)의 어휘를
    발굴해 게이트 후보로 제안한다 - '어깨' 누락 사고(2026-09-08)의
    자동화판. 후보는 단어만 저장한다 (원문 문장은 원장에 안 들어간다)."""
    vocab = _voice_vocab()
    counts: collections.Counter = collections.Counter()
    for path in jsonl_files():
        try:
            for line in path.read_text(encoding="utf-8",
                                       errors="replace").splitlines():
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if not isinstance(e, dict):
                    continue
                if (e.get("outcome") or e.get("status")) not in (
                        "null_moves", "rejected"):
                    continue
                for tok in _KO_TOKEN.findall(str(e.get("text") or "")):
                    if tok in _GATE_STOP:
                        continue
                    if any(v in tok or tok in v for v in vocab):
                        continue
                    counts[tok] += 1
        except OSError:
            continue
    cand = [{"word": w, "count": n} for w, n in counts.most_common(12)
            if n >= min_count]
    out_path = PROJECT_ROOT / "config" / "gate_word_candidates.json"
    out_path.write_text(json.dumps(
        {"generated_at": now(), "note":
         "동작 게이트에 없어 잡담으로 샌 명령들의 어휘 후보 - 사람 검토 후 "
         "voice_chat.py BODY_WORDS/GESTURE 에 반영",
         "candidates": cand}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    conn = connect()
    for c in cand:
        add_fact(conn, source="real", kind="gate_word_candidate",
                 fact_key=c["word"], value=c, confidence=0.6,
                 status="proposed", evidence=str(out_path))
    conn.commit(); conn.close()
    return {"candidates": cand, "path": str(out_path)}


def weekly_digest() -> str:
    """사람 검토용 주간 다이제스트 마크다운."""
    summary = {}
    try:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    conn = connect()
    proposed = conn.execute(
        "SELECT id, source, kind, fact_key, status, generation FROM facts "
        "WHERE status='proposed' ORDER BY id DESC LIMIT 15").fetchall()
    conn.close()
    lines = ["# 지식 다이제스트 (%s)" % now()[:10], "",
             "지식 원장(04-Archives/knowledge)의 사람 검토용 요약. "
             "승격/기각은 `knowledge_ledger.py --promote/--reject <id>`.", "",
             "## 실전", "```json",
             json.dumps(summary.get("real", {}), ensure_ascii=False, indent=1),
             "```", "## 시뮬", "```json",
             json.dumps(summary.get("sim", {}), ensure_ascii=False, indent=1),
             "```", "", "## 승격 대기 (proposed)"]
    for r in proposed:
        lines.append("- [%d] %s/%s %s (%s)" % (
            r["id"], r["source"], r["kind"], r["fact_key"][:60],
            r["generation"] or "?"))
    md = "\n".join(lines) + "\n"
    out = PROJECT_ROOT / "docs" / "eval" / "knowledge-weekly.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return str(out)


def set_status(fact_id: int, status: str, note: str = "") -> None:
    conn = connect()
    conn.execute("UPDATE facts SET status=?, note=? WHERE id=?",
                 (status, note or None, fact_id))
    conn.commit(); conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true", help="collect current real/sim evidence")
    parser.add_argument("--full", action="store_true",
                        help="커서를 무시하고 원시 전체 재구축 (그날 요약 덮어씀)")
    parser.add_argument("--stats", action="store_true", help="print the generated summary")
    parser.add_argument("--digest-director", action="store_true",
                        help="감독 증거용 압축 요약 JSON 출력")
    parser.add_argument("--propose-gates", action="store_true",
                        help="실전 미라우팅 명령에서 게이트 단어 후보 발굴")
    parser.add_argument("--weekly", action="store_true",
                        help="docs/eval/knowledge-weekly.md 생성")
    parser.add_argument("--promote", type=int, metavar="ID",
                        help="사실을 promoted 로 (정책 반영됨 표시)")
    parser.add_argument("--reject", type=int, metavar="ID")
    parser.add_argument("--note", default="", help="promote/reject 사유")
    args = parser.parse_args()
    if args.digest_director:
        print(json.dumps(digest_for_director(), ensure_ascii=False, indent=1))
        return 0
    if args.propose_gates:
        print(json.dumps(propose_gate_words(), ensure_ascii=False, indent=1))
        return 0
    if args.weekly:
        print(weekly_digest())
        return 0
    if args.promote is not None:
        set_status(args.promote, "promoted", args.note); return 0
    if args.reject is not None:
        set_status(args.reject, "rejected", args.note); return 0
    summary = (sync(full=args.full) if args.sync or args.full or not args.stats
               else json.loads(SUMMARY_PATH.read_text(encoding="utf-8")))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
