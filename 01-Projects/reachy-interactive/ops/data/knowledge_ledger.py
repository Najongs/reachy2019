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
    last_seen TEXT,
    generation TEXT,
    git_rev TEXT,
    note TEXT,
    fingerprint TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_facts_source_kind ON facts(source, kind);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
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
            # TurnLogger 는 'kind' 로 쓴다 - type/event 만 보던 초판은
            # 실전 이벤트 376건을 전부 unknown 으로 뭉갰다.
            kind = str(event.get("kind") or event.get("type")
                       or event.get("event") or "unknown")[:80]
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


# ---------------------------------------------------------------- 소비자들
# 축적만 하고 흐르는 곳이 없으면 원장은 장식이다. 세 소비자가 있다:
#   1) 감독 증거 요약 (digest_for_director) - 매 사이클 방향 결정에 주입
#   2) 게이트 단어 제안 (propose_gate_words) - 실전 null_moves 에서 어휘 발굴
#   3) 주간 다이제스트 (weekly_digest) - 사람 검토용 마크다운

def digest_for_director(top: int = 5) -> dict:
    """감독의 방향 결정에 넣을 압축 요약. LLM 호출 없음 - offline 안전."""
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
    summary = sync() if args.sync or not args.stats else json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
