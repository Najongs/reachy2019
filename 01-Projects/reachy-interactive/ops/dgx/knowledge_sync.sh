#!/bin/bash
# 지식 싱크 (4시간마다) - Pi 와 DGX 의 지식을 한 바퀴 돌린다.
#
#   1) 당겨오기   Pi 의 대화 로그·사람 사진  ->  DGX 원시 보관고(raw/)
#   2) 지식화     원시 증분 -> 일별 요약 -> 메인 지식 (knowledge_ledger --sync)
#   3) 되돌려주기 갱신된 지식 자산 -> Pi (즉답 노트·STT 교정·미리합성 목록)
#   4) 반영       Pi 와 브로커가 '재시작 없이' 새 판을 집어 든다 (핫 리로드)
#
# 왜 4시간인가: 하루 1회(daily_update)는 데이터가 하루 묵고, 매 분은 Pi SD
# 와 네트워크를 헛되이 긁는다. 근무시간(9~18) 안에 두어 번 돌면 그날 배운
# 것이 그날 반영된다.
#
# 재시작을 안 하는 이유: 로봇 서비스 재시작은 대화를 끊고, 브로커 재시작은
# 모델 재적재(~40초)를 부른다. 그래서 파일만 바꾸고, 양쪽의 감시자가
# mtime 을 보고 조용히 갈아 끼운다 (voice_chat.reload_knowledge,
# llm_broker.PromptFile).
#
#   bash ops/dgx/knowledge_sync.sh          # 한 바퀴
#   bash ops/dgx/knowledge_sync.sh --pull-only   # 당겨오기+지식화만
#   bash ops/dgx/knowledge_sync.sh --quiet       # cron 용 (요약만 출력)
set -u
cd "$(dirname "$0")/../.." || exit 1                 # reachy-interactive/

SSH="ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=10"
SCP="scp -P 2222 -o BatchMode=yes -o ConnectTimeout=10"
PI=pi@localhost
PULL_ONLY=0; QUIET=0
for a in "$@"; do
  case "$a" in
    --pull-only) PULL_ONLY=1 ;;
    --quiet) QUIET=1 ;;
    *) echo "unknown arg: $a"; exit 1 ;;
  esac
done

PI_LOGS=$(python3 -c 'import sys; sys.path.insert(0, "."); import para; print(para.PI_LOGS)')
KNOW=$(python3 -c 'import sys; sys.path.insert(0, "."); import para; print(para.KNOWLEDGE)')
LOG="$(dirname "$PI_LOGS")/../ops-logs/knowledge_sync.log"
mkdir -p "$PI_LOGS" "$(dirname "$LOG")"

say() { [ "$QUIET" = 1 ] || echo "$@"; }
stamp() { echo "$(TZ=Asia/Seoul date '+%F %T KST') $*" >> "$LOG"; }

stamp "싱크 시작"

# ── 1) Pi -> DGX: 원시 당겨오기 ───────────────────────────────────────────
PULLED=0
if $SSH $PI true 2>/dev/null; then
  if rsync -az --partial --exclude='persons/' \
       -e "$SSH" $PI:reachy_logs/ "$PI_LOGS/" 2>/dev/null; then
    PULLED=1
    say "1) 로그 당겨옴"
  else
    say "1) ! 로그 일부 실패 - 보관본으로 계속"
  fi
  # 사람 사진은 전용 동기화가 중복·유일키를 관리한다 (여기서 rsync 하면 두 벌)
  python3 ops/data/sync_persons.py >/dev/null 2>&1 \
    && say "   사람 사진 동기화 완료" \
    || say "   ! 사람 사진 동기화 실패 (계속)"
else
  say "1) ✗ Pi 연결 불가 (터널 2222) - 지식화만 진행"
  stamp "Pi 연결 불가"
fi

# ── 2) 지식화: 원시 증분 -> 일별 요약 -> 메인 지식 ────────────────────────
SYNC_OUT=$(python3 ops/data/knowledge_ledger.py --sync 2>&1)
if [ $? -eq 0 ]; then
  FACTS=$(printf '%s' "$SYNC_OUT" | python3 -c \
    'import json,sys; print(json.load(sys.stdin).get("facts","?"))' 2>/dev/null)
  NEW=$(printf '%s' "$SYNC_OUT" | python3 -c \
    'import json,sys; d=json.load(sys.stdin)["delta"]; print("%s이벤트/%s런" % (d["real"]["events"], d["sim"]["runs"]))' 2>/dev/null)
  say "2) 지식화: 새 ${NEW:-?} · 사실 ${FACTS:-?}건"
  stamp "지식화 새 ${NEW:-?} 사실 ${FACTS:-?}"
else
  say "2) ! 지식화 실패"
  stamp "지식화 실패"
fi
# 대화에서 자란 것들 (즉답 후보·실수요·미리합성 목록)
python3 broker/build_notes.py --logs "$PI_LOGS" >/dev/null 2>&1
python3 ops/dgx/build_tts_cache_list.py --write >/dev/null 2>&1
python3 ops/data/knowledge_ledger.py --propose-gates >/dev/null 2>&1

[ "$PULL_ONLY" = 1 ] && { say "(--pull-only) 끝"; stamp "pull-only 종료"; exit 0; }

# ── 3) DGX -> Pi: 갱신된 지식 자산 돌려주기 ──────────────────────────────
# 로봇이 실제로 읽는 것만 보낸다. 페르소나·동작 프롬프트는 브로커(DGX)가
# 쓰므로 보내지 않는다 - 그쪽은 PromptFile 이 파일에서 바로 다시 읽는다.
SENT=0
if [ "$PULLED" = 1 ] || $SSH $PI true 2>/dev/null; then
  for f in quick_notes.json stt_corrections.json cached_lines.json; do
    [ -f "config/$f" ] || continue
    # 내용이 같으면 보내지 않는다 - mtime 만 바뀌어도 로봇이 헛되이 재적재한다
    LOCAL=$(md5sum "config/$f" | cut -d' ' -f1)
    REMOTE=$($SSH $PI "md5sum Documents/$f 2>/dev/null | cut -d' ' -f1" 2>/dev/null)
    if [ "$LOCAL" != "$REMOTE" ]; then
      if $SCP "config/$f" $PI:~/Documents/ >/dev/null 2>&1; then
        SENT=$((SENT + 1))
        say "3) 보냄: $f"
        stamp "Pi 로 $f 전송"
      fi
    fi
  done
  [ "$SENT" = 0 ] && say "3) 보낼 변경 없음"
else
  say "3) ✗ Pi 연결 불가 - 다음 싱크에 재시도"
fi

# ── 4) 반영 확인 (재시작 없이) ───────────────────────────────────────────
# 로봇: voice_chat 이 60초마다 mtime 을 본다. 브로커: PromptFile 이 5초마다.
if [ "$SENT" -gt 0 ]; then
  say "4) 로봇이 60초 안에 스스로 집어 듭니다 (재시작 없음)"
fi
curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null 2>&1 \
  && say "   브로커 정상 (페르소나 변경은 5초 내 자동 반영)" \
  || { say "   ! 브로커 응답 없음 - broker_watchdog 이 처리"; stamp "브로커 무응답"; }

stamp "싱크 종료 (보낸 자산 $SENT)"
say "완료. 메인 지식: $KNOW/KNOWLEDGE.md"
