#!/bin/bash
# 하루치 로그 당겨오기 + 활동 다이제스트 + 노트 제안 (하루 1회, 수동 실행).
#
#   bash ops/daily_update.sh              # 당겨오기 + 요약 + 노트 제안(quick_notes.proposed.json)
#   bash ops/daily_update.sh --since 2026-09-03   # 그 날짜 이후만 요약
#   bash ops/daily_update.sh --merge      # 제안 중 안전·일관된 것 자동 승격까지
#   bash ops/daily_update.sh --no-clean   # Pi 로그 정리(voice_chat.log 비우기) 생략
#   bash ops/daily_update.sh --no-persons # 사람 사진 가져오기 생략
#   bash ops/daily_update.sh --purge-persons  # 사진을 옮긴 뒤 Pi 쪽 원본 삭제(SD 확보)
#
# 사람 데이터가 쌓이는 곳 (DGX):
#   04-Archives/person-dataset/persons/   원본 + persons.db (모으는 곳)
#   04-Archives/person-dataset/dataset/   걸러서 뽑아 낸 것 (images/ faces/ persons/)
#
# 로봇은 계속 켜져 있으니 logrotate 대신, 여기서 로그를 안전히 당겨온 뒤(=DGX 보관본
# 갱신) Pi 의 커지는 voice_chat.log 만 비운다. 세션 events.jsonl 은 건드리지 않는다.
# 노트는 기본 '제안'만 만든다 (config/quick_notes.proposed.json). 검토 후 --merge 로 반영.
set -e
cd "$(dirname "$0")/.."                                  # reachy-interactive/
PI_LOGS=../../04-Archives/conversation-logs/pi           # 저장소 로그 보관 위치
SINCE=""; MERGE=""; CLEAN=1; PERSONS=1; PURGE=
while [ $# -gt 0 ]; do
  case "$1" in
    --since) SINCE="--since $2"; shift 2 ;;
    --merge) MERGE="--merge"; shift ;;
    --no-clean) CLEAN=0; shift ;;
    --no-persons) PERSONS=0; shift ;;
    --purge-persons) PURGE="--purge-remote"; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$PI_LOGS"

echo "── [0/4] 수집이 살아 있나"
python3 ops/dgx/collect_health.py || true      # 문제가 있어도 나머지는 계속 돈다

echo
echo "── [1/4] Pi 로그 당겨오기 (역터널 2222)"
PULLED=0
if ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=10 pi@localhost true 2>/dev/null; then
  # 작은 로그(<1MB) — 매번 전체를 가져와 로컬 보관본 갱신.
  # persons/ 는 제외한다: 사람 사진은 [2/4] 가 중앙 데이터셋(~/reachy-data)으로
  # 따로 옮기므로, 여기서까지 받으면 같은 사진을 두 벌 쌓게 된다.
  if rsync -az --partial --exclude='persons/' \
       -e 'ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=10' \
       pi@localhost:reachy_logs/ "$PI_LOGS/" 2>/dev/null; then
    echo "   가져오기 완료"; PULLED=1
  else
    echo "   ! 일부 파일 복사 실패 - 로컬 보관본 사용"
  fi
  # 안전히 보관됐으면(PULLED) 커지는 원시 로그만 비운다(제자리 truncate — 서비스 계속 씀).
  # systemd 가 root 로 만든 로그라 sudo 필요(Pi 는 passwordless sudo).
  if [ "$PULLED" = 1 ] && [ "$CLEAN" = 1 ]; then
    ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=10 pi@localhost \
      'sudo truncate -s 0 /home/pi/reachy_logs/voice_chat.log' 2>/dev/null \
      && echo "   Pi voice_chat.log 비움(보관본에 이미 저장됨)"
  fi
else
  echo "   ✗ Pi 연결 불가 (터널 2222 확인: systemctl status pi_tunnel). 로컬 보관본으로 계속."
fi
turns=$(cat "$PI_LOGS"/*/events.jsonl 2>/dev/null | wc -l)
echo "   현재 총 턴 수: $turns"

echo
echo "── [2/4] 사람 사진 → 04-Archives/person-dataset/persons"
if [ "$PERSONS" = 1 ]; then
  python3 ops/data/sync_persons.py $PURGE || echo "   ! 사진 동기화 실패 (계속 진행)"
else
  echo "   건너뜀 (--no-persons)"
fi

echo
echo "── [3/4] 활동 다이제스트"
python3 ops/dgx/log_digest.py "$PI_LOGS" $SINCE

echo
echo "── [4/4] 노트 제안 (실제 로그 답변 재사용, 실행약속/동작류 제외)"
python3 broker/build_notes.py --logs "$PI_LOGS" $MERGE

echo
echo "── 사람 인식 DB 생성 (여기서 제대로 된 모델로)"
# 로봇은 '사람이 있다'만 판단해 사진을 모으고, 사람 위치는 여기서 다시 잡는다.
# torch 는 학습용 venv 에 있다. GPU 는 쓰지 않는다 - 다른 학습이 8장을 100% 로
# 쓰고 있어 끼어들면 그쪽이 느려진다.
VENV_PY=/home/kiro-ai/NAJY/trossen-ai-simulation/.venv/bin/python3
if [ "$PERSONS" = 1 ] && [ -x "$VENV_PY" ]; then
  "$VENV_PY" ops/data/backfill_person_boxes.py --write 2>&1 | tail -3 \
    || echo "   ! 사람 인식 실패 (계속 진행)"
elif [ "$PERSONS" = 1 ]; then
  echo "   건너뜀 (torch 가 있는 venv 를 찾지 못함: $VENV_PY)"
else
  echo "   건너뜀 (--no-persons)"
fi

echo
echo "── 사람 이미지 폴더 갱신 (04-Archives/person-dataset/dataset)"
if [ "$PERSONS" = 1 ]; then
  python3 ops/data/persons_export.py --out --crop-persons 2>&1 \
    | tail -4 || echo "   ! 추출 실패 (계속 진행)"
else
  echo "   건너뜀 (--no-persons)"
fi

echo
echo "── 방문별 요약 (누가 있었고 무슨 말이 오갔나)"
python3 ops/visit_report.py --limit 5 2>&1 | head -20 || true

echo
echo "── 미리 합성할 답변 목록 갱신 (자주 나온 답변)"
python3 ops/build_tts_cache_list.py --write | tail -3

echo
echo "── 실수요 흡수 (대화 로그 -> 시뮬 태스크 수요, config/real_commands.json)"
if [ -x "$VENV_PY" ]; then
  "$VENV_PY" sim/sim_bridge.py --absorb 2>&1 | tail -3 || echo "   ! 흡수 실패 (계속 진행)"
else
  python3 sim/sim_bridge.py --absorb 2>&1 | tail -3 || echo "   ! 흡수 실패 (계속 진행)"
fi

echo
echo "── 지식 원장 갱신 (실제/시뮬 aggregate, 원본 개인정보 미복사)"
python3 ops/data/knowledge_ledger.py --sync 2>&1 | tail -12 \
  || echo "   ! 지식 원장 갱신 실패 (기존 데이터로 계속)"

echo
echo "── 완료. 다음 단계:"
echo "   • 제안 검토: config/quick_notes.proposed.json"
echo "   • 반영: build_notes.py --merge  또는  quick_notes.json 직접 편집"
echo "   • 지식 원장: ../../04-Archives/knowledge/knowledge_summary.json"
echo "   • 배포: bash ops/deploy.sh   (반영했을 때만)"
