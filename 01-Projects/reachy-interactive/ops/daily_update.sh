#!/bin/bash
# 하루치 로그 당겨오기 + 활동 다이제스트 + 노트 제안 (하루 1회, 수동 실행).
#
#   bash ops/daily_update.sh              # 당겨오기 + 요약 + 노트 제안(quick_notes.proposed.json)
#   bash ops/daily_update.sh --since 2026-09-03   # 그 날짜 이후만 요약
#   bash ops/daily_update.sh --merge      # 제안 중 안전·일관된 것 자동 승격까지
#
# 노트는 기본적으로 '제안'만 만든다 (config/quick_notes.proposed.json). 검토 후
# --merge 또는 build_notes.py --merge 로 quick_notes.json 에 반영하고 deploy.sh 배포.
set -e
cd "$(dirname "$0")/.."                                  # reachy-interactive/
PI_LOGS=../../04-Archives/conversation-logs/pi           # 저장소 로그 보관 위치
SINCE=""; MERGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --since) SINCE="--since $2"; shift 2 ;;
    --merge) MERGE="--merge"; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$PI_LOGS"

echo "── [1/3] Pi 로그 당겨오기 (역터널 2222)"
if ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=10 pi@localhost true 2>/dev/null; then
  # 작은 로그(<1MB) — 매번 전체를 scp 로 가져와 로컬 보관본 갱신.
  scp -P 2222 -o BatchMode=yes -o ConnectTimeout=10 -q -r \
    pi@localhost:'~/reachy_logs/*' "$PI_LOGS/" 2>/dev/null \
    && echo "   가져오기 완료" || echo "   ! 일부 파일 복사 실패 - 로컬 보관본 사용"
else
  echo "   ✗ Pi 연결 불가 (터널 2222 확인: systemctl status pi_tunnel). 로컬 보관본으로 계속."
fi
turns=$(cat "$PI_LOGS"/*/events.jsonl 2>/dev/null | wc -l)
echo "   현재 총 턴 수: $turns"

echo
echo "── [2/3] 활동 다이제스트"
python3 ops/log_digest.py "$PI_LOGS" $SINCE

echo
echo "── [3/3] 노트 제안 (실제 로그 답변 재사용, 실행약속/동작류 제외)"
python3 broker/build_notes.py --logs "$PI_LOGS" $MERGE

echo
echo "── 완료. 다음 단계:"
echo "   • 제안 검토: config/quick_notes.proposed.json"
echo "   • 반영: build_notes.py --merge  또는  quick_notes.json 직접 편집"
echo "   • 배포: bash ops/deploy.sh   (반영했을 때만)"
