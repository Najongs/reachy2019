#!/bin/bash
# 하루치 로그 당겨오기 + 활동 다이제스트 + 노트 제안 (하루 1회, 수동 실행).
#
#   bash ops/daily_update.sh              # 당겨오기 + 요약 + 노트 제안(quick_notes.proposed.json)
#   bash ops/daily_update.sh --since 2026-09-03   # 그 날짜 이후만 요약
#   bash ops/daily_update.sh --merge      # 제안 중 안전·일관된 것 자동 승격까지
#   bash ops/daily_update.sh --no-clean   # Pi 로그 정리(voice_chat.log 비우기) 생략
#
# 로봇은 계속 켜져 있으니 logrotate 대신, 여기서 로그를 안전히 당겨온 뒤(=DGX 보관본
# 갱신) Pi 의 커지는 voice_chat.log 만 비운다. 세션 events.jsonl 은 건드리지 않는다.
# 노트는 기본 '제안'만 만든다 (config/quick_notes.proposed.json). 검토 후 --merge 로 반영.
set -e
cd "$(dirname "$0")/.."                                  # reachy-interactive/
PI_LOGS=../../04-Archives/conversation-logs/pi           # 저장소 로그 보관 위치
SINCE=""; MERGE=""; CLEAN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --since) SINCE="--since $2"; shift 2 ;;
    --merge) MERGE="--merge"; shift ;;
    --no-clean) CLEAN=0; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$PI_LOGS"

echo "── [1/3] Pi 로그 당겨오기 (역터널 2222)"
PULLED=0
if ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=10 pi@localhost true 2>/dev/null; then
  # 작은 로그(<1MB) — 매번 전체를 scp 로 가져와 로컬 보관본 갱신.
  if scp -P 2222 -o BatchMode=yes -o ConnectTimeout=10 -q -r \
       pi@localhost:'~/reachy_logs/*' "$PI_LOGS/" 2>/dev/null; then
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
