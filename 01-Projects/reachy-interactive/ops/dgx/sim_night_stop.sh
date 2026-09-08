#!/bin/bash
# 야간 시뮬 종료 - 낮(09~18 KST, 로봇 근무시간)이면 내린다. 서버 타임존 무관.
KH=$(TZ=Asia/Seoul date +%H)
if [ "$KH" -ge 18 ] || [ "$KH" -lt 9 ]; then
    exit 0                                  # 밤 - 계속 돌게 둔다
fi
BASE=/home/kiro-ai/NAJY/reachy-2019/01-Projects/reachy-interactive
PIDF=$BASE/sim_data/director.pid
if [ -f "$PIDF" ]; then
    P=$(cat "$PIDF")
    kill "$P" 2>/dev/null; sleep 5; kill -9 "$P" 2>/dev/null
    rm -f "$PIDF"
    echo "$(TZ=Asia/Seoul date '+%F %T KST') 야간 시뮬 종료" \
        >> "$BASE/sim_data/night_schedule.log"
fi
for p in $(pgrep -f "sim_improve[.]py"); do kill -9 "$p" 2>/dev/null; done
