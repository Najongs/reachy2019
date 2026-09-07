#!/bin/bash
# 야간 시뮬 종료 백스톱 (07:55 KST = 22:55 UTC). 낮의 opus 는 실물 몫.
BASE=/home/kiro-ai/NAJY/reachy-2019/01-Projects/reachy-interactive
PIDF=$BASE/sim_data/director.pid
if [ -f "$PIDF" ]; then
    P=$(cat "$PIDF")
    kill "$P" 2>/dev/null; sleep 5; kill -9 "$P" 2>/dev/null
    rm -f "$PIDF"
fi
# 감독이 낳은 학습 자식도 정리 (pgrep 브래킷 - cron 셸은 자기오탐 없음)
for p in $(pgrep -f "sim_improve[.]py"); do kill -9 "$p" 2>/dev/null; done
echo "$(date -u '+%F %T') 야간 시뮬 종료" >> "$BASE/sim_data/night_schedule.log"
