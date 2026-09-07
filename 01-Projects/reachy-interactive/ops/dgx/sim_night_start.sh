#!/bin/bash
# 야간 시뮬 학습 시작 (로봇 수면 20~08 KST 에만).
#
# 이유(사용자 결정 2026-09-07): 시뮬과 실물이 같은 opus CLI(브로커
# /motion·/vision)를 나눠 쓴다. 낮에 시뮬이 돌면 세션 한도를 먹어
# 실물이 동작 답변을 못 한다. 로봇이 자는 밤에만 학습한다.
# DGX 시계는 UTC: 20:10 KST = 11:10 UTC 에 cron 이 부른다.
# 중복 검사는 pidfile 로 - pgrep -f 는 부모 셸 문자열에 오탐한다.
BASE=/home/kiro-ai/NAJY/reachy-2019/01-Projects/reachy-interactive
PIDF=$BASE/sim_data/director.pid
cd "$BASE" || exit 1
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    exit 0                                  # 이미 돌고 있음
fi
PY=/home/kiro-ai/NAJY/trossen-ai-simulation/.venv/bin/python3
setsid nohup "$PY" sim/sim_director.py --hours 11.5 --token reachy2019 \
    --seed "$(date +%j)" > sim_data/director.log 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "$(date -u '+%F %T') 야간 시뮬 시작 (pid $(cat "$PIDF"))" \
    >> sim_data/night_schedule.log
