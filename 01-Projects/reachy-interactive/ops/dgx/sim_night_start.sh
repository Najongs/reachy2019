#!/bin/bash
# 야간 시뮬 학습 시작 - 로봇 수면시간(20~08 KST)에만.
#
# 이유(사용자 결정 2026-09-07): 시뮬과 실물이 같은 opus CLI 를 나눠
# 쓴다. 낮에 시뮬이 돌면 실물이 동작 답변을 못 한다.
# 시각 판정은 스크립트가 KST 로 직접 한다 - 서버 타임존이 무엇이든
# (UTC->KST 전환 중이든) cron 은 30분마다 부르기만 하면 된다.
KH=$(TZ=Asia/Seoul date +%H)
if [ "$KH" -lt 20 ] && [ "$KH" -ge 8 ]; then
    exit 0                                  # 낮 - 실물 몫
fi
BASE=/home/kiro-ai/NAJY/reachy-2019/01-Projects/reachy-interactive
PIDF=$BASE/sim_data/director.pid
cd "$BASE" || exit 1
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    exit 0                                  # 이미 돌고 있음
fi
PY=/home/kiro-ai/NAJY/trossen-ai-simulation/.venv/bin/python3
# 학습 전에 그날의 실데이터(대화 로그 + 사람 사진)부터 - 하루 한 번.
# 사용자 결정: 업데이트에는 텍스트 대화와 사람 데이터 갱신이 동행한다.
KD=$(TZ=Asia/Seoul date +%F)
STAMP=$BASE/sim_data/daily_update.stamp
if [ "$(cat "$STAMP" 2>/dev/null)" != "$KD" ]; then
    echo "$(TZ=Asia/Seoul date '+%F %T KST') 실데이터 일일 갱신 시작" \
        >> sim_data/night_schedule.log
    timeout 3000 bash ops/daily_update.sh >> sim_data/daily_update.log 2>&1 \
        && echo "$KD" > "$STAMP" \
        || echo "$(TZ=Asia/Seoul date '+%F %T KST') 일일 갱신 실패 - 내일 재시도" \
            >> sim_data/night_schedule.log
fi
setsid nohup "$PY" sim/sim_director.py --hours 11.5 --token reachy2019 \
    --seed "$(date +%j)" > sim_data/director.log 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "$(TZ=Asia/Seoul date '+%F %T KST') 야간 시뮬 시작 (pid $(cat "$PIDF"))" \
    >> sim_data/night_schedule.log
