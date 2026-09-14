#!/bin/bash
# 야간 시뮬 학습 시작 - 로봇 수면시간(18~09 KST, 사용자 지정 근무 9~18)에만.
#
# 시뮬의 MuJoCo GPU 부하와 Codex 평가 호출을 낮 시간 실물 운용과 분리한다.
# 시각 판정은 스크립트가 KST 로 직접 한다 - 서버 타임존이 무엇이든
# (UTC->KST 전환 중이든) cron 은 30분마다 부르기만 하면 된다.
KH=$(TZ=Asia/Seoul date +%H)
if [ "$KH" -lt 18 ] && [ "$KH" -ge 9 ]; then
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
# 밤 창(20~08)이 자정을 넘으므로 '그 밤이 시작된 저녁 날짜' 를 키로 쓴다
if [ "$KH" -lt 9 ]; then
    KD=$(TZ=Asia/Seoul date -d yesterday +%F)
else
    KD=$(TZ=Asia/Seoul date +%F)
fi
STAMP=$BASE/sim_data/daily_update.stamp
if [ "$(cat "$STAMP" 2>/dev/null)" != "$KD" ]; then
    echo "$(TZ=Asia/Seoul date '+%F %T KST') 실데이터 일일 갱신 시작" \
        >> sim_data/night_schedule.log
    timeout 3000 bash ops/daily_update.sh >> sim_data/daily_update.log 2>&1 \
        && echo "$KD" > "$STAMP" \
        || echo "$(TZ=Asia/Seoul date '+%F %T KST') 일일 갱신 실패 - 내일 재시도" \
            >> sim_data/night_schedule.log
fi
# 야간 최적화는 실제 대화용 브로커/Codex 쿼터와 분리한다. Oracle 기반
# 결정론 CEM은 비전/평가 모델 없이도 행동 파라미터를 개선할 수 있다.
setsid nohup "$PY" sim/sim_director.py --hours 14.5 --offline \
    --seed "$(date +%j)" > sim_data/director.log 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "$(TZ=Asia/Seoul date '+%F %T KST') 야간 시뮬 시작 (pid $(cat "$PIDF"))" \
    >> sim_data/night_schedule.log
