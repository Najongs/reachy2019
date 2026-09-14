#!/bin/bash
# DGX 브로커/Ollama 감시. cron 이 2분마다 돌린다.
#
#   */2 * * * * /home/kiro-ai/NAJY/reachy-2019/01-Projects/reachy-interactive/ops/dgx/broker_watchdog.sh
#
# 예전 판은 /health 가 안 열리면 무조건 브로커만 재시작했다. 그런데 그때
# /health 는 브로커가 떠 있다는 것만 확인했으므로, **Ollama 가 죽어도 늘 정상**
# 이었다 - 워치독은 아무것도 하지 않고 로봇은 하루 종일 캔 답변만 했다.
# 이제 /health 가 Ollama 와 모델까지 확인하고 실패하면 503 을 준다. 그래서
# '누가 아픈지' 를 보고 그 쪽을 고친다.
#
# 재시작 폭주 방지: 같은 대상을 COOLDOWN 안에 두 번 건드리지 않는다. 고장이
# 재시작으로 안 낫는 종류라면 2분마다 영원히 흔드는 것이 더 나쁘다.
set -u
HEALTH=http://127.0.0.1:8080/health
# 런타임 상태(재시작 타임스탬프)와 로그는 보관고 원시층에 둔다 -
# 예전에는 홈의 ~/reachy-ops 에 흩어져 있었다 (2026-09-14 정리).
OPS=/home/kiro-ai/NAJY/reachy-2019/04-Archives/raw/ops-logs
LOG=$OPS/watchdog.log
COOLDOWN=600            # 같은 대상 재시작 최소 간격(초)

mkdir -p "$OPS"

log() {
  echo "$(date '+%F %T') $*" >> "$LOG"
  # 로그가 무한정 자라지 않게 스스로 자른다. 사용자 홈의 파일이라 root
  # logrotate 가 안 돌고, 이 스크립트는 2분마다 실행되므로 여기가 맞는 자리다.
  if [ "$(wc -l < "$LOG" 2>/dev/null || echo 0)" -gt 2000 ]; then
    tail -1000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
  fi
}

# 최근에 건드렸으면 건너뛴다.
recently_restarted() {
  local stamp=$OPS/.last_restart_$1
  [ -f "$stamp" ] || return 1
  local age=$(( $(date +%s) - $(stat -c %Y "$stamp") ))
  [ "$age" -lt "$COOLDOWN" ]
}
mark_restart() { touch "$OPS/.last_restart_$1"; }

restart() {                     # restart <이름> <systemd유닛> <이유>
  if recently_restarted "$1"; then
    log "$3 - 그러나 $1 을 최근에 재시작했으므로 기다립니다"
    return
  fi
  log "$3 - $2 재시작"
  mark_restart "$1"
  systemctl --user restart "$2"
}

BODY=$(curl -s -m 10 -w '\n%{http_code}' "$HEALTH" 2>/dev/null)
CODE=$(printf '%s' "$BODY" | tail -1)
JSON=$(printf '%s' "$BODY" | head -n -1)

# 1) 아예 안 열린다 = 브로커가 죽었다.
if [ -z "$CODE" ] || [ "$CODE" = "000" ]; then
  restart broker reachy-broker.service "/health 에 닿지 못했습니다"
  exit 0
fi

# 2) 200 이면 브로커도 Ollama 도 모델도 정상.
if [ "$CODE" = "200" ]; then
  # 모델이 내려가 있으면(keep_alive 가 풀렸거나 누가 내렸거나) 미리 데워 둔다.
  # 안 그러면 다음 방문객이 40초짜리 적재를 기다린다.
  if printf '%s' "$JSON" | grep -q '"loaded": *false'; then
    log "모델이 내려가 있습니다 - 미리 데웁니다"
    curl -s -m 180 http://127.0.0.1:11434/api/chat \
      -d '{"model":"exaone3.5:7.8b","messages":[{"role":"user","content":"안녕"}],"stream":false,"keep_alive":-1}' \
      >/dev/null 2>&1
  fi
  exit 0
fi

# 3) 503 = 브로커는 살아 있는데 뒤가 아프다. Ollama 부터 고친다.
if printf '%s' "$JSON" | grep -q 'ollama 에 닿지 못했습니다'; then
  restart ollama ollama.service "Ollama 가 응답하지 않습니다: $JSON"
  exit 0
fi

log "브로커가 $CODE 를 돌려줍니다: $JSON"
restart broker reachy-broker.service "브로커 상태가 비정상입니다"
