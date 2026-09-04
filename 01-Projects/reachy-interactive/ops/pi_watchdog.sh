#!/bin/bash
# 로봇 대화 루프가 '살아 있는 척하며 멎은' 상태를 잡는다. cron 이 5분마다.
#
#   crontab -e  ->  */5 * * * * /home/pi/Documents/pi_watchdog.sh
#
# systemd 의 Restart=always 는 프로세스가 죽어야 반응한다. 그런데 실제로
# 겪은 고장은 대부분 '떠 있는데 멎은' 쪽이었다 - 마이크 스트림이 물리거나,
# 카메라가 프레임을 안 주거나, SDK 호출이 영영 안 돌아오거나. 그때 로봇은
# 복도에 멀쩡히 서서 아무 반응도 하지 않는다.
#
# voice_chat 이 5분마다 '심장박동' 을 로그에 남기므로, 그게 끊기면 멎은 것이다.
set -u
LOG=$HOME/reachy_logs/voice_chat.log
WD=$HOME/reachy_logs/watchdog.log
STALE=900               # 심장박동이 이만큼 끊기면 멎은 것으로 본다(초)
COOLDOWN=600            # 같은 이유로 연달아 재시작하지 않는다
STAMP=$HOME/reachy_logs/.last_restart

log() {
  echo "$(date '+%F %T') $*" >> "$WD"
  if [ "$(wc -l < "$WD" 2>/dev/null || echo 0)" -gt 2000 ]; then
    tail -1000 "$WD" > "$WD.tmp" && mv "$WD.tmp" "$WD"
  fi
}

# 서비스가 아예 안 돌면 systemd 에 맡긴다 (Restart=always 가 이미 붙어 있다).
[ "$(systemctl is-active voice_chat)" = active ] || exit 0

# 심장박동이 아직 한 번도 안 나왔으면(기동 직후) 기다린다.
LAST=$(grep -a "심장박동" "$LOG" 2>/dev/null | tail -1)
if [ -z "$LAST" ]; then
  # 기동한 지 오래됐는데도 없으면 그건 그것대로 문제다.
  UP=$(systemctl show voice_chat -p ActiveEnterTimestampMonotonic --value)
  NOW=$(awk '{printf "%d", $1*1000000}' /proc/uptime)
  [ $(( (NOW - UP) / 1000000 )) -gt "$STALE" ] && \
    log "기동 후 ${STALE}초가 지나도 심장박동이 없습니다 - voice_chat 재시작" && \
    { touch "$STAMP"; sudo systemctl restart voice_chat; }
  exit 0
fi

# 마지막 심장박동이 언제였나. 로그 줄머리가 "YYYY-MM-DD HH:MM:SS" 형식이 아닐
# 수 있으므로, 파일이 마지막으로 쓰인 시각을 함께 본다.
AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
[ "$AGE" -lt "$STALE" ] && exit 0

if [ -f "$STAMP" ] && [ $(( $(date +%s) - $(stat -c %Y "$STAMP") )) -lt "$COOLDOWN" ]; then
  log "로그가 ${AGE}초째 조용하지만, 최근에 재시작했으므로 기다립니다"
  exit 0
fi

log "로그가 ${AGE}초째 조용합니다 (심장박동 끊김) - voice_chat 재시작"
touch "$STAMP"
sudo systemctl restart voice_chat
