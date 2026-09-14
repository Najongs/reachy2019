#!/bin/bash
# 운영 로그는 보관고 원시층에 모였다 (2026-09-14)
OPS_LOGS=/home/kiro-ai/NAJY/reachy-2019/04-Archives/raw/ops-logs
# 로봇 전체 상태를 한 화면에. DGX 에서 실행한다.
#
#   bash ops/status.sh          # 요약
#   bash ops/status.sh -v       # 최근 로그까지
#
# 두 대에 흩어진 서비스를 매번 따로 확인하다 보면 하나가 조용히 죽어 있어도
# 며칠씩 모른다. 실제로 /health 가 Ollama 를 안 보던 시절엔 그런 식으로
# 하루치 대화가 통째로 캔 답변이었다.
set -u
VERBOSE=${1:-}
PI="ssh -p 2222 -o BatchMode=yes -o ConnectTimeout=8 pi@localhost"

green() { printf '\033[32m%s\033[0m' "$1"; }
red()   { printf '\033[31m%s\033[0m' "$1"; }
mark()  { [ "$1" = "active" ] && green "●" || red "✗"; }

echo "════ DGX ($(hostname)) ════"
for u in ollama.service reachy-broker.service; do
  st=$(systemctl --user is-active "$u" 2>/dev/null)
  printf '  %s %-24s %s  (재시작 %s회)\n' "$(mark "$st")" "${u%.service}" "$st" \
    "$(systemctl --user show "$u" -p NRestarts --value 2>/dev/null)"
done

H=$(curl -s -m 10 -w '\n%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)
CODE=$(printf '%s' "$H" | tail -1); JSON=$(printf '%s' "$H" | head -n -1)
if [ "$CODE" = "200" ]; then
  printf '  %s 브로커 /health           %s\n' "$(green '●')" \
    "$(printf '%s' "$JSON" | python3 -c 'import json,sys;d=json.load(sys.stdin);print("모델 %s, 적재 %s, 가동 %d분"%(d.get("model"),d.get("loaded"),d.get("uptime_s",0)//60))' 2>/dev/null)"
else
  printf '  %s 브로커 /health           HTTP %s %s\n' "$(red '✗')" "$CODE" "$JSON"
fi

curl -s -m 10 http://127.0.0.1:8080/stats 2>/dev/null | python3 -c '
import json, sys
try: d = json.load(sys.stdin)
except Exception: sys.exit()
c = d.get("counts", {})
lat = d.get("latency_s", {})
if not c:
    print("  · 아직 대화가 없습니다 (브로커 기동 후)")
for kind in ("stream", "reply", "motion"):
    n = c.get(kind, 0)
    if not n: continue
    err = c.get(kind + "_error", 0)
    name = {"stream":"대화(스트리밍)","reply":"대화(한번에)","motion":"동작생성"}[kind]
    line = "  · %-14s %4d건" % (name, n)
    if err: line += "  실패 %d건 (%.0f%%)" % (err, 100.0*err/n)
    L = lat.get(kind)
    if L: line += "  첫소리 p50 %.2fs / p90 %.2fs" % (L["p50"], L["p90"])
    print(line)
if d.get("last_error"):
    print("  · 마지막 오류(%d분 전): %s" % (d["last_error_ago_s"]//60, d["last_error"][:70]))
' 2>/dev/null

# 놀고 있는 카드(수백 MiB 이하)는 빼고, 실제로 모델을 물고 있는 것만 보인다.
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' '$2 > 1000 {printf "GPU%s %.1fGB  ", $1, $2/1024}')
[ -n "$GPU" ] && echo "  · 모델이 올라간 GPU: $GPU"

echo
echo "════ Pi (로봇) ════"
if ! $PI true 2>/dev/null; then
  printf '  %s 로봇에 닿지 못합니다 (터널 확인: systemctl status pi_tunnel)\n' "$(red '✗')"
  exit 1
fi

$PI 'for u in voice_chat pi_tunnel pi_viewer respeaker_gain; do
       printf "  %s %-24s %s  (재시작 %s회)\n" \
         "$([ "$(systemctl is-active $u)" = active ] && printf "\033[32m●\033[0m" || printf "\033[31m✗\033[0m")" \
         "$u" "$(systemctl is-active $u)" "$(systemctl show $u -p NRestarts --value)"
     done
     echo "  · 가동 $(uptime -p 2>/dev/null | sed "s/up //"), 메모리 $(free -m | awk "/^Mem/{print \$3\"/\"\$2\"MB\"}"), 디스크 $(df -h / | awk "NR==2{print \$5}")"
     S=$(cd ~/Documents && python3 -c "
from sleep_mode import SleepWatcher
print(\"근무시간 밖(잘 시간)\" if SleepWatcher(None).in_quiet_hours() else \"근무시간\")" 2>/dev/null)
     # 심장박동 줄이 지금 상태다. 로그 전체에서 밝기를 긁으면 모터가 꺼져
     # 카메라가 없어진 뒤에도 어제 낮 값을 보여 준다.
     HB=$(grep -a "심장박동" ~/reachy_logs/voice_chat.log 2>/dev/null | tail -1)
     if [ -n "$HB" ]; then
       AGE=$(( $(date +%s) - $(stat -c %Y ~/reachy_logs/voice_chat.log) ))
       echo "  · 지금 $S | ${HB#*심장박동: } (로그 ${AGE}초 전)"
     else
       echo "  · 지금 $S | 심장박동 아직 없음 (기동 직후)"
     fi
     M=$(grep -ac "연결하지 못했습니다" ~/reachy_logs/voice_chat.log 2>/dev/null)
     tail -n 200 ~/reachy_logs/voice_chat.log 2>/dev/null | grep -q "움직임 없이 소리만" \
       && echo "  · 모터: 꺼져 있음 (소리만 내는 모드)" || echo "  · 모터: 연결됨"
     T=$(ls -t ~/reachy_logs/*/events.jsonl 2>/dev/null | head -1)
     [ -n "$T" ] && echo "  · 오늘 턴 기록: $(wc -l < "$T")건 ($(basename $(dirname "$T")))"
     ' 2>/dev/null

if [ "$VERBOSE" = "-v" ]; then
  echo
  echo "════ 최근 로그 (Pi) ════"
  $PI 'tail -n 12 ~/reachy_logs/voice_chat.log' 2>/dev/null | sed 's/^/  /'
  echo
  echo "════ 워치독 ════"
  tail -n 5 "$OPS_LOGS/watchdog.log" 2>/dev/null | sed 's/^/  /' || echo "  (기록 없음 = 여태 정상)"
fi
