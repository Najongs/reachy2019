#!/bin/bash
# 화면 밝기를 시간대별로 뽑는다 - 수면 모드 문턱값을 실측으로 정하기 위한 것.
#
# presence.py 가 5분마다 "화면 밝기 0.577" 을 로그에 남긴다. 하룻밤 돌린 뒤
# 이걸 보면 낮과 밤이 실제로 몇 인지 알 수 있고, 그 사이에 문턱을 놓으면 된다.
#
#   로봇에서:  bash brightness_report.sh
#   DGX에서:   ssh -p 2222 pi@localhost 'bash ~/Documents/brightness_report.sh'
LOG=${1:-$HOME/reachy_logs/voice_chat.log}

grep -a "화면 밝기" "$LOG" | awk '
  { for (i=1; i<=NF; i++) if ($i == "밝기") v=$(i+1) }
  { split($2, t, ":"); h=t[1]; n[h]++; s[h]+=v
    if (v+0 < lo[h] || n[h]==1) lo[h]=v+0
    if (v+0 > hi[h]) hi[h]=v+0 }
  END {
    if (length(n) == 0) { print "  아직 기록이 없습니다 (5분마다 남습니다)"; exit }
    printf "  %-6s %6s %6s %6s %6s\n", "시각", "표본", "최소", "평균", "최대"
    for (h in n) printf "  %-6s %6d %6.3f %6.3f %6.3f\n", h"시", n[h], lo[h], s[h]/n[h], hi[h]
  }' | sort
echo
echo "  문턱값은 '밤 최대' 와 '낮 최소' 사이에 놓는다."
echo "  voice_chat.service 의 --dark-below / --light-above 로 고친다."
