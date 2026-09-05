#!/bin/bash
# DGX -> Pi 원버튼 배포 (역터널 2222 필요: pi_tunnel.service).
# Pi 는 모든 파일을 ~/Documents/ 에 평면으로 둔다(경로 고정). PARA 구조는 DGX/git 에만.
set -e
cd "$(dirname "$0")/.."          # reachy-interactive/
ROBOT=robot; SIM=sim; OPS=ops; CFG=config
FILES="$ROBOT/base_pose.py $ROBOT/custom_hands.py $ROBOT/llm_client.py \
$ROBOT/motion_exec.py $ROBOT/motion_presets.py $ROBOT/say_and_move.py \
$ROBOT/snap_view.py $ROBOT/voice_chat.py $ROBOT/state_mirror.py $ROBOT/calibrate_real.py \
$ROBOT/quick_notes.py $ROBOT/presence.py $ROBOT/hallway.py $ROBOT/sleep_mode.py $ROBOT/object_vision.py $ROBOT/music.py $ROBOT/camera_check.py $ROBOT/person_db.py \
$SIM/sim_play.py $SIM/sim_viewer.html $SIM/sim_eval.py $SIM/sim_safety.py \
$CFG/quick_notes.json $CFG/stt_corrections.json $CFG/cached_lines.json \
$CFG/campaign_lines.json \
$OPS/pi_tunnel.service $OPS/pi_viewer.service $OPS/respeaker_gain.py \
$OPS/brightness_report.sh $OPS/logrotate-reachy $OPS/pi_watchdog.sh"
# reachy.glb 는 대용량 → 03-Resources 에서 별도(있을 때만)
GLB=../../03-Resources/3d-models/reachy.glb

scp -P 2222 -o BatchMode=yes $FILES pi@localhost:~/Documents/
[ -f "$GLB" ] && scp -P 2222 -o BatchMode=yes "$GLB" pi@localhost:~/Documents/reachy.glb

echo "── 배포 완료. 버전 대조:"
for f in $FILES; do
  b=$(basename "$f")
  L=$(md5sum "$f" | cut -d' ' -f1)
  R=$(ssh -p 2222 -o BatchMode=yes pi@localhost "md5sum ~/Documents/$b | cut -d' ' -f1")
  [ "$L" = "$R" ] && echo "  ✓ $b" || echo "  ✗ $b 불일치!"
done
