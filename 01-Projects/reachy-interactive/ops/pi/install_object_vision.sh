#!/bin/bash
# Install the MobileNet-SSD object-detection model on the Pi (for "저거 봐" /
# vision-guided attention). ~23MB, NOT in git. Run ON the Pi (needs internet
# once). After this, voice_chat looks at objects offline.
#
#   ssh -p 2222 pi@localhost 'bash -s' < ops/install_object_vision.sh
set -e
DIR="$HOME/Documents"
PROTO_URL="https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
MODEL_URL="https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

cd "$DIR"
[ -f mnssd.prototxt ]  || curl -L -o mnssd.prototxt  "$PROTO_URL"
[ -f mnssd.caffemodel ] || curl -L -o mnssd.caffemodel "$MODEL_URL"
ls -lh mnssd.prototxt mnssd.caffemodel

python3 - <<'PYEOF'
import cv2
net = cv2.dnn.readNetFromCaffe("mnssd.prototxt", "mnssd.caffemodel")
print("OK - object model loads (MobileNet-SSD, 20 VOC classes)")
PYEOF
echo "완료. voice_chat 이 자동으로 물체 주시(저거 봐/물건 봐)를 사용."
