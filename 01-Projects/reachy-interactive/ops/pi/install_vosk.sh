#!/bin/bash
# Install offline Korean STT (Vosk) on the Pi. Run ON the Pi (needs internet
# once, to fetch the wheel + model). The 253MB model lives on the SD card at
# ~/vosk-ko-model and is NOT in git. After this, voice_chat --stt vosk (and the
# voice_chat.service) recognize speech with zero internet - essential where the
# hallway wifi is unreliable.
#
#   ssh -p 2222 pi@localhost 'bash -s' < ops/pi/install_vosk.sh   # from the DGX
#   # or copy over and run:  bash install_vosk.sh
set -e
MODEL_DIR="$HOME/vosk-ko-model"
MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip"

echo "── vosk 파이썬 패키지 설치 (piwheels armv7l)"
pip3 install --user vosk
python3 -c "import vosk; print('   vosk', getattr(vosk,'__version__',''))"

if [ -d "$MODEL_DIR" ]; then
  echo "── 모델 이미 있음: $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1))"
else
  echo "── 한국어 소형 모델 다운로드 (~82MB, 압축 해제 253MB)"
  cd "$HOME"
  # /tmp 은 tmpfs 100MB 라 홈(SD)에 받는다.
  curl -L -o vosk-ko.zip "$MODEL_URL"
  python3 -c "import zipfile; zipfile.ZipFile('vosk-ko.zip').extractall('.')"
  mv -f vosk-model-small-ko-0.22 "$MODEL_DIR"
  rm -f vosk-ko.zip
  echo "   설치됨: $MODEL_DIR"
fi

echo "── 로드 확인"
python3 - <<PYEOF
import vosk, json
vosk.SetLogLevel(-1)
m = vosk.Model("$MODEL_DIR")
r = vosk.KaldiRecognizer(m, 16000)
r.AcceptWaveform(b"\x00\x00" * 16000)
print("   OK - 무음 결과:", json.loads(r.FinalResult()))
PYEOF
echo "── 완료. 이제 voice_chat.py --stt vosk 로 오프라인 인식 사용."
