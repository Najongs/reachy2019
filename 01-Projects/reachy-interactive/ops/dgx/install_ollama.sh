#!/bin/bash
# Install the free local chat LLM (Ollama + EXAONE Korean) on the DGX.
#
# IMPORTANT - version pin: the newest Ollama refuses this box's GPUs
# ("NVIDIA driver too old ... required 550 or newer"; the DGX has 535) and
# silently falls back to CPU (~11 tok/s). v0.3.14 ships CUDA 12.2 runners, so
# it drives the V100s with NO driver upgrade and no reboot: ~100 tok/s, replies
# in under a second. Do not "upgrade" this unless the driver moves to 550+.
set -e
V=v0.3.14
DEST=$HOME/ollama-old            # binary+libs for the pinned version
MODELS=$HOME/ollama/models       # model blobs (shared, keep across versions)

[ -x "$DEST/bin/ollama" ] || {
  curl -fsSL -o /tmp/ollama.tgz \
    https://github.com/ollama/ollama/releases/download/$V/ollama-linux-amd64.tgz
  mkdir -p "$DEST" && tar -xzf /tmp/ollama.tgz -C "$DEST" && rm -f /tmp/ollama.tgz
}

mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/ollama.service <<UNIT
[Unit]
Description=Ollama local LLM server (Reachy chat, GPU)
[Service]
Environment=OLLAMA_HOST=127.0.0.1:11434
Environment=OLLAMA_KEEP_ALIVE=-1
Environment=OLLAMA_MODELS=$MODELS
Environment=LD_LIBRARY_PATH=$DEST/lib
ExecStart=$DEST/bin/ollama serve
Restart=always
RestartSec=5
[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now ollama
sleep 6

OLLAMA_HOST=127.0.0.1:11434 "$DEST/bin/ollama" pull exaone3.5:7.8b
# Warm it so the first visitor does not wait for a cold load.
curl -s -m180 http://127.0.0.1:11434/api/chat \
  -d '{"model":"exaone3.5:7.8b","messages":[{"role":"user","content":"안녕"}],"stream":false,"keep_alive":-1}' >/dev/null

echo "GPU 사용 확인:"
journalctl --user -u ollama --no-pager -n 40 | grep -o "library=cuda" | head -1 || echo "  (CPU 폴백 - 로그 확인 필요)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -v ", 0 MiB" | head -2
echo "완료. 브로커: --backend ollama --ollama-model exaone3.5:7.8b"
