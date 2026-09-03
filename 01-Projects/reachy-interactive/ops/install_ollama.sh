#!/bin/bash
# Install a local free LLM (Ollama + EXAONE Korean) on the DGX for chat.
# The DGX driver (535) is too old for the newest Ollama GPU runners, so this
# runs CPU inference (fine for a 7.8B on the DGX's many cores, ~4-6s/reply,
# kept warm). Chat uses this; motion generation stays on Claude opus.
#
#   bash ops/install_ollama.sh
set -e
cd ~
V=v0.5.7
[ -x ~/ollama/bin/ollama ] || {
  curl -fsSL -o /tmp/ollama.tgz \
    https://github.com/ollama/ollama/releases/download/$V/ollama-linux-amd64.tgz
  mkdir -p ~/ollama && tar -xzf /tmp/ollama.tgz -C ~/ollama && rm -f /tmp/ollama.tgz
}
# user service so it stays up + keeps the model warm (no cold-load stalls)
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/ollama.service <<UNIT
[Unit]
Description=Ollama local LLM server (Reachy chat)
[Service]
Environment=OLLAMA_HOST=127.0.0.1:11434
Environment=OLLAMA_KEEP_ALIVE=-1
Environment=OLLAMA_MODELS=$HOME/ollama/models
Environment=LD_LIBRARY_PATH=$HOME/ollama/lib
ExecStart=$HOME/ollama/bin/ollama serve
Restart=always
RestartSec=5
[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now ollama
sleep 4
OLLAMA_HOST=127.0.0.1:11434 ~/ollama/bin/ollama pull exaone3.5:7.8b
echo "완료. 브로커를 --backend ollama --ollama-model exaone3.5:7.8b 로 실행."
