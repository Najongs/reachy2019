# Robot Ops — 운영 문서

Reachy 로봇의 하드웨어 특성, 네트워크, 부팅/서비스, 배포, 로그 운영. 코드·기능 설명은 `../../01-Projects/reachy-interactive/README.md`.

## 한 장 요약

| | 무엇이 | 어디서 | 자동시작 |
|---|---|---|---|
| 로봇 음성 루프 | `voice_chat.py --motions --hallway` | Pi | `voice_chat.service` |
| 마이크 게인 | ReSpeaker AGC 설정 | Pi | `respeaker_gain.service` |
| DGX 터널 | SSH 포워딩 | Pi | `pi_tunnel.service` |
| 와이파이 복구 | 2분마다 점검 | Pi | `wifi-ensure.timer` |
| LLM 브로커 | 대화·동작(로컬)+시뮬 평가(Codex) | DGX | `reachy-broker.service` (--user) |
| 로컬 LLM | Ollama + EXAONE | DGX | `ollama.service` (--user) |

전원만 켜면 위가 전부 자동으로 올라온다. DGX 쪽은 `loginctl enable-linger kiro-ai` 로 로그아웃/재부팅에도 유지.

## 하드웨어 (이 로봇 실물)

- **없는 모터** (2026-09-02 버스 스캔): 오른팔 wrist_roll(dxl16), 왼팔 wrist_roll(dxl26)·gripper(dxl27)·힘센서. 기본 손 클래스는 연결 시 죽으므로 `robot/custom_hands.py` 사용
- **왼손 = 3D 프린팅 보관함(트레이)**, 오른손 = 힘센서 그리퍼. pick&place: 오른손이 집고 왼쪽 트레이가 받음
- 테이블 면 ≈ z -0.27 (토르소 기준). 팔 늘어뜨리면 손이 상판 아래로
- 관절: 사용자 프레임. shoulder_pitch 음수=앞/위(들 땐 팔꿈치 굽힐 것 — 편 채 들면 스톨), elbow_pitch 음수=굽힘
- **상완이 몸통과 rest 에서 ~3cm** 뿐 → SAFE_LIMITS 로 shoulder_roll 좁힘
- **목 = Orbita 3디스크 병렬 로봇**(모터 3개). 관절 각도가 아니라 *시선 방향*으로 제어하므로 팔 키프레임 파이프라인에 못 들어간다 → `say_and_move.run_head_gesture` 로 별도 구동(끄덕임·도리도리·상하좌우·젖히기·한바퀴). **속도 상한 설정이 없다**(`Head.moving_speed` 는 무동작) → 글라이드 시간이 유일한 속도 제어
- 아래를 오래 응시하면 과열로 힘빠짐 → GAZE_TILT 완화 + neck 하트비트
- compliant 상태에선 goal_position 무시 → stiffen 직후 seed 재기록(팔 튐 방지)
- Pi: **라즈베리파이 4, armv7l(32bit), Buster, Python 3.7**. 최신 Anthropic SDK 설치 불가 → `llm_client` 는 stdlib urllib
- 마이크: ReSpeaker 4-Mic Array. **ALSA 볼륨이 없고** XMOS DSP AGC 를 USB 벤더 명령으로 설정(`ops/respeaker_gain.py`), 전원 뽑으면 초기화되므로 부팅 서비스로 재적용

## 네트워크

- Pi → 사내 와이파이(SSID `robot`) → 인터넷. DGX 는 인터넷에서 **210.183.240.125:20411**
- **Pi ↔ DGX 직접 라우팅 안 됨** → Pi 가 나가는 SSH 에 포워딩을 실어보낸다:
  - `-L 8080` : Pi 의 127.0.0.1:8080 = DGX 브로커
  - `-R 2222` : DGX 에서 `ssh -p 2222 pi@localhost` = Pi (역방향 배포·로그수집)
  - `-R 8090/6171/6172` : 뷰어·미러
- `ops/pi_tunnel.service` (자동 재연결). 키: `~/.ssh/dgx_tunnel`
- 터널은 **밖으로 나가는 연결**이라 Pi 가 어느 와이파이에 있든 상관없다. 단 그 망이 **20411 아웃바운드를 허용**해야 함(핸드폰 핫스팟이 가장 안전)

### 와이파이 불안정 — 원인과 처방 (2026-09-03 해결)

세 가지가 겹쳐 있었다. 셋 다 적용된 상태:

1. **동글 전력절약** — MT7601U 가 유휴 시 패킷을 버려 **손실 30%**. `ops/70-wifi-powersave-off.rules`(udev) + rc.local 로 항상 off → **손실 0%**
2. **같은 SSID 를 두 AP 가 방송** (`...46:c0` 강함 / `...47:c0` 약함) → wpa_supplicant 가 6초마다 로밍/재접속. 로봇은 고정이므로 **강한 AP 에 BSSID 고정**:
   `/etc/wpa_supplicant/wpa_supplicant.conf` 의 network 블록에 `bssid=6c:71:0d:fd:46:c0`, `bgscan=""`, `scan_freq=2462`
3. **wpa_supplicant 중복 실행** (독립 서비스 + dhcpcd 관리분) → `systemctl disable --now wpa_supplicant.service` (Raspbian 은 dhcpcd 쪽을 쓴다)

**증상별 확인**: `iw dev wlan0 get power_save` → off, `pgrep -c wpa_supplicant` → 1, `sudo wpa_cli -i wlan0 status | grep bssid`, `dmesg | grep deauth`(반복되면 로밍 재발).

### DGX 쪽 스테일 터널

Pi 와이파이가 불결하게 끊기면 DGX sshd 가 죽은 세션과 그 -R 포트를 최대 2시간 붙들어, Pi 재접속이 2222 바인딩에 실패한다(터널 전체가 안 붙음). 처방: `ops/sshd_keepalive.conf` 를 `/etc/ssh/sshd_config.d/` 에 설치(root 필요). 급할 땐 DGX 에서 오래된 `sshd: kiro-ai@notty` 프로세스를 kill.

## 부팅 / 서비스

### Pi

- `voice_chat.service` — 복도 데모 자동 실행, `Restart=always`, 로그 `~/reachy_logs/voice_chat.log`
  - ⚠️ **`After=` 를 넣지 말 것.** `network-online.target`/`respeaker_gain.service` 를 After 로 걸었더니 systemd 가 **ordering cycle** 을 감지해 부팅 시 **start job 자체를 삭제**했다(서비스가 enabled 인데 안 뜸). 준비 상태는 `ExecStartPre` 가 `/dev/ttyUSB2`(로봇 게이트)와 브로커 health 를 기다리는 것으로 충분
- `respeaker_gain.service` — 마이크 AGC 재적용(전원 리셋 대응)
- `pi_tunnel.service` — DGX 터널
- `wifi-ensure.timer` — 2분마다 와이파이 점검/복구
- `reachy-dashboard` 만 남기고 `demo.service`·`reachy-access-point` 는 disabled

수동 조작:
```bash
sudo systemctl stop voice_chat      # 손으로 테스트할 때
sudo systemctl start voice_chat
systemctl status voice_chat
tail -f ~/reachy_logs/voice_chat.log
```

### DGX (--user 서비스, linger 켜짐)

```bash
systemctl --user status ollama reachy-broker
systemctl --user restart reachy-broker      # 페르소나/프롬프트 바꾼 뒤
```
- `ollama.service` — **버전 고정 v0.3.14**. 최신 Ollama 는 드라이버 550+ 를 요구하는데 이 DGX 는 535 라 GPU 를 못 잡고 CPU 로 떨어진다(11 tok/s). v0.3.14 는 CUDA 12.2 런너를 포함해 **드라이버 업그레이드 없이 V100 사용**(100 tok/s). 모델은 `OLLAMA_KEEP_ALIVE=-1` 로 상주
- `reachy-broker.service` — 대화·동작 생성은 로컬 Ollama, 시뮬 접지·평가는 Codex CLI
  - 저장소 기준본: `01-Projects/reachy-interactive/ops/dgx/reachy-broker.service`
- `~/reachy-ops/broker_watchdog.sh` (cron 2분) — health 무응답 시 재시작

## 배포 (DGX → Pi)

역터널(2222) 경유, 한 번에 전송 + md5 대조:
```bash
cd 01-Projects/reachy-interactive && bash ops/deploy.sh
```
Pi 파일은 `~/Documents/` 에 **평면 배치**(경로 고정). 대용량 모델(Vosk 253MB, MobileNet-SSD 23MB)은 git·deploy 대상이 아니라 Pi 에 상주하며 `ops/install_vosk.sh`, `ops/install_object_vision.sh` 로 재설치.

## 로그 / 개선 루프

```bash
bash ops/daily_update.sh        # Pi 로그 pull → 다이제스트 → 노트 제안 → Pi 원시로그 정리
```
- Pi: `~/reachy_logs/<세션>/events.jsonl` + `frames/` — 대화·동작·비전·복도이벤트 + STT엔진/LLM지연/마이크RMS
- 로그 회전은 따로 두지 않고 위 pull 단계에서 `voice_chat.log` 를 비운다(`--no-clean` 로 생략)
- 다이제스트가 **마이크 RMS 분포**를 뽑아주므로 `--fixed-energy` 튜닝 근거로 쓴다

## 자주 겪은 함정 (재발 방지)

- `pkill -f "..."` 은 **자기 자신(쉘 명령줄)** 까지 매칭해 세션을 죽인다 → `pgrep` 로 PID 확인 후 kill
- 서비스가 `enabled` 인데 안 뜨면 `journalctl -b | grep "ordering cycle"` 확인
- 모델/드라이버 조합을 "최신으로 업그레이드"하지 말 것(위 Ollama 항목)
- Pi 로그가 root 소유(systemd append) 라 비울 때 `sudo truncate -s 0` 필요
