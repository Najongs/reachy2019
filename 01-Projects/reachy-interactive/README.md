# reachy-interactive

Reachy 2019 음성·비전·동작 상호작용 시스템.

```
[Pi: 로봇]                                   [DGX: 서버]
robot/voice_chat.py ── STT(구글) ── HTTP ──▶ broker/llm_broker.py
  ├ robot/say_and_move.py  (TTS·머리모션)      ├ haiku 세션  ← 대화 (config/persona.txt)
  ├ robot/motion_exec.py   (동작 검증·실행)     └ opus 세션   ← 동작 JSON (config/motion_prompt.txt)
  ├ robot/motion_presets.py (프리셋)
  ├ robot/state_mirror.py  (실물→시뮬 방송)
  └ robot/custom_hands.py  (실물 관절 구성)
      경로: Pi → (SSH 터널) → DGX. ops/pi_tunnel.service 참조
```

## 폴더

| 폴더 | 실행 위치 | 내용 |
|---|---|---|
| `robot/` | **Pi** | 로봇에서 도는 전부 (음성 루프, 모션, TTS, 미러, 캘리브레이션) |
| `broker/` | **DGX** | LLM 브로커, 로그 내보내기 |
| `sim/` | 브라우저 | 3D 시뮬 뷰어 + 키보드 구동 |
| `config/` | DGX | 시스템 프롬프트(persona/motion), 동작 후보 |
| `ops/` | Pi/DGX | 배포 스크립트, systemd 유닛 |

세부 운영(네트워크·부팅·배포)은 `../../02-Areas/robot-ops/` 참조.

## robot/ 파일

| 파일 | 역할 |
|---|---|
| `voice_chat.py` | 메인 루프: 마이크→STT→라우팅(대화/동작)→TTS+모션. 턴 로그, 이름정규화, neck 하트비트 |
| `say_and_move.py` | TTS(edge>gtts>espeak), 말하기/생각/idle 머리모션, `point_head`, GAZE_TILT |
| `motion_exec.py` | 동작 안전층(관절 화이트리스트·클램프·SAFE_LIMITS·속도·팔 전체 FK 충돌·양팔거리·온도) + 50Hz 실행기 + 손 시야추적 + 레디포즈 |
| `motion_presets.py` | 프리셋 (wave/greet/bow/handshake/pick_to_tray/hug_open/cheer/antenna_dance) |
| `custom_hands.py` | 실물 관절: 오른손 `gripper_no_wrist_roll`, 왼손 `wrist_pitch_only` |
| `base_pose.py` | 로봇 연결(`connect()`)·기본자세·stiffen/relax |
| `llm_client.py` | 브로커 HTTP 클라이언트 (stdlib만, Py3.7). `--direct` 로 API 직접호출도 |
| `state_mirror.py` | 실물 관절 실제 엔코더값을 ws(6171)로 방송 → sim 미러 |
| `calibrate_real.py` | 관절 하나씩 왕복시켜 시뮬과 방향 대조 |
| `snap_view.py` | 머리 숙여 카메라 프레임 저장 |

## 실행

```bash
# [DGX] 브로커
python3 broker/llm_broker.py --host 127.0.0.1 --port 8080 \
  --backend claude-cli --cli-model haiku --token <TOKEN> \
  --system-prompt-file config/persona.txt \
  --motion-model opus --motion-prompt-file config/motion_prompt.txt &

# [Pi] 메인 (동작 포함; 전원만 켜면 준비 — 부팅 자동화는 02-Areas 참조)
python3 robot/voice_chat.py --motions --url http://127.0.0.1:8080 --token <TOKEN>

# [Pi/DGX] 시뮬만
python3 sim/sim_play.py --candidates --url http://127.0.0.1:8080 --token <TOKEN>
#   → 브라우저에서 sim/sim_viewer.html, ws://<host>:6171(실물)/6172(시뮬)
```

Pi 의존성: `pip3 install gTTS edge-tts SpeechRecognition` + `sudo apt install mpg123 python3-pyaudio flac`

## 동작 파이프라인

- 라우팅(voice_chat.wants_motion): 신체어∧동작어 또는 제스처어(강한 것은 단독). 이름 오인식 정규화 포함
- Opus 출력: `{"say", "preset"|null, "moves"|null}`. moves = 키프레임(pose+duration) 또는 `{"grasp":"close"/"open"}`(힘센서)
- 안전층(모델 불신): 관절 화이트리스트, 실한계 클램프 + **SAFE_LIMITS**(기계한계보다 좁은 안전범위), 평균속도 상한, **팔 전체 FK 충돌검사**(몸통·머리·정중선·양팔거리), 온도게이트 55°C, 뮤텍스
- 실행: 단일 50Hz 스레드, 명령값 FK로 손 시야추적, 키프레임마다 실측 동기화(스톨 감지)
- 상태: 시작(인사+레디포즈) → 대기(idle) → 생각/말하기 → 동작(손추적) → 레디복귀 → 3분후 자동이완

## 시뮬레이터

- `sim/sim_viewer.html`: 실물 CAD 모델(reachy.glb) 3D 렌더. [실물](6171)/[시뮬](6172) 전환, [캘리브레이션] 패널
- 전 관절 로컬 z축, 부호는 실물 대조로 확정 (2026-09-02). neck 은 gaze 방향 근사
- `sim/sim_play.py --candidates`: 생성 동작 후보를 시뮬에서 재생·채택 → 프리셋 승격

## 조정 손잡이

| 항목 | 위치 |
|---|---|
| 시선 기본 높이 | `robot/say_and_move.py` GAZE_TILT (-0.15) 또는 `--gaze-tilt` |
| 안전 가동범위 | `robot/motion_exec.py` SAFE_LIMITS (shoulder_roll ±80 등) |
| 레디자세·유지토크·자동이완 | `motion_exec` READY_POSE / HOLD_TORQUE / SETTLE_AFTER |
| 집기 좌표 | `motion_presets.pick_to_tray` + `config/motion_prompt.txt` 환경절 |
| 시뮬 관절 축/부호 | `sim/sim_viewer.html` J 테이블 |
