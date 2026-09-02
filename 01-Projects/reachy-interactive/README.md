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
| `quick_notes.py` | 단순 패턴(인사/FAQ) 즉답 — CLI 안 거치고 `config/quick_notes.json`에서 바로 응답 |
| `presence.py` | 머리 카메라로 얼굴 검출(로컬·무료). 사람 있으면 idle이 고개를 사람 쪽으로(`--attend`) |

## 실행

```bash
# [DGX] 브로커
python3 broker/llm_broker.py --host 127.0.0.1 --port 8080 \
  --backend claude-cli --cli-model haiku --token <TOKEN> \
  --system-prompt-file config/persona.txt \
  --motion-model opus --motion-prompt-file config/motion_prompt.txt &

# [Pi] 메인 (동작 포함; 전원만 켜면 준비 — 부팅 자동화는 02-Areas 참조)
#   --attend: 사람 얼굴 쪽으로 고개 (얼굴검출은 기본 on, 고개추적만 옵트인)
python3 robot/voice_chat.py --motions --attend --url http://127.0.0.1:8080 --token <TOKEN>

# [DGX] 대화 로그에서 즉답 노트 자동 구축 (로그 갱신 후)
python3 broker/build_notes.py          # 후보만 quick_notes.proposed.json 에
python3 broker/build_notes.py --merge  # 안전·일관된 것만 quick_notes.json 에 승격

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

## 응답성 (CLI 지연 가리기 · 즉답 · 사람 인지)

- **연속 맞장구**: CLI 응답(4~10s)을 기다리는 동안 `--ack-delay`(0.8s) 후부터 뜸들이는 소리를
  이어 붙여 침묵을 없앰. 응답 오면 즉시 컷, 빠른 답(0.8s 이내)은 스킵. 문구는 `voice_chat.prepare_fillers`
- **즉답 노트**(`quick_notes.py` + `config/quick_notes.json`): 인사/이름/위치 같은 단순·정형 패턴은
  CLI 없이 즉시 응답. 매칭은 보수적(짧은 발화만) — 실제 질문은 절대 가로채지 않음. 동작/비전 라우팅이 먼저 걸러짐
- **노트 자동 성장**(`broker/build_notes.py`): 대화 로그에서 자주 나오는 짧은 발화를 찾아 **로그에 실제로 나온 답변**을
  재사용해 제안(할루시네이션 방지, 실행 약속성 답변 제외). `--merge`로 일관·안전한 것만 승격
- **머리 항상 천천히**: Orbita neck 은 속도 상한이 없어(Head.moving_speed 무동작) 글라이드 시간이 유일한 제어 —
  begin_ramp 1.3s, home 1.8s, 핸드팔로우 램프 1.2s·τ0.35, 베이스복귀 하한 2.0s
- **사람 인지**(`presence.py`): 머리 카메라로 얼굴 검출(로컬·무료·상시). `--attend` 시 idle 이 고개를 사람 쪽으로
  (상대 서보, 부호는 `IdleMotion.ATTEND_SIGN`). STT 앞부분 잘림도 수정 — 마이크 스트림 상시 개방 + 듣기 직전 버퍼 flush

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
