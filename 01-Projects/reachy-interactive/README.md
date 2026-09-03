# reachy-interactive

Reachy 2019 음성·비전·동작 상호작용 시스템. 사무실 복도에 무인으로 켜두고 지나가는 사람과 대화·인사·동작하며 데이터를 쌓는다.

```
[Pi: 로봇]                                        [DGX: 서버]
robot/voice_chat.py                                broker/llm_broker.py
  │  마이크 → STT(구글, 끊기면 Vosk 오프라인)
  │
  ├─ 즉답 노트 (quick_notes)  ─── 로컬, 0초 ──────  (LLM 안 씀)
  ├─ 목 제스처 (say_and_move) ─── 로컬 ───────────  (LLM 안 씀)
  ├─ 물체 주시 (object_vision) ── 로컬 DNN ────────  (LLM 안 씀)
  ├─ 대화 ───────── HTTP /reply ──────────────────▶ Ollama + EXAONE (무료·GPU)
  └─ 팔 동작 ────── HTTP /motion ─────────────────▶ Claude opus (동작 JSON)
                                                     ↑ config/persona.txt, motion_prompt.txt
       경로: Pi → (SSH 터널 -L 8080) → DGX.  ops/pi_tunnel.service
```

**설계 원칙**: 인터넷·LLM 없이도 되는 건 전부 로컬에서 (인사·FAQ·목·물체주시·오프라인 STT). LLM 은 열린 대화(무료 로컬)와 새 동작 생성(opus)에만.

## 폴더

| 폴더 | 실행 위치 | 내용 |
|---|---|---|
| `robot/` | **Pi** | 로봇에서 도는 전부 (음성 루프, 모션, TTS, 비전, 미러) |
| `broker/` | **DGX** | LLM 브로커, 로그 내보내기·노트 마이닝 |
| `sim/` | 브라우저 | 3D 시뮬 뷰어 + 재생 |
| `config/` | DGX | 페르소나·동작 프롬프트, 즉답 노트, STT 교정, 테스트셋 |
| `ops/` | Pi/DGX | 배포·설치 스크립트, systemd 유닛 |

운영(네트워크·부팅·서비스·배포·트러블슈팅)은 **`../../02-Areas/robot-ops/README.md`**.

## robot/ 파일

| 파일 | 역할 |
|---|---|
| `voice_chat.py` | 메인 루프: 마이크→STT→**라우팅**(목/물체/동작/노트/비전/대화)→TTS+모션. 턴 로그, STT 교정, neck 하트비트 |
| `say_and_move.py` | TTS(edge>gtts>espeak), 말하기/생각/idle 머리모션, `point_head`, **`run_head_gesture`**(목 제스처) |
| `motion_exec.py` | 동작 안전층(화이트리스트·클램프·SAFE_LIMITS·속도·FK 충돌·온도) + 50Hz 실행기 + 손 시야추적 + 레디포즈·대기 숨쉬기 |
| `motion_presets.py` | 프리셋 (wave/greet/bow/handshake/pick_to_tray/hug_open/cheer/antenna_dance) |
| `quick_notes.py` | 단순 패턴(인사/FAQ) 즉답 — LLM 안 거치고 `config/quick_notes.json` 에서 바로 |
| `presence.py` | 얼굴 검출 + 프레임 차분 움직임 감지(로컬). 사람 쪽으로 고개(`--attend`) |
| `hallway.py` | 복도 데모(`--hallway`): 방문자 인사·말걸기 유도, 자기 목소리 에코 가드, 이벤트 로그 |
| `object_vision.py` | MobileNet-SSD 로 물체 검출 → 목을 그쪽으로 조준(비전 유도 파지 1단계) |
| `custom_hands.py` | 실물 관절: 오른손 `gripper_no_wrist_roll`, 왼손 `wrist_pitch_only` |
| `base_pose.py` | 로봇 연결(`connect()`)·기본자세·stiffen/relax |
| `llm_client.py` | 브로커 HTTP 클라이언트 (stdlib만, Py3.7) |
| `state_mirror.py` | 실물 엔코더값을 ws(6171)로 방송 → 시뮬 미러 |
| `calibrate_real.py` / `snap_view.py` | 관절 방향 대조 / 카메라 프레임 저장 |

## 실행

평소엔 **전부 서비스로 자동**(전원만 켜면 됨). 아래는 손으로 돌릴 때.

```bash
# [Pi] 복도 데모 (서비스가 이걸 실행한다)
python3 voice_chat.py --motions --hallway --stt auto --fixed-energy 550 \
  --url http://127.0.0.1:8080 --token <TOKEN>

# [Pi] 조용한 모드(불러야 반응) / STT 튜닝
python3 voice_chat.py --motions --url ... --token ...
python3 voice_chat.py --stt-test            # 들은 것 + captured RMS 만 출력

# [DGX] 브로커 (서비스: systemctl --user restart reachy-broker)
python3 broker/llm_broker.py --host 127.0.0.1 --port 8080 \
  --backend ollama --ollama-model exaone3.5:7.8b --token <TOKEN> \
  --system-prompt-file config/persona.txt \
  --motion-model opus --motion-prompt-file config/motion_prompt.txt

# [DGX] 로그 당겨와 분석·개선
bash ops/daily_update.sh
```

**설치 스크립트**(1회): `ops/install_ollama.sh`(DGX 로컬 LLM), `ops/install_vosk.sh`(오프라인 STT), `ops/install_object_vision.sh`(물체 검출 모델). 대용량 모델은 git 제외.

Pi 의존성: `pip3 install gTTS edge-tts SpeechRecognition vosk` + `sudo apt install mpg123 python3-pyaudio flac`

## 라우팅 (한 발화가 어디로 가나)

순서대로 검사하고 처음 걸리는 곳으로 간다:

1. **STT 교정** — `config/stt_corrections.json` (양파→양팔 등 49개) + 이름 오인식. *라우팅 전*에 적용되어 동작 인식률까지 올린다
2. **에코 가드** — 로봇 자기 인사말이 마이크로 되돌아온 것 버림
3. **목 제스처** (`wants_head_gesture`) — 고개/목 + 방향·동작 → 로컬 즉시 실행
4. **물체 주시** (`wants_object_look`) — "저거 봐/물건 확인해봐" → 검출 후 목 조준
5. **팔 동작** (`wants_motion`) — 신체어∧동작어 또는 제스처명사 → opus 생성/프리셋
6. **종료어** — 근사-정확 매칭만(복도 모드에선 프로그램 안 끝내고 대화만 종료)
7. **즉답 노트** — 인사/FAQ 35종
8. **비전** → 카메라 프레임 첨부, 그 외 → **대화 LLM**

회귀 테스트: `config/hallway_testset.json`(38케이스), `config/routing_testset.json`.

## 동작 파이프라인 (팔)

- Opus 출력: `{"say", "preset"|null, "moves"|null}`. moves = 키프레임(pose+duration) 또는 `{"grasp":"close"/"open"}`
- **안전층(모델 불신)**: 관절 화이트리스트, 실한계 클램프 + SAFE_LIMITS, 평균속도 상한, 팔 전체 FK 충돌검사(몸통·머리·정중선·양팔거리), 온도게이트 55°C, 뮤텍스. **실행 직전 실제 자세로 충돌 재검사**(사람이 팔을 손으로 옮겨놨을 수 있음)
- 실행: 단일 50Hz 스레드, 명령값 FK 로 손 시야추적, 키프레임마다 실측 동기화
- 상태: 시작(인사+레디포즈) → 대기(idle+팔 숨쉬기) → 생각/말하기 → 동작(손추적) → 레디복귀 → 3분후 자동이완

## 기능 노트

**대화 (무료 로컬 LLM)**
- DGX 의 Ollama + **EXAONE 3.5 7.8B**(한국어 네이티브). V100 GPU 로 ~100 tok/s, 응답 **0.7~1.5초**, 비용 0
- 브로커가 발화용으로 후처리: 이모지 제거, **3인칭 교정**(리치는→저는), 괄호 제거, KIRO→키로, **길이 제한**(1문단·3문장) — 작은 모델은 지침만으론 안 지킨다
- 페르소나(`config/persona.txt`): 성격·복도 지침·STT 관대해석·실행약속 금지 규칙

**음성 인식**
- `--stt auto` = 구글 우선(정확) + 끊기면 **Vosk 오프라인** 폴백. `--stt vosk` 는 완전 오프라인
- `--fixed-energy N` 으로 감지 임계값 고정(동적 조정 끔). 복도 노이즈 바닥이 ~420 RMS 라 **550** 사용. 매 발화 `captured RMS` 를 로그에 남겨 튜닝 근거로
- 마이크 스트림 상시 개방 + 듣기 직전 flush → 발화 앞부분 잘림 방지

**지연 가리기**
- 응답 대기 중 `--ack-delay`(0.8s) 후부터 뜸들이는 소리를 이어 붙이고 머리는 생각 모션. 응답 오면 즉시 컷 (GPU 전환으로 대부분 필요 없어짐)

**사람·물체 인지 (로컬·무료)**
- `presence.py`: 얼굴 검출 상시, `--attend` 면 idle 이 고개를 사람 쪽으로. 얼굴 없어도 프레임 차분으로 지나가는 움직임에 반응
- `object_vision.py`: "저거 봐" → MobileNet-SSD 검출(~0.6s) → 목 조준 → 뭘 봤는지 말함
- 복도 모드: 음성 인사 1시간 간격(소음 방지), 그 사이 새 방문자엔 **말 없이 손만 흔듦**(5분 간격), 대화 중엔 완전 침묵

**머리 움직임**
- 목은 Orbita 3디스크 병렬 로봇 — 시선으로 제어하므로 팔 키프레임 파이프라인 밖. `run_head_gesture` 로 끄덕임/도리도리/상하좌우/젖히기/한바퀴를 로컬 실행
- **속도 상한 설정이 없다** → 글라이드 시간이 유일한 제어(begin_ramp 1.3s, home 1.8s, 핸드팔로우 램프 1.2s·τ0.35)

## 지속적 개선 루프 (로그 → 분석 → 반영)

운영 중 모든 턴이 Pi `~/reachy_logs/<세션>/events.jsonl` 에 쌓인다 (대화·동작·노트·비전·복도이벤트 + STT엔진·LLM지연·마이크RMS + 프레임).

```bash
bash ops/daily_update.sh     # ① 로그 pull → ② 다이제스트 → ③ 노트 제안 → ④ Pi 원시로그 정리
```
- **다이제스트**(`ops/log_digest.py`): 종류별 건수, 복도 통계(등장/인사/체류), 자주 나온 발화, 동작 실패 사유, STT 엔진 분포·LLM 지연, **마이크 RMS 분포**(→ `--fixed-energy` 제안값)
- **노트 제안**(`broker/build_notes.py`): 반복 발화를 *로그에 실제로 나온 답변*으로 제안 → `quick_notes.proposed.json` (할루시네이션 방지)
- **반영**: 제안 검토 → `quick_notes.json` / `stt_corrections.json` 편집 → `bash ops/deploy.sh`
- ⚠️ 제안이 STT garbage("이다", "노 는")면 반영하지 말 것 — 그건 노트가 아니라 인식 문제

## 시뮬레이터

- `sim/sim_viewer.html`: 실물 CAD(reachy.glb) 3D 렌더. [실물](6171)/[시뮬](6172) 전환, 캘리브레이션 패널
- 전 관절 로컬 z축, 부호는 실물 대조로 확정. neck 은 gaze 방향 근사
- `sim/sim_play.py --candidates`: 생성 동작 후보를 시뮬에서 재생·채택 → 프리셋 승격

## 조정 손잡이

| 항목 | 위치 |
|---|---|
| 마이크 감도 | `--fixed-energy`(현재 550). 튜닝은 `--stt-test` 로 captured RMS 보며 |
| 마이크 게인 | `ops/respeaker_gain.py --max-gain-db --desired-level` (부팅 서비스에 반영) |
| 대화 모델·길이 | 브로커 `--ollama-model`, `OllamaBackend(num_predict)`, `spoken_trim()` |
| 성격·말투 | `config/persona.txt` (수정 후 `systemctl --user restart reachy-broker`) |
| 인사 빈도 | `robot/hallway.py` greet_cooldown / gesture_cooldown / absence_reset |
| 시선 기본 높이 | `say_and_move.GAZE_TILT`(-0.15) 또는 `--gaze-tilt` |
| 고개 추적 방향 | `IdleMotion.ATTEND_SIGN` (반대로 돌면 -1.0) |
| 안전 가동범위 | `motion_exec.SAFE_LIMITS` (shoulder_roll ±80 등) |
| 레디자세·유지토크·자동이완 | `motion_exec` READY_POSE / HOLD_TORQUE / SETTLE_AFTER |
| 집기 좌표 | `motion_presets.pick_to_tray` + `config/motion_prompt.txt` 환경절 |
| 시뮬 관절 축/부호 | `sim/sim_viewer.html` J 테이블 |
