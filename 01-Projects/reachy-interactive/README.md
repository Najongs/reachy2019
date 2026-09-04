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
| `person_db.py` | 지나가는 사람 사진 데이터셋(SQLite+jpg). 방문 단위 묶기, 흐린 사진 버리기, 보관 30일·SD 여유 확보 |
| `hallway.py` | 복도 데모(`--hallway`): 방문자 인사·말걸기 유도, 자기 목소리 에코 가드, 이벤트 로그 |
| `object_vision.py` | MobileNet-SSD 로 물체 검출 → 목을 그쪽으로 조준(비전 유도 파지 1단계) |
| `music.py` | 짧은 무료 음원 재생 + 박자에 맞춰 춤(머리·안테나·팔). `music/` 클립, 오프라인 |
| `custom_hands.py` | 실물 관절: 오른손 `gripper_no_wrist_roll`, 왼손 `wrist_pitch_only` |
| `base_pose.py` | 로봇 연결(`connect()`)·기본자세·stiffen/relax |
| `llm_client.py` | 브로커 HTTP 클라이언트 (stdlib만, Py3.7) |
| `state_mirror.py` | 실물 엔코더값을 ws(6171)로 방송 → 시뮬 미러 |
| `calibrate_real.py` / `snap_view.py` | 관절 방향 대조 / 카메라 프레임 저장 |
| `camera_check.py` | 카메라 초점·수평 맞추기 도구(선명도 실시간 표시, main/sub 구분) |

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

# [DGX] 로그 당겨와 분석·개선 (사람 사진 가져오기 포함)
bash ops/daily_update.sh

# [DGX] 사람 사진만 따로 가져오기 / 현황 / 학습용으로 뽑기
python3 ops/sync_persons.py                      # Pi -> 중앙 DB 로 합치기
python3 ops/sync_persons.py --purge-remote       # 옮긴 뒤 Pi 쪽 원본 삭제(SD 확보)
python3 ops/persons_export.py --list             # 어떤 게 걸러지는지 미리 보기
python3 ops/persons_export.py --faces-only --min-face 80 --out ~/ds/faces
```

**설치 스크립트**(1회): `ops/install_ollama.sh`(DGX 로컬 LLM), `ops/install_vosk.sh`(오프라인 STT), `ops/install_object_vision.sh`(물체 검출 모델). 대용량 모델은 git 제외.

Pi 의존성: `pip3 install gTTS edge-tts SpeechRecognition vosk` + `sudo apt install mpg123 python3-pyaudio flac`

## 라우팅 (한 발화가 어디로 가나)

순서대로 검사하고 처음 걸리는 곳으로 간다:

1. **STT 교정** — `config/stt_corrections.json` (양파→양팔 등 49개) + 이름 오인식. *라우팅 전*에 적용되어 동작 인식률까지 올린다
2. **에코 가드** — 로봇 자기 인사말이 마이크로 되돌아온 것 버림
3. **음악** (`music.wants_music`) — "노래 틀어줘" → 20초 클립 재생 + 춤
4. **목 제스처** (`wants_head_gesture`) — 고개/목 + 방향·동작 → 로컬 즉시 실행
5. **물체 주시** (`wants_object_look`) — "저거 봐/물건 확인해봐" → 검출 후 목 조준
6. **팔 동작** (`wants_motion`) — 신체어∧동작어 또는 제스처명사 → opus 생성/프리셋
7. **종료어** — 근사-정확 매칭만(복도 모드에선 프로그램 안 끝내고 대화만 종료)
8. **즉답 노트** — 인사/FAQ 35종
9. **비전** → 카메라 프레임 첨부, 그 외 → **대화 LLM**

회귀 테스트: `config/hallway_testset.json`(38케이스), `config/routing_testset.json`.

## 동작 파이프라인 (팔)

- Opus 출력: `{"say", "preset"|null, "moves"|null}`. moves = 키프레임(pose+duration) 또는 `{"grasp":"close"/"open"}`
- **안전층(모델 불신)**: 관절 화이트리스트, 실한계 클램프 + SAFE_LIMITS, 평균속도 상한, 팔 전체 FK 충돌검사(몸통·머리·정중선·양팔거리), 온도게이트 55°C, 뮤텍스. **실행 직전 실제 자세로 충돌 재검사**(사람이 팔을 손으로 옮겨놨을 수 있음)
- 실행: 단일 50Hz 스레드, 명령값 FK 로 손 시야추적, 키프레임마다 실측 동기화
- 상태: 시작(인사+레디포즈) → 대기(idle+팔 숨쉬기) → 생각/말하기 → 동작(손추적) → 레디복귀 → 3분후 자동이완

## 목소리 (TTS)

`edge` (마이크로소프트 신경망 음성) → `gtts` → `espeak` 순으로 내려간다.
뒤로 갈수록 기계음에 가깝다.

**systemd 로 돌리면 PATH 가 다르다.** `pip3 install --user` 로 깐 명령은
`~/.local/bin` 에 들어가는데 systemd 기본 PATH 에는 그 디렉터리가 없다. 그래서
`shutil.which('edge-tts')` 가 None 을 돌려주고, 로봇은 아무 말 없이 `gtts`(더
기계적인 목소리)로 내려가 있었다. 사람이 ssh 로 손수 실행하면 로그인 셸 PATH 에
`~/.local/bin` 이 있어 잘 되니, 더더욱 드러나지 않는다.

두 겹으로 막았다.
- `say_and_move.which_tool()` 이 PATH 에 없으면 `~/.local/bin`, `/usr/local/bin`
  까지 찾는다.
- `ops/voice_chat.service` 에 `Environment=PATH=/home/pi/.local/bin:...` 를 넣었다.

확인 방법: 기동 로그에 `TTS engine: edge` 와
`filler sounds ready (edge-koKRSunHiNeural)` 가 찍혀야 한다. `gtts` 로 찍혀 있으면
edge 를 못 찾은 것이다.

**주의: edge 도 gtts 도 인터넷이 필요하다.** 복도 무선이 끊기면 결국 espeak
(오프라인·기계음)까지 내려간다.

**오프라인 신경망 TTS 는 이 Pi 에서 못 쓴다.** piper 를 확인해 봤다 — armv7l
빌드도 있고 한국어 음성(`ko_KR-kss-medium`)도 있지만, 어느 릴리스든 바이너리가
`GLIBC_2.29~2.30` 을 요구한다. 이 Pi 는 Buster(glibc 2.28)라 실행되지 않는다
(실제로 올려서 확인: `version 'GLIBC_2.29' not found`). OS 를 올리지 않는 한
오프라인 선택지는 espeak(기계음)뿐이다.

**그래서 '늘 하는 말'을 미리 합성해 둔다.** 기동할 때 복도 인사말과 노트
즉답(총 39개)을 `~/.cache/reachy_tts` 에 만들어 두고, 말할 때 캐시가 있으면
그대로 튼다. 인터넷이 끊겨도 이 말들은 자연스러운 목소리로 나오고, 합성 대기도
없다(실측: 캐시 문구는 0.004초 만에 재생 시작).

합성 자체에는 인터넷이 필요한데 복도 와이파이는 부팅 직후에 특히 잘 끊긴다
(실제로 39개 중 7개가 그때 이름 풀이 실패로 빠졌다). 그래서 미리 합성은
백그라운드에서 돌면서 1분 간격으로 최대 10번까지 빠진 것을 다시 채운다.
필러(생각하는 소리)도 같은 방식으로 `~/.cache/reachy_fillers` 에 캐시된다.

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

**대화 중 제스처**
- 말할 때 머리만 움직이면 뻣뻣해 보여서, 응답의 45%(노트 응답은 50%)에 가벼운 팔 제스처를 곁들인다
  (`motion_exec.talk_accent`). 이미 파워가 들어간 레디 자세에서 ±14° 이내 오프셋이라 충돌 검사가 필요 없고
  발화를 지연시키지도 않는다

**노래**
- `music/` 에 20초 클립 4곡(Kevin MacLeod, CC-BY 3.0 — `music/CREDITS.txt` 에 출처 표기 필수).
  "노래 틀어줘/한 곡 들려줘/춤춰봐" → 재생하며 머리·안테나로 박자 타고 팔 제스처를 섞는다. "노래 그만" 으로 정지
- mp3 는 git 제외. 곡을 바꾸려면 `music/` 에 mp3 를 넣기만 하면 자동 인식

**머리 움직임**
- 목은 Orbita 3디스크 병렬 로봇 — 시선으로 제어하므로 팔 키프레임 파이프라인 밖. `run_head_gesture` 로 끄덕임/도리도리/상하좌우/젖히기/한바퀴를 로컬 실행
- **속도 상한 설정이 없다** → 글라이드 시간이 유일한 제어(begin_ramp 1.3s, home 1.8s, 핸드팔로우 램프 1.2s·τ0.35)

## 목을 계속 붙잡아 두기 (NeckHold)

**'팍' 하고 튀던 진짜 원인은 따로 있었다.** 목이 풀려서가 아니라, **로봇이 기억하는
시선과 실제 자세가 어긋나서**였다. 말이 끝나면 `home()` 이 `look_at` 으로 중립에
돌아갔는데, `look_at` 은 `head._soft_gaze` 를 갱신하지 않는다. IdleMotion 은 켜질 때
`_soft_gaze` 에서 이어받으므로, 어긋난 지점에서 출발하며 첫 프레임에 확 튄다.

실측(같은 상황 A/B):

| 중립 복귀 방식 | 기억된 시선 | idle 재개 후 0.1초당 최대 이동 |
|---|---|---|
| 예전: `look_at` | (0.3, 0.2) — 실제와 어긋남 | **35.74도** |
| 현재: `point_head` 글라이드 | (0.0, -0.05) — 실제와 일치 | **0.04도** |

그래서 `home()` 도 `point_head` 로 부드럽게 미끄러지게 바꿨다. 목을 움직이는
경로가 하나로 모이면 기억과 실제가 항상 같이 간다.


Orbita 목은 `connect()` 직후 **풀린 채로** 온다(실측: 디스크 셋 다 `compliant=True`).
풀린 목의 무서운 점은 처지는 것이 아니라 — 실측해 보면 이 목은 풀어도 3초간
0.00도, 제자리를 지킨다 — **명령이 조용히 무시된다**는 것이다. 실제로 목을 굳히지
않고 시선을 여섯 번 바꿔가며 사진을 찍었더니 여섯 장이 완전히 동일했다.
그 동안 쌓인 목표값은 누군가 목을 다시 굳히는 순간 한꺼번에 반영된다.
대기 중에 이따금 '팍' 하고 움직이는 정체가 이것이다.

대화 루프도 매 바퀴 `head.compliant = False` 를 다시 걸지만 한 바퀴에 한 번뿐이라,
아무도 말을 걸지 않는 복도에서는 그 간격이 십수 초까지 벌어진다. `NeckHold` 는
1초마다 확인해 그 창을 닫는다.

**되잡을 때 튀지 않게** 두 가지를 한다.
1. 굳히기 **전에** 목표값을 '지금 있는 자리'로 덮어쓴다. 풀린 상태의 쓰기는
   무시될 수 있으므로 굳힌 **뒤에** 한 번 더 쓴다.
2. 그러고 나서 마지막으로 명령했던 자세로 `neck.goto(..., minjerk)` 로 천천히
   되돌린다. 어긋남이 1.5도 미만이면 아예 건드리지 않는다.

시선 `(y, z)` 만 기억해서는 안 된다. `point_head(tilt=...)` 에 따라 같은 `(y, z)`
가 다른 목 각도가 되기 때문이다(실측: 그래서 10.4도 엉뚱한 곳으로 갔다). 그래서
`point_head` 가 **실제로 명령한 디스크 각도**를 `head._soft_thetas` 에 남긴다.

검증: 7도 어긋난 상태에서 되잡으니 0.15초당 최대 1.70도로 부드럽게 움직여
원래 자세와 0.04도 차이로 복귀했다. 어긋나지 않은 상태에서 되잡으면 0.32도
(측정 잡음 수준)만 움직인다. 되잡은 횟수는 로그로 남는다 — 정말 풀리는 일이
있는지, 있다면 얼마나 잦은지 추측하지 않고 알기 위해서다.

## 사람 데이터셋 (수집 → 중앙 DB → 학습용 추출)

복도를 지나가는 사람을 계속 찍어 모은다. 나중에 분류 학습에 쓰는 것이 목적이라,
"많이"보다 **쓸 수 있는 것**이 남도록 걸러 가며 모은다.

```
Pi: presence.py --collect-people        DGX: sync_persons.py        persons_export.py
    얼굴/움직임 감지 → person_db      →   rsync + DB 병합       →   조건 걸어 추출
    ~/reachy_logs/persons/                ~/reachy-data/persons/     images/ faces/ index.csv
```

**언제 찍나** — **SSD 가 사람을 검출했을 때만** 찍는다(신뢰도 0.5 이상).
Haar 얼굴 검출은 '찍을지'를 정하지 않고 **얼굴 상자만 얹는** 역할이다. 예전에는
Haar 가 잡으면 그대로 찍었는데, 복도 끝 유리문과 바닥 표식을 얼굴로 잡아
사람 없는 사진이 데이터셋에 들어갔다(`minNeighbors` 를 8 로 올려도 남았다).
이제 문틀은 촬영을 유발하지 못하고, Haar 를 10 까지 올려도 사진을 놓치지 않는다.

얼굴 상자가 **SSD 사람 상자 밖**이면 크롭을 남기지 않는다. 사람이 프레임에
있더라도 Haar 가 엉뚱한 곳을 잡았다면 그건 얼굴이 아니기 때문이다.

**사람 위치를 함께 저장한다** (`person_x/y/w/h`, `person_conf`). 이 복도에서는
얼굴이 잘 안 잡히므로, 나중에 분류 학습에 쓸 실질적인 기준은 이쪽이다.
`persons_export.py --crop-persons` 로 사람만 잘라 낼 수 있고, `--min-person 150`
으로 멀리 찍힌 것을 걸러낼 수 있다.

화면이 움직였다는 것만으로는 찍지 않는다. 처음에는 "옆모습·뒷모습으로 지나가는
사람"을 놓치지 않으려고 움직임만으로도 찍게 했는데, **빈 복도 사진만 쌓였다.**
자동노출 출렁임, 머리 움직임, 조명 변화가 전부 '움직임'으로 보이기 때문이다.
지금은 움직임이 보이면 사람 검출기를 곧바로 한 번 더 돌리는 **힌트**로만 쓴다.

**얼굴 검출만 믿으면 사람을 놓친다.** 크게 찍힌 정면 얼굴 사진으로 실측했더니
Haar 는 **어떤 설정으로도 못 잡았다**(폭 320~960, equalizeHist 유무, scaleFactor
1.05~1.2, 회전 ±20도 전부 0개). 역광이 세고 안경을 썼고 화면이 13도 기울어
있어서다. 같은 사진을 MobileNet-SSD 는 `person` 0.98 로 잡았다(다른 두 장은
1.00, 0.45). 모델은 이미 물체 인식용으로 올라와 있으니 그대로 나눠 쓴다.
Pi 4 에서 한 번에 ~350ms 라 1.6초 간격으로만 돌린다.

**SDK 가 카메라를 먼저 연다 (2026-09-04 에 잡은 버그).** `connect()` 는 곧바로
`/dev/video0` 을 열어 백그라운드로 스트리밍한다(`reachy/io/cam.py`). 그래서
같은 장치를 `cv2.VideoCapture` 로 또 열면 **생성에서 영영 돌아오지 않는다.**
presence 스레드가 첫 프레임에서 멎고, 얼굴 검출·복도 인사·사람 수집이 전부
아무 로그 없이 죽는다(예외가 아니라 무한 대기라 조용하다). `grab_frame` 은
이미 열려 있는 `head.camera` 를 먼저 쓰고, SDK 카메라가 있으면 장치를 절대
직접 열지 않는다. SDK 는 인덱스 0 을 고정으로 열기 때문에, 기동 시
`check_sdk_camera()` 가 `/dev/reachy-cam-main` 이 정말 video0 인지 확인하고
아니면 경고한다(흐린 쪽 카메라로 조용히 보고 있으면 안 되니까).

**카메라가 머리에 달려 있다는 함정.** 목이 돌면 화면 전체가 흐른다. 실측하면
머리 회전이 만드는 변화(0.25~0.54)가 사람이 지나갈 때(0.12~0.18)보다 **더 크다**.
즉 화면 변화량만으로는 둘을 절대 못 가른다. 그래서 로봇이 아는 사실을 쓴다 —
`say_and_move.head_still_for()` 가 목 시선이 마지막으로 바뀐 시각을 알려 주고,
목이 0.8초 이상 멎어 있을 때만 화면을 믿는다.

여기에 함정이 하나 더 있다. 사람이 지나가면 idle 모션이 **그쪽으로 고개를 돌린다**.
찍고 싶은 바로 그 순간에 목이 움직이는 것이다. 그래서 "지금 움직이는가"가 아니라
"최근 4초 안에 움직였는가"로 판단해, 고개가 멎은 직후를 노려 찍는다.

그런데 이것만으로는 부족했다. 머리가 돈 직후 멎으면, 머리 때문에 생긴 변화가
"최근 움직임"으로 남아 **빈 복도를 찍는다**(실제로 그렇게 찍혔다). 그래서
`_update_motion` 이 **목이 멎어 있는 동안 연속으로 찍힌 두 프레임만** 비교한다.
목이 움직이면 직전 프레임을 버려, 움직임을 가로질러 빼는 일이 아예 없게 했다.
덤으로 idle 이 자기 머리 움직임을 보고 두리번거리던 것도 사라진다.

**"목이 멎었나"는 명령이 아니라 실측으로 판단한다.** 처음에는 `point_head` 에
"시선이 바뀐 시각"을 남겨 썼다. 그런데 그 값이 한 번도 갱신되지 않는데도 화면은
계속 바뀌었다 — 목을 움직이는 경로가 그것 하나가 아니었던 것이다. 명령 경로를
계측하는 방식은 **한 군데만 놓쳐도 조용히 틀리고**, 틀린 줄도 모른 채 빈 복도
사진이 쌓인다. 지금은 `neck.disks[i].rot_position`(실측 각도)을 매 폴링마다 읽어
비교한다. 어느 코드가 움직였든 상관없이 카메라가 실제로 움직였는지를 본다.
읽기 비용은 0ms(캐시된 값), 정지 상태 변동폭은 최대 0.081도라 허용오차 0.3도로
충분하다.

**자동노출 스윙도 걸러야 한다.** 복도 끝이 유리문이라 역광이 세고, 카메라가
노출을 계속 조정하면 화면 전체 밝기가 출렁인다. 프레임 차이에서 **전체 밝기
변화분(중앙값)을 빼고** 국소적으로 변한 곳만 센다(실측: 밝기만 +35 바뀐 경우
넓이 0.914 → 0.066). 여기에 넓이 하한 0.03·상한 0.30 을 둔다. 사람이 지나갈 때는
0.12~0.18 이고, 화면 전체가 흔들리면 0.5 를 넘는다.

**정면을 오래 보게 하려고** `IdleMotion.ATTEND_DEADBAND` 를 둔다. 얼굴이 이미 화면
가운데(±0.15) 면 고개를 더 움직이지 않는다. 계속 미세하게 쫓아가면 시선이 떨리고,
머리에 달린 카메라가 흔들려 사진도 흐려진다. 좋은 정면 사진은 머리가 멎어 있을 때 나온다.

**품질·용량 안전장치**
- 선명도 `MIN_SHARPNESS=60` 미만은 버린다(움직이는 중에 찍힌 흐린 사진). 실제 복도 프레임은 370 안팎이라 여유가 크다.
- 방문당 6장 · 하루 800장 · 보관 30일 · SD 여유 700MB 미만이면 촬영 중단.
- **방문(visit) 단위로 묶는다.** 복도 로직이 방문을 열어 주지만, 움직임 촬영은 그 로직을 안 거치므로 `person_db` 가 스스로 방문을 연다(`auto-...`). 이게 없으면 장수 카운터가 리셋되지 않아 **6장 뒤로 수집이 영영 멈춘다**.

**사진과 DB 는 항상 같이 간다.** 사진 저장에 실패하면 행을 넣지 않는다
(`cv2.imwrite` 는 실패해도 예외를 던지지 않는다 — 실제로 18행 중 3행이 파일 없이
남아 있었다). 중앙 DB 로 옮길 때도 사진이 아직 안 온 행은 건너뛰고, 다음 번에
사진과 함께 들어온다.

**중앙 DB** — `UNIQUE(robot, src_id)` 라서 몇 번을 돌려도 중복이 안 쌓이고, 중간에
끊겨도 다시 돌리면 이어서 받는다. `--purge-remote` 로 옮긴 뒤 Pi 원본을 지워 SD 를 비운다.

**학습용 추출** — `persons_export.py` 가 선명도·얼굴크기·방문당 장수로 걸러
`images/`, `faces/`, `index.csv` 로 복사한다. 한 사람이 오래 서 있었다고 그 사람만
잔뜩 들어가지 않도록 방문당 상한(기본 4장)을 두고, 선명한 것부터 남긴다.

## 지속적 개선 루프 (로그 → 분석 → 반영)

운영 중 모든 턴이 Pi `~/reachy_logs/<세션>/events.jsonl` 에 쌓인다 (대화·동작·노트·비전·복도이벤트 + STT엔진·LLM지연·마이크RMS + 프레임).

```bash
bash ops/daily_update.sh     # ① 로그 pull → ② 다이제스트 → ③ 노트 제안 → ④ Pi 원시로그 정리
```
- **다이제스트**(`ops/log_digest.py`): 종류별 건수, 복도 통계(등장/인사/체류), 자주 나온 발화, 동작 실패 사유, STT 엔진 분포·LLM 지연, **마이크 RMS 분포**(→ `--fixed-energy` 제안값)
- **노트 제안**(`broker/build_notes.py`): 반복 발화를 *로그에 실제로 나온 답변*으로 제안 → `quick_notes.proposed.json` (할루시네이션 방지)
- **반영**: 제안 검토 → `quick_notes.json` / `stt_corrections.json` 편집 → `bash ops/deploy.sh`
- **이미지 축적**: 방문자가 나타나면 자동으로 한 장 촬영해 `frames/` 에 쌓는다(90초 간격·세션당 200장 상한,
  SD 여유 500MB 미만이면 중단). 비전 질문·동작 스냅샷도 함께 쌓여 학습·분석 소재가 된다
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
