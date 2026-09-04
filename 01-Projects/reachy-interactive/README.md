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

## 사람 데이터셋 (수집 → 중앙 DB → 학습용 추출)

복도를 지나가는 사람을 계속 찍어 모은다. 나중에 분류 학습에 쓰는 것이 목적이라,
"많이"보다 **쓸 수 있는 것**이 남도록 걸러 가며 모은다.

```
Pi: presence.py --collect-people        DGX: sync_persons.py        persons_export.py
    얼굴/움직임 감지 → person_db      →   rsync + DB 병합       →   조건 걸어 추출
    ~/reachy_logs/persons/                ~/reachy-data/persons/     images/ faces/ index.csv
```

**언제 찍나** — 세 경로가 있다.
1. **얼굴이 보일 때** (Haar 정면 검출). 4초 간격, 방문당 6장까지. 얼굴 크롭도 남는다.
2. **사람 몸이 보일 때** (MobileNet-SSD `person`). 이 복도에서는 이쪽이 주력이다.
3. **얼굴도 몸도 못 잡았지만 화면이 움직일 때** — 마지막 그물. 12초 간격.

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
