# Pi 런타임 (robot/)

본체는 `voice_chat.py` 하나. 마이크 입력을 분류해 가장 싼 경로로 응답한다.

## 응답 경로 (싼 것부터)

1. **즉답 노트** (`quick_notes.py` + `config/quick_notes.json`) - 인사,
   FAQ, 회사 캠페인 멘트("먼저 건넨 한마디..."), 포스터 QR 행사 권유.
   LLM 없이 0초.
2. **목 제스처** (`say_and_move.py`) - 말하며 고개 동작. 로컬.
   목 속도 상한 있음 (실물 커밋: 목이 빨리 돌지 않게).
3. **물체 주시** (`object_vision.py`) - 로컬 DNN. 사람 상자 기반 조준,
   사람을 놓치면 실측 정면으로 복귀.
4. **대화** - 브로커 `/reply`. 스트리밍 + 문장 단위 TTS.
5. **팔 동작** - 브로커 `/motion` → `motion_exec.py` 가 검증 후 실행.

## 수면 모드 (sleep_mode.py)

- **조건**: 어둡고(화면 밝기 문턱, `ops/pi/brightness_report.sh` 로 보정)
  + 사람 없음(presence) → 스스로 내는 소리·동작 전부 중지.
- **통신은 켜 둔다**: SSH 터널, 브로커 연결, 로그, 심장박동 유지.
- **근무시간**: 08-20시 (`--quiet-hours 20-8`).
- **모터 전원**: 퇴근 때 사람이 끈다. 모터 무응답 = 오류 아님. 로그에
  `움직임 없이 소리만 내는 모드` 로 표시. 아침에 켜면 자동 복귀.
  주의: 2026-09-08 왼손 모듈 교체로 왼쪽 그리퍼 장착됨 (dxl_27, 실행 한계 [-10,50], 양수=벌림 - 오른손의 거울). wrist_roll 은 양쪽 다 여전히 없음.
  **오른손은 그리퍼로 변환 완료 - 잡기 사용 가능 (2026-09-06 확인).**

## 로그와 상태

```bash
tail -f ~/reachy_logs/voice_chat.log       # [Pi] 지금 뭐 하나
grep -a 심장박동 ~/reachy_logs/voice_chat.log | tail -3
bash ops/status.sh                          # [DGX] 전체 요약
```

---

# 상세 (README 에서 이동)

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
| `calibrate_real.py` | 관절 방향 대조 |
| `snap_view.py` | 카메라 프레임 한 장 저장 (조준 확인용) |
| `camera_check.py` | 카메라 초점·수평 맞추기 도구(선명도 실시간 표시, main/sub 구분) |

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

미리 합성할 목록은 두 곳에서 온다.
- 복도 인사말과 노트 즉답 (코드/설정에 적힌 것)
- **로그에서 실제로 되풀이된 답변** — `ops/build_tts_cache_list.py` 가 events.jsonl
  에서 2번 이상 나온 답변을 뽑아 `config/quick_notes.json` 옆의
  `cached_lines.json` 에 적는다. 한 번만 나온 문장은 넣지 않는다(LLM 이 그때그때
  만든 말은 다시 나오지 않아 자리만 차지한다). 시각·온도처럼 값이 바뀌는 문장도
  거른다. `daily_update.sh` 가 매번 갱신하고, `deploy.sh` 가 Pi 로 보낸다.

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

## 밤에는 잔다 (수면 모드)

빈 사무실에서 로봇이 혼자 인사하고 팔을 움직이면 무섭다. 그래서 **불이 꺼지고
사람도 없으면 스스로는 아무 소리도 내지 않고 움직이지도 않는다**.

**통신은 끄지 않는다.** 터널·브로커·ssh·뷰어·상태 미러·로그·카메라 감시는 자는
동안에도 그대로 돈다. 멈추는 것은 로봇이 '스스로 시작하는' 소리와 동작뿐이라,
자는 밤에도 원격으로 들여다보고 손볼 수 있다.

```
동작한다  근무시간 아침 8시 ~ 저녁 8시
잠든다    그 밖이거나 어두운데(밝기 < 0.15) 사람이 3분간 없으면
깬다      불이 켜짐(근무시간 밖에는 제외) · 사람이 보임 · 말을 걺(잡음은 무시)
```

밝기는 `presence.PresenceWatcher` 가 이미 만들어 둔 축소 흑백 프레임의 평균으로
잰다 — 카메라를 따로 열지 않는다(카메라는 한 번에 하나만 열린다). 5분마다 로그에
남으므로 `ops/pi/brightness_report.sh` 로 시간대별 값을 보고 문턱을 맞춘다.
**실측: 불 켜진 복도 0.577.**

시간대 규칙이 따로 있는 이유는 **퇴근할 때 모터 전원을 내리면 카메라도 없어지기**
때문이다(SDK 카메라는 `parts.Head` 생성에 딸려 온다). 밝기만 믿는 설계는 정작
재워야 할 밤에 눈이 없다.

## 모터가 꺼져 있어도 죽지 않는다

퇴근 전 모터 전원을 내리면 Luos 게이트(USB)는 살아 있어도 다이나믹셀이 응답하지
않아 `base_pose.connect()` 가 `LuosModuleNotFoundError` 로 실패한다. 예전에는 그
예외가 `main()` 을 뚫고 나가 서비스가 12초마다 재시작을 되풀이했다.

지금은 **소리만 내는 모드**로 내려간다: 듣고, 브로커에 묻고, 말한다. 움직이지만
않는다. 말하기 경로도 `say_and_move.head_ready()` 로 죽은 버스를 감지해 넘어가므로,
밤에 누가 말을 걸어도 대화 루프가 죽지 않는다(경고는 한 번만 남는다).

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

## 조정 손잡이

| 항목 | 위치 |
|---|---|
| 마이크 감도 | `--fixed-energy`(현재 550). 튜닝은 `--stt-test` 로 captured RMS 보며 |
| 마이크 게인 | `ops/pi/respeaker_gain.py --max-gain-db --desired-level` (부팅 서비스에 반영) |
| 대화 모델·길이 | 브로커 `--ollama-model`, `OllamaBackend(num_predict)`, `spoken_trim()` |
| 성격·말투 | `config/persona.txt` (수정 후 `systemctl --user restart reachy-broker`) |
| 인사 빈도 | `robot/hallway.py` greet_cooldown / gesture_cooldown / absence_reset |
| 시선 기본 높이 | `say_and_move.GAZE_TILT`(-0.15) 또는 `--gaze-tilt` |
| 고개 추적 방향 | `IdleMotion.ATTEND_SIGN` (반대로 돌면 -1.0) |
| 안전 가동범위 | `motion_exec.SAFE_LIMITS` (shoulder_roll ±80 등) |
| 레디자세·유지토크·자동이완 | `motion_exec` READY_POSE / HOLD_TORQUE / SETTLE_AFTER |
| 집기 좌표 | `motion_presets.pick_to_tray` + `config/motion_prompt.txt` 환경절 |
| 시뮬 관절 축/부호 | `sim/sim_viewer.html` J 테이블 |
