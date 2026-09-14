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
  └─ 팔 동작 ────── HTTP /motion ─────────────────▶ Ollama + Qwen (동작 JSON)
                                                     ↑ config/persona.txt, motion_prompt.txt
       경로: Pi → (SSH 터널 -L 8080) → DGX.  ops/pi_tunnel.service
```

**설계 원칙**: 인터넷·LLM 없이도 되는 건 전부 로컬에서 (인사·FAQ·목·물체주시·오프라인 STT). 실물의 열린 대화와 새 동작 생성은 로컬 모델을 쓰고, Codex는 시뮬의 접지·평가에만 쓴다.

## 빠른 확인 (나중에 돌려볼 때 여기부터)

```bash
# [DGX] 전부 한 화면에 — 이거 하나면 대개 끝난다
bash ops/status.sh              # 요약    ●=정상 ✗=문제
bash ops/status.sh -v           # 최근 로그·워치독 기록까지

# [DGX] 로봇에게 직접 말 걸어 보기 (스피커로 나온다)
curl -s -H 'X-Auth-Token: reachy2019' -H 'Content-Type: application/json' \
     http://127.0.0.1:8080/reply -d '{"text":"너 이름이 뭐야?"}'
# -> {"reply": "저는 리치예요. ..."}   'EXAONE' 이라고 하면 페르소나가 잘린 것

# [DGX] 얼마나 쓰였고 얼마나 실패했나
curl -s http://127.0.0.1:8080/stats | python3 -m json.tool

# [DGX] 로봇에 들어가기 / 코드 배포
ssh -p 2222 pi@localhost
bash ops/deploy.sh              # 파일 보내고 md5 대조까지

# [로봇] 지금 뭐 하고 있나
tail -f ~/reachy_logs/voice_chat.log
grep -a 심장박동 ~/reachy_logs/voice_chat.log | tail -3
bash ~/Documents/brightness_report.sh    # 시간대별 화면 밝기(수면 문턱값 정할 때)
```

증상별 대응표·재시작 명령은 [docs/ops/runbook.md](docs/ops/runbook.md).
워치독이 2분(DGX)·5분(로봇)마다 스스로 되살리므로 대개 손댈 일 없다.

## 전체 구조 (무엇이 어디서 도는가)

```
        복도 (KIRO 수도권센터)                          DGX (172.16.201.166)
 ┌──────────────────────────────────┐        ┌──────────────────────────────────┐
 │  Reachy (Pi 4, Buster, armv7l)   │        │                                  │
 │                                  │        │  llm_broker :8080                │
 │  voice_chat.py  ─ 마이크→STT     │◀──────▶│    대화 → ollama/EXAONE (로컬)   │
 │    │              →라우팅→TTS    │  터널  │    동작 → ollama/Qwen            │
 │    │                             │        │    시뮬 접지·평가 → Codex CLI    │
 │    ├─ presence   사람 검출(SSD)  │        │                                  │
 │    ├─ hallway    인사·데이터수집 │        │  ollama :11434 (GPU 0)           │
 │    ├─ motion_exec 팔 동작 안전층 │        │                                  │
 │    └─ person_db  사진+메타 저장  │───────▶│  04-Archives/person-dataset/     │
 │                                  │ 하루1회│    persons/  원본 + DB           │
 │  목: NeckHold 가 계속 붙잡음     │        │    dataset/  사람 크롭 등        │
 └──────────────────────────────────┘        └──────────────────────────────────┘
   전원만 켜면 자동 실행 (systemd)             수동 실행: bash ops/daily_update.sh
```

**역할 구분이 이 시스템의 뼈대다.**

| | 로봇(Pi) | DGX |
|---|---|---|
| 듣기 | Google STT → 끊기면 vosk(오프라인) | — |
| 대화 | — | ollama + EXAONE 3.5 (무료·로컬) |
| 동작 생성 | 안전 검증·실행 | ollama/Qwen 이 키프레임 생성 |
| 사람 검출 | MobileNet-SSD (armv7l 한계) | Faster R-CNN 으로 다시 제대로 |
| 말하기 | edge-tts, 자주 쓰는 말은 미리 합성 | — |

로봇은 **실시간으로 해야 하는 것**만 한다. 무겁거나 정확해야 하는 일(대화 생성,
동작 생성, 사람 인식 DB)은 DGX 로 넘긴다. 인터넷이 끊겨도 로봇이 멈추지 않도록,
끊겼을 때 쓸 수 있는 길(오프라인 STT, 미리 합성한 목소리, 오프라인 목 제스처와
프리셋 동작)을 각각 남겨 두었다.

**Pi 에서 늘 도는 서비스** (전부 `enabled`, 전원만 켜면 시작)

| 서비스 | 하는 일 |
|---|---|
| `voice_chat` | 메인 루프. 죽으면 자동 재시작 |
| `pi_tunnel` | DGX 로 역터널 (브로커 접속 + 여기서 Pi 접속) |
| `respeaker_gain` | 마이크 게인 고정 (재부팅하면 풀린다) |
| `camera_tune` | 카메라 노출·선명도 고정 (재연결하면 풀린다) |
| `pi_viewer` | 시뮬레이터 뷰어 |

**DGX 에서 늘 도는 것**: `ollama serve`(:11434), `llm_broker`(:8080).
브로커는 터널로만 열려 있어 바깥에서 접근할 수 없다.

## 폴더

| 폴더 | 실행 위치 | 내용 |
|---|---|---|
| `robot/` | **Pi** | 로봇에서 도는 전부 → [docs/architecture/robot-runtime.md](docs/architecture/robot-runtime.md) |
| `broker/` | **DGX** | LLM 브로커 (/reply /motion /ground /evaluate) → [docs/architecture/broker.md](docs/architecture/broker.md) |
| `sim/` | **DGX GPU0** | MuJoCo 시뮬 축 전부 → [docs/sim/pipeline.md](docs/sim/pipeline.md) |
| `config/` | DGX | 페르소나·프롬프트·행동 파라미터·즉답 노트 |
| `ops/` | Pi/DGX | 엔트리포인트(status/deploy/daily_update) + `pi/` `dgx/` `data/` → [docs/ops/processes.md](docs/ops/processes.md) |
| `docs/` | - | **문서 전부** → [docs/README.md](docs/README.md) |
| `sim_data/` | - | 시뮬 산출물 (git 제외, archive/ 에 옛 런) |

운영(네트워크·부팅·서비스·배포)은 `../../02-Areas/robot-ops/README.md`.

## 더 읽기

- 전체 문서 지도: [docs/README.md](docs/README.md)
- 시뮬 개선 루프 (10시간 자율 학습): [docs/sim/improve-loop.md](docs/sim/improve-loop.md)
- 평가 체계 (3단계 판정·결투): [docs/sim/evaluation.md](docs/sim/evaluation.md)
- 피로 배운 원칙: [docs/sim/lessons.md](docs/sim/lessons.md)
