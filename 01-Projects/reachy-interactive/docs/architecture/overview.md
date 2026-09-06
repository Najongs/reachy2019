# 전체 그림

Reachy 2019 한 대를 사무실 복도에 무인으로 켜 두고, 지나가는 사람과
대화·인사·동작하면서 데이터를 쌓는다. 동시에 DGX 시뮬레이션에서 동작을
스스로 개선해 실물에 반영한다.

## 두 축

```
[실물 축 - 복도]                      [시뮬 축 - DGX GPU0]
Pi(로봇) voice_chat.py                sim/sim_improve.py (장기 루프)
  마이크→STT→분류                       태스크 생성→서보 실행→3단계 판정
  즉답/제스처/주시 = 로컬                 opus 비평·결투→파라미터 채택
  대화/동작 = 브로커                     config/behavior_params.json ─┐
        │                                                            │
        ▼                                    개선된 행동 파라미터 ◀──┘
[DGX] broker/llm_broker.py :8080          (경로 모양·서보 이득·목 속도)
  /reply   Ollama EXAONE (무료, GPU0)
  /motion  Claude opus CLI (동작 JSON)
  /vision  Claude opus CLI (시각 접지)
```

- **Pi ↔ DGX 연결**: Pi 가 SSH 터널(`ops/pi_tunnel.service`, `-L 8080`)로
  DGX 브로커에 붙는다. 역방향은 `ssh -p 2222 pi@localhost`.
- **GPU 규칙**: GPU0 만 쓴다 (Ollama 와 MuJoCo EGL 동거,
  `MUJOCO_EGL_DEVICE_ID=0`). GPU1-7 은 다른 프로젝트 학습용.

## 설계 원칙

1. **로컬 우선**: 인터넷·LLM 없이 되는 건 전부 Pi 로컬에서
   (인사·FAQ·목 제스처·물체 주시·오프라인 STT). LLM 은 열린 대화(무료
   Ollama)와 새 동작 생성(opus)에만.
2. **검증은 결정론**: 성공/충돌 판정은 참값·기하로만 한다. 언어모델은
   품질(자연스러움)만 말한다. 모델이 성공을 판정하면 자신 있게 틀린다 -
   [sim/lessons.md](../sim/lessons.md).
3. **밤에는 침묵**: 어둡고 사람 없으면 스스로 소리·동작을 내지 않는다.
   통신은 유지. 근무시간 08-20시 (`--quiet-hours 20-8`).
4. **모터가 꺼져 있을 수 있다**: 퇴근 시 모터 전원을 끈다. 코드는 모터
   응답 없음을 정상 상황으로 취급한다.

## 저장소 구조

```
robot/    Pi 에서 도는 것 (voice_chat, motion_exec, presence, sleep_mode...)
broker/   DGX 브로커 (llm_broker.py 하나가 본체)
sim/      시뮬 축 전부 (sim_*.py) - docs/sim/pipeline.md 에 대응표
config/   페르소나·프롬프트·행동 파라미터·즉답 노트
ops/      엔트리포인트(status/deploy/daily_update) + pi/ dgx/ data/ 하위분류
docs/     이 문서들 + eval/(자동 리포트)
sim_data/ 시뮬 산출물 전부 (git 제외) - 실행 기록, 영상, gallery/, archive/
```
