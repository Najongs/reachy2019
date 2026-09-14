# 지식 창고 (3층 구조)

실전 로봇과 시뮬레이션이 쌓는 데이터를 **원시 → 주기 요약 → 메인 지식**
세 층으로 접는다. 읽는 쪽은 보통 맨 위 한 장만 본다 — **매번 모든 로그를
열지 않는 것이 이 구조의 목적**이다.

코드는 `ops/data/knowledge_ledger.py`, 위치는 전부 `para.py` 에서 얻는다.

```
04-Archives/
├─ raw/                        1층 · 원시 축적 (한 곳에 모음)
│  ├─ conversation-logs/pi/    로봇 대화·이벤트 (events.jsonl, 세션별)
│  ├─ person-dataset/          사람 사진 + persons.db
│  ├─ sim-runs/                시뮬 런 산출물 (세대별: legacy/codex-v1…)
│  └─ ops-logs/                브로커·워치독·ollama 로그 (~/reachy-ops 는 링크)
├─ knowledge/
│  ├─ digests/YYYY-MM-DD/      2층 · 일별 요약 (real.json, sim.json)
│  ├─ knowledge.db             3층 · 사실 원장 (종합값 + 커서 + 요약 색인)
│  ├─ KNOWLEDGE.md             3층 · 사람과 모델이 읽는 한 장
│  └─ knowledge_summary.json   마지막 동기화 상태 (델타·남은 원시)
└─ reference/                  참고 데모 소스 (flyers·tictactoe·readcam)
```

## 층별 역할

**1층 원시** — 로그·사진·런이 그대로 쌓인다. 지우지 않는다. 쓰는 쪽은
각 수집기(voice_chat, sync_persons, sim_director)이고, 읽는 쪽은 아래
동기화 하나뿐이다.

**2층 주기 요약(지식화)** — `--sync` 가 **새로 생긴 부분만** 읽어 그날의
요약 한 장으로 접는다. 같은 날 여러 번 돌면 합산된다. 이 층이 생긴 뒤로
메인 지식은 원시를 직접 열지 않는다.

**3층 메인 지식** — 요약 장들을 합쳐(rollup) 사실 원장을 갱신하고,
`KNOWLEDGE.md` 한 장을 다시 쓴다. 감독·사람·다른 코드는 여기만 본다.

## 매번 전체를 안 읽는 장치: 커서

`cursors` 테이블이 파일별로 `(크기, 오프셋)` 을 기억한다. 로그는
append-only 라 지난 오프셋부터 읽으면 같은 줄을 다시 파싱하지 않는다.
파일이 줄었으면(truncate·교체) 처음부터 다시 읽는다.

실측 (원시 1,279 이벤트 + 899 런):
- 전체 재구축 `--full`: 0.4초
- 이후 증분 `--sync`: **0.3초, 새 이벤트 0건 · 런 899건 전량 건너뜀**

주 1회(일요일) 자동으로 `--full` 이 돌아 커서가 놓친 것을 바로잡는다.

## 사실(fact)의 수명주기

```
observed ──> proposed ──> promoted      (정책에 반영됨)
   │             └──────> rejected      (검토 후 기각)
   └ last_seen 이 멎으면 사라진 사실 (은퇴 판단 근거)
```

- 메인 원장에는 **종합값만** 둔다(`upsert_fact`). 사이클마다 행이 쌓여
  원장이 로그가 되는 것을 막는다 — 실제로 `policy_candidate` 가 899행까지
  불어 있었고, 지금은 구간 최고 후보 1건으로 접힌다.
- `--promote/--reject <id> --note 사유` 로 사람이 결정하고 이력을 남긴다.

## 무엇이 지식인가 (계측만이 아니다)

원장은 네 축을 모은다 — 어느 하나만 있으면 "왜 그렇게 됐는지"를 못 잇는다.

| 축 | 사실 종류 | 어디서 | 무엇을 남기나 |
|---|---|---|---|
| **계측** | `event_count` `outcome_count` `failure_cause` `latest_metrics` `policy_candidate` `grasp_label` `person_dataset` | 실전 이벤트, 시뮬 run | 무슨 일이 얼마나 일어났나 |
| **말·규칙** | `dialog_rule` `dialog_candidate` `stt_correction` | quick_notes(승격된 즉답), 제안 후보, STT 교정 | 대화에서 배워 규칙이 된 것 |
| **프롬프트·페르소나** | `prompt_asset`(현재 판) `prompt_version`(이력) | `config/*.txt`, 코드 내 `*_PROMPT` 상수 | 로봇·평가자가 무엇을 말하도록 되어 있나, 언제 바뀌었나 |
| **판단** | `feedback_note` `direction_decision` | 지적 원장, direction-log | 사람·감독이 무엇을 고치라 했나 |

프롬프트는 **전문이 아니라 지문(sha)·크기·규칙 수**만 남긴다. 원문은 git과
`config/` 에 있고, 원장의 역할은 *"어느 판으로 돌 때 이 성과가 나왔나"* 를
잇는 것이다. 그래서 `prompt_asset` 은 현재 판(upsert), `prompt_version` 은
판이 바뀔 때마다 한 줄씩 쌓이는 이력이다. 감독 증거에도 프롬프트 지문이
들어가므로, 성과 변화를 파라미터뿐 아니라 프롬프트 개정에도 귀속시킬 수
있다.

대화 원문은 여전히 원장에 들어가지 않는다 — 담기는 것은 이미 규칙으로
승격된 패턴→답변과, 사람 검토를 기다리는 후보뿐이다.

## 원칙

1. **집계만 저장** — 원문 음성·대화·얼굴·프레임은 원장에 안 들어간다.
   원시 보관고에 있고, 원장은 참조 경로만 가진다.
2. **출처 없는 사실은 없다** — 모든 사실에 `git_rev` 와 `generation`
   (알고리즘 세대, 예: `codex-v1`)이 붙는다. 세대가 바뀌면 옛 교훈이
   오염되므로 어느 판에서 나온 사실인지 항상 남는다.
3. **자동 승격 금지** — 원장은 제안까지. 정책 반영은 사람이 한다.

## 소비자 (지식이 흐르는 곳)

| 소비자 | 명령 | 무엇을 받나 |
|---|---|---|
| 감독 루프 | `--digest-director` | 실전 수요·실패 분포, 시뮬 원인 누적, 파지 라벨, 승격 대기 수 — 사이클마다 증거에 주입 (원시 미접근·LLM 무호출이라 offline 안전) |
| 게이트 어휘 | `--propose-gates` | 동작으로 라우팅 못 된 명령의 어휘 후보 → `config/gate_word_candidates.json` ('어깨' 누락 사고의 자동화판) |
| 사람 검토 | `--weekly` | `docs/eval/knowledge-weekly.md` 주간 다이제스트 |
| 누구나 | (파일) | `04-Archives/knowledge/KNOWLEDGE.md` 한 장 |

실전 파지 라벨은 로봇에서 시작된다: 파지 동작 후 90초 안의
"잡았어/놓쳤어" 발화가 `grasp_label` 이벤트가 되고, 다음 동기화에
요약 → 메인 지식으로 흐른다.

## 운영 명령

```bash
python3 ops/data/knowledge_ledger.py --sync            # 증분 (daily_update 자동)
python3 ops/data/knowledge_ledger.py --sync --full     # 전체 재구축 (주 1회 자동)
python3 ops/data/knowledge_ledger.py --stats           # 마지막 상태
python3 ops/data/knowledge_ledger.py --digest-director # 감독 주입용 요약
python3 ops/data/knowledge_ledger.py --propose-gates   # 게이트 어휘 후보
python3 ops/data/knowledge_ledger.py --weekly          # 주간 다이제스트
python3 ops/data/knowledge_ledger.py --promote 12 --note "실기 검증 완료"
```

## 지킴이

- `tests/test_repo_hygiene.py` — 가짜 중첩 보관고(01-Projects 안 04-Archives)
  감지. 상대 깊이 셈 사고가 두 번(09-08, 09-14) 났다. 원장이 para 를
  쓰는지도 검사한다.
- `ops/dgx/collect_health.py` — 낮 12시가 지나도 오늘 세션이 0건이면 경고.
- 평가기 폴백 사다리(브로커 `FallbackBackend`: codex → opus, 감독은
  offline 강등) — 평가기 장애가 지식 생산(밤 학습)을 멈추지 않게 한다.
