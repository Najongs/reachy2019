# 지식 창고 (Knowledge Ledger)

실전 로봇과 시뮬레이션이 쌓는 증거를 **한 곳에 모으고, 출처를 남기고,
사람 검토를 거쳐 정책에 반영**하는 체계. 코드는
`ops/data/knowledge_ledger.py`, 저장소는 `04-Archives/knowledge/`
(`knowledge.db` + `knowledge_summary.json`), 경로는 전부 `para.KNOWLEDGE`.

## 원칙

1. **집계만 저장한다** - 원문 음성·대화·얼굴·프레임은 원장에 안 들어간다.
   그것들은 각자의 보관고(conversation-logs, person-dataset)에 있고,
   원장은 참조(evidence 경로)만 가진다.
2. **출처 없는 사실은 없다** - 모든 사실에 `git_rev`(코드 판)와
   `generation`(알고리즘 세대, 예: codex-v1)이 붙는다. 알고리즘이 계속
   바뀌므로, 어떤 정책에서 나온 교훈인지 모르면 나중에 오염된다.
3. **자동 승격 금지** - 원장은 제안까지만 한다. 실제 정책(로봇 코드,
   게이트 어휘, behavior 파라미터) 반영은 사람이 결정하고, 반영되면
   `promoted` 로 표시해 이력을 남긴다.

## 사실(fact)의 수명주기

```
observed ──> proposed ──> promoted      (정책에 반영됨)
   │             └──────> rejected      (검토 후 기각)
   └ last_seen 이 오래 멎으면 사라진 사실로 간주 (은퇴 판단 근거)
```

- `observed`: 동기화가 수집한 집계 (이벤트 수, 원인 분포, 데이터셋 규모)
- `proposed`: 검토가 필요한 후보 (시뮬 정책 후보, 게이트 단어 후보)
- `promoted`/`rejected`: 사람 결정. `--promote/--reject <id> --note 사유`
- 같은 사실을 다시 보면 `last_seen` 만 갱신된다 - 살아있는 사실과
  사라진 사실이 구분된다.

## 흐름 (원천 -> 원장 -> 소비자)

```
실전: events.jsonl ─┐                        ┌─> 감독 증거 '지식 요약'
      persons.db ───┤                        │   (--digest-director,
      real_commands ┼─ --sync ─> knowledge.db ┤    사이클마다 자동 주입)
시뮬: improve-*.json┤   (매일 밤, daily_update)│
      direction-log ┘                        ├─> 게이트 단어 제안
                                             │   (--propose-gates ->
                                             │    config/gate_word_candidates.json)
                                             └─> 주간 다이제스트 (사람용)
                                                 (--weekly ->
                                                  docs/eval/knowledge-weekly.md)
```

- **동기화**: `daily_update.sh` 가 밤마다 `--sync` 를 돌린다 (로그 당김
  직후라 그날 데이터까지 포함).
- **감독 소비**: `sim_director` 가 사이클마다 `digest_for_director()` 를
  증거에 넣는다 - 실전 수요 상위, 실전 실패 분포, 시뮬 실패원인 누적,
  승격 대기 수. LLM 호출이 없어 offline 강등 모드에서도 안전하다.
- **게이트 단어 제안**: 실전에서 동작으로 라우팅되지 못한 명령
  (null_moves/rejected)의 어휘 중 `voice_chat.py` 게이트 어휘(AST 파싱)에
  없는 것을 후보로 발굴한다. '어깨' 누락 사고(2026-09-08)의 자동화판.
  후보엔 단어만 저장한다 (원문 문장 미저장 원칙 유지).
- **실전 파지 라벨**: 파지류 동작 후 90초 안의 "잡았어/놓쳤어" 발화가
  `grasp_label` 이벤트로 기록되고 다음 동기화 때 원장으로 흐른다 -
  실전 파지 성패 데이터의 시작점.

## 운영 명령 모음

```bash
python3 ops/data/knowledge_ledger.py --sync             # 수집 (daily_update 가 자동)
python3 ops/data/knowledge_ledger.py --stats            # 최근 요약 보기
python3 ops/data/knowledge_ledger.py --digest-director  # 감독 주입용 요약
python3 ops/data/knowledge_ledger.py --propose-gates    # 게이트 어휘 후보 발굴
python3 ops/data/knowledge_ledger.py --weekly           # 사람 검토용 다이제스트
python3 ops/data/knowledge_ledger.py --promote 123 --note "실기 검증 완료"
python3 ops/data/knowledge_ledger.py --reject 124 --note "오탐"
```

## 지킴이

- `tests/test_repo_hygiene.py`: 가짜 중첩 보관고(01-Projects/**/04-Archives)
  가 생기면 실패 - 상대 깊이 셈 사고(2026-09-08, 09-14 두 번)의 감시자.
  원장이 para.py 를 쓰는지도 검사한다.
- `ops/dgx/collect_health.py`: 실전 수집 연속성 - 낮 12시가 지나도 오늘
  세션이 0건이면 경고 (09-12~13 공백처럼 조용히 새는 것 방지).
- 평가기 폴백 사다리(브로커 FallbackBackend: codex -> opus -> 감독
  offline): 평가기 장애가 지식 생산(밤 학습)을 통째로 멈추지 않게 한다.
