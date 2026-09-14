# 메인 지식 (Reachy)

실전·시뮬 원시 데이터를 일별 요약으로 접고, 그 요약들을 합친 종합본.
**원시 로그를 열 필요 없이 이 장과 knowledge.db 만 보면 된다.**

- 갱신: 2026-09-14T02:17:28+00:00
- 요약 구간: 2026-09-14 ~ 2026-09-14 (1일치)

## 실전 (로봇)

- 이벤트 1285건, 종류: hallway 908, chat 242, noise 70, motion 54, note 4, object_look 4
- 동작 결과: {'executed': 45, 'null_moves': 6, 'broker_failed': 2, 'failed': 1}
- 파지 라벨(사람이 말해 준 성패): 아직 없음
- 사람 데이터셋: {'people': 417, 'boxes': 487}

## 시뮬

- 학습 run 924건 접힘
- 최신 지표: {'quality': 46.0, 'success': 4} (improve-20260913-232046.json)
- 실패 원인 상위: 들기/운반 중 놓침 10825, 정렬 미달(닫힘 미발행) 7489, 맞물림 실패 3032, 경로 충돌 623, 품질 미달 553

## 말과 규칙 (로봇이 무엇을 말하도록 되어 있나)

- 페르소나: d50bcef85ee9 (4146자, 규칙 40줄)
- 프롬프트 판본 8종: CRITIQUE_PROMPT=813b8014e4e2, DIRECTOR_PROMPT=7fd0eaee97b1, DUEL_PROMPT=a4f6699c0dbc, ORCHESTRA_PROMPT=4f66de6f4d34, PILOT_PROMPT=bfbf0d244603, motion_prompt.txt=9cabe50bee04, persona.txt=d50bcef85ee9, vision_prompt.txt=a68256e6674c
- 즉답 규칙(대화에서 승격): 35건, 검토 대기 후보 8건
- STT 오인식 교정: {'word_fixes': 49, 'name_mishears': 26, 'persona_hints': 14}
- 동작 지적 원장: {'total': 423, 'open': 8}
- 기록된 방향 결정: 29건

## 검토 대기 (proposed)

- [real/dialog_candidate] 하얀
- [real/dialog_candidate] A
- [real/dialog_candidate] 이제 안녕
- [real/dialog_candidate] 다
- [real/dialog_candidate] 이다
- [real/dialog_candidate] 호소 야
- [real/dialog_candidate] 야 고사양
- [real/dialog_candidate] 몇 시야

---

구조·운영법: `01-Projects/reachy-interactive/docs/architecture/knowledge.md`
