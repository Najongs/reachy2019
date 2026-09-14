# 메인 지식 (Reachy)

실전·시뮬 원시 데이터를 일별 요약으로 접고, 그 요약들을 합친 종합본.
**원시 로그를 열 필요 없이 이 장과 knowledge.db 만 보면 된다.**

- 갱신: 2026-09-14T02:00:27+00:00
- 요약 구간: 2026-09-14 ~ 2026-09-14 (1일치)

## 실전 (로봇)

- 이벤트 1279건, 종류: hallway 903, chat 241, noise 70, motion 54, note 4, object_look 4
- 동작 결과: {'executed': 45, 'null_moves': 6, 'broker_failed': 2, 'failed': 1}
- 파지 라벨(사람이 말해 준 성패): 아직 없음
- 사람 데이터셋: {'people': 416, 'boxes': 487}

## 시뮬

- 학습 run 899건 접힘
- 최신 지표: {'quality': 46.0, 'success': 4} (improve-20260913-232046.json)
- 실패 원인 상위: 들기/운반 중 놓침 10788, 정렬 미달(닫힘 미발행) 7478, 맞물림 실패 3011, 경로 충돌 623, 품질 미달 323

## 검토 대기 (proposed)

- [sim/policy_candidate] best

---

구조·운영법: `01-Projects/reachy-interactive/docs/architecture/knowledge.md`
