# config/ 분류

경로가 Pi 배포본·서비스 유닛에 박혀 있어 **파일 위치는 옮기지 않는다.**
대신 성격을 여기 명시한다.

| 성격 | 파일 | 쓰는 쪽 |
|---|---|---|
| 손으로 쓰는 프롬프트 | persona.txt, motion_prompt.txt, vision_prompt.txt | 브로커 |
| 손으로 관리하는 데이터 | quick_notes.json, campaign_lines.json, stt_corrections.json | Pi |
| 기계가 갱신하는 상태 | behavior_params.json (시뮬 루프가 채택 파라미터 저장), sim_lessons.json, motion_candidates.json, cached_lines.json, quick_notes.proposed.json | 시뮬/브로커/ops |
| 테스트셋 | hallway_testset.json, routing_testset.json | 회귀 검사 |

- behavior_params.json 은 **시뮬 전용**이다. 실물 반영은 검토 후 수동.
- 생성 결과물(sim_results*.json 류)은 이제 `sim_data/` 에 쓴다 - config 에
  새 산출물을 만들지 말 것.
