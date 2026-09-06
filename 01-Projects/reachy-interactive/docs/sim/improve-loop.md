# 개선 루프 (sim_improve.py)

강화학습 흉내의 파라미터 탐색. 정책은 신경망이 아니라
`config/behavior_params.json` 의 손잡이 10개다.

| 파라미터 | 뜻 |
|---|---|
| step_deg / gain / damping | 서보 스텝 크기·이득·감쇠 |
| table_min | 테이블 여유 하한 (cm) |
| lift_steps | 들어올리기 보간 수 |
| via1/2/3_{pitch,roll,yaw,elbow} | 들어올리기 경유 자세 3개 = 경로 **구조** (12개) |
| gaze_step_deg | 목 회전 한 스텝 각도 (시선 활강 속도) |

옛 판의 ready_* 설정은 load_behavior 가 경유 자세로 자동 이관한다.

## 돌리는 법

```bash
# 짧은 예행 (2-3 반복)
python3 sim/sim_improve.py --iters 3 --token reachy2019 --seed 21

# 장기 (10시간) - 반드시 setsid 로 분리
nohup setsid python3 sim/sim_improve.py --hours 10 --token reachy2019 \
    --seed 31 > sim_data/improve_10h.log 2>&1 < /dev/null &

# 중지 (자기 셸을 죽이지 않으려면 반드시 브래킷 패턴)
pgrep -f "sim_improve[.]py" | while read p; do kill $p; done
```

## 한 반복

1. 후보 생성: **CEM** - 엘리트(상위 3)의 평균·분산으로 다음 세대 분포를
   만든다 (정체 시 폭↑ 담금질). opus 조언 후보도 합류
2. 각 후보를 **훈련셋** 4개로 평가 → 단계 계수 + 거리 + 품질
3. 선택: `(충돌 적게, 성공 많이, 목표에 가깝게, 품질)` 사전식.
   충돌이 기준보다 늘면 즉시 탈락 (안전 후퇴 금지)
4. 품질 박빙이면 opus 결투가 심판 (성분 상세는 보정용으로 기록)
5. 채택 → **고정 시험지** 4개로 채점. 기록·래칫·best_ever·탈출구 판단은
   전부 시험지 기준 - 태스크 교체에 추이가 흔들리지 않는다
6. `config/behavior_params.json` 저장 (시뮬 전용 - 실물 반영은 검토 후 수동)

주기 작업: 훈련셋 교체 6회마다(시험지는 불변), opus 비평 3회마다,
end-to-end 검증 60회마다, 래칫은 시험지 전원 성공 2연속마다,
발산 감사(품질↑·성공↓) 15회마다, 시험 성공 0 이 10회면 best_ever 복귀.

## 산출물

```
sim_data/improve-<시각>.json        체크포인트 (history + best_ever)
sim_data/improve-<시각>-media/      과정 영상
  iterNNNN.mp4                      비평 반복의 에피소드 (눈|제3자)
  iterNNNN_strip.jpg                opus 가 본 필름 스트립
  iterNNNN_duel.jpg                 결투 A/B 비교
sim_data/<시각>-improve-checkN/     60회마다 실전 배치 (mp4 포함)
```

## 망가졌을 때

- **파라미터 복원**: 체크포인트의 `best_ever.params` 를
  `config/behavior_params.json` 에 덮어쓰면 된다 (전원 성공 + 최고 품질
  시점만 기록되므로 안전).
- **가중치 보정**: `python3 sim/sim_calibrate.py` - 쌓인 결투 기록으로
  사람 눈의 가중치를 회귀해 계측 배점과 비교(제안만, 자동 적용 없음).
- **태스크 실현가능성은 기준 파라미터(`sim_agent.DEFAULTS`)로 판정**한다.
  현재 파라미터로 판정하게 바꾸면 안 된다 - 파라미터가 나빠질수록 가능한
  태스크가 사라지는 순환에 빠진다 (lessons.md 의 공회전 사고).
