# docs 지도

이 프로젝트는 두 축이다. **실물 축**(복도의 로봇이 사람과 상호작용)과
**시뮬 축**(MuJoCo 에서 동작을 스스로 개선). 문서도 그렇게 나뉜다.

```
docs/
├─ architecture/          실물 시스템이 어떻게 생겼나
│  ├─ overview.md         전체 그림: Pi ↔ DGX, 두 축의 관계
│  ├─ broker.md           DGX 브로커: /reply /motion /vision
│  └─ robot-runtime.md    Pi 런타임: 대화 흐름, 수면 모드, 모터 꺼짐 대응
├─ sim/                   시뮬 학습 파이프라인
│  ├─ pipeline.md         7-역할 구조와 파일 대응표
│  ├─ evaluation.md       3단계 판정, 품질 지표, opus 비평·결투, 래칫
│  ├─ improve-loop.md     개선 루프 돌리는 법, 산출물, 복구
│  ├─ world.md            MuJoCo 월드: 기구학 일치, 카메라, 시선
│  ├─ motion-presets.md   키프레임 동작 학습 분기 (프리셋 만들기)
│  ├─ sim-to-real.md      시뮬->실물 전이 로드맵, 다리 도구(sim_bridge)
│  └─ lessons.md          이 세션이 피로 배운 원칙들 (지표 감사)
├─ ops/
│  ├─ runbook.md          증상 → 볼 곳 (장애 대응)
│  ├─ processes.md        Pi/DGX 프로세스 목록 + 수동 실행 명령
│  └─ data-pipeline.md    사람 데이터셋 수집→DB→추출, 로그 개선 루프
└─ eval/                  자동 생성 리포트 (sim_record 가 쓴다, git 제외)
```

처음 읽는다면: `architecture/overview.md` → `sim/pipeline.md` →
`sim/lessons.md` 순서를 권한다.
