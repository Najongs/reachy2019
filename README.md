# Reachy 2019 — Interactive Voice · Vision · Motion

Pollen Robotics Reachy 2019 로봇을 **음성으로 대화하고, 카메라로 보고, 자연어 명령으로 팔·머리를 움직이는** 상호작용 시스템으로 확장한 작업 저장소.

Claude(대화 haiku / 동작 생성 opus)를 브로커로 두고, 라즈베리파이 로봇이 마이크→STT→LLM→TTS+모션 파이프라인을 돌린다. 실물과 동기화되는 3D 브라우저 시뮬레이터도 포함.

## PARA 구조

이 저장소는 [PARA](https://fortelabs.com/blog/para/) 방식으로 정리되어 있다.

| 폴더 | 내용 |
|---|---|
| **01-Projects/** | 진행 중인 작업 결과물. `reachy-interactive/` = 음성·비전·동작 시스템 전체 |
| **02-Areas/** | 지속 운영 영역. `robot-ops/` = 배포·네트워크·부팅 등 운영 문서 |
| **03-Resources/** | 참고 자료. Reachy 2019 원본 SDK, Luos 펌웨어, 3D 모델, 로봇 사진 |
| **04-Archives/** | 비활성. 참고용 데모 소스(flyers·tictactoe), 대화 로그 |

## 빠른 시작

전체 아키텍처·실행법은 **[01-Projects/reachy-interactive/README.md](01-Projects/reachy-interactive/README.md)** 참조.

```
[Pi: 로봇]  voice_chat.py ── STT ── HTTP ──▶ [DGX] llm_broker.py ── Claude
                 └ 팔·머리 모션 + 3D 미러(브라우저 시뮬)
```

## 원본에 대하여

`03-Resources/reachy-2019-sdk/` 는 [pollen-robotics/reachy-2019](https://github.com/pollen-robotics/reachy-2019)
원본 SDK(Apache-2.0)이며 수정하지 않았다. 이 저장소의 창작물은 `01-Projects/`·`02-Areas/` 에 있다.

## 제외된 것 (.gitignore)

대용량 3D 모델(`*.glb`), 대화·동작 로그, 로봇 사진, 비밀 파일(API 키)은 저장소에서 제외.
3D 모델 재생성은 `03-Resources/3d-models/README.md` 참조.
