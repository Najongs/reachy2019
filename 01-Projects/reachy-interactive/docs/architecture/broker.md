# DGX 브로커 (broker/llm_broker.py)

한 프로세스가 역할별 언어모델을 HTTP 로 중개한다. Pi 와 시뮬 양쪽이
같은 브로커를 쓴다. 인증은 `X-Auth-Token`.

## 엔드포인트

| 경로 | 백엔드 | 용도 | 반환 |
|---|---|---|---|
| `/reply` | Ollama EXAONE 3.5 7.8B | 대화 답변 (페르소나 적용) | `{"reply": ...}` |
| `/motion` | Ollama Qwen 2.5 7B | 동작 키프레임 JSON 생성 | 모션 JSON |
| `/ground` | Codex CLI | 번호 박스 시각 접지 | 스키마 검증 JSON |
| `/evaluate` | Codex CLI | 비평·결투·오케스트라·감독 | 역할별 스키마 검증 JSON |
| `/vision` | `/ground` 호환 별칭 | 구형 클라이언트 접지 | JSON 텍스트 |
| `/health` | - | Ollama 살아있나 (워치독용) | 200/503 |
| `/stats` | - | 호출 수·실패·지연 집계 | JSON |

- **Ollama 컨텍스트**: `num_ctx=8192` 고정. 기본값 2048 이면 페르소나가
  잘려 자기를 EXAONE 이라고 한다 (실제로 겪음).
- **Codex 백엔드**: `CodexCliBackend`가 `codex exec --ephemeral`로 접지와
  평가를 각각 호출한다. `--output-schema`와 `--output-last-message`를 써서
  자유 텍스트가 제어 경로로 흘러들지 않게 한다.
- 접지는 `config/vision_prompt.txt`의 번호 박스 규약을 사용한다. 평가는
  `health`, `critique`, `duel`, `review`, `orchestra`, `director`별 스키마를 쓴다.
- 접지와 평가는 프롬프트·엔드포인트·실패 상태가 분리되어 있다. 평가 장애를
  행동 실패나 `noop`으로 기록하지 않고 `infra_error`로 중단한다.
- 클라이언트는 `robot/llm_client.py` 의 `BrokerClient`
  (`ask`/`ask_motion`/`ask_grounding`/`ask_evaluation`).

## 운영

```bash
systemctl --user status reachy-broker     # 서비스 (DGX)
bash ops/status.sh                        # 전체 요약에 포함됨
curl -s http://127.0.0.1:8080/stats | python3 -m json.tool
```

- 워치독: cron 이 `ops/dgx/broker_watchdog.sh` 를 **직접** 돌려 `/health` 503
  이면 Ollama/브로커를 살린다. 예전에는 `~/reachy-ops/` 에 실행 심을 두었지만,
  홈에 로그·상태·스크립트가 흩어지는 원인이라 2026-09-14 에 없앴다 - 운영
  로그·상태는 `04-Archives/raw/ops-logs/`, 스크립트는 저장소 한 곳이다.

### Ollama 판올림 주의 (폴더 이름이 헷갈린다)

| 경로 | 버전 | 상태 |
|---|---|---|
| `~/ollama-old/bin/ollama` | 0.3.14 | **현역** - `ollama.service` 가 이걸 실행 |
| `~/ollama/bin/ollama` | 0.33.2 | 받아만 둠 (드라이버 550+ 필요) |
| `~/ollama/models/` | - | 모델 11GB, 두 판이 공유(`OLLAMA_MODELS`) |

이 DGX 의 드라이버는 535 다. 0.33.2 로 올리면 GPU 를 못 잡아 CPU 로 떨어진다
(실측 1.5 tok/s vs 정상 83 tok/s). 그래서 CUDA 12.2 런너를 쓰는 0.3.14 를
유지한다. 드라이버를 550+ 로 올린 뒤 `ollama.service` 의 `ExecStart` 와
`LD_LIBRARY_PATH` 를 `~/ollama` 로 바꾸면 qwen3.5 계열도 받을 수 있다.
구판 설치본 백업: `04-Archives/reference/ollama-builds/`.
- 서비스 유닛에는 `--motion-backend ollama --motion-model qwen2.5:7b`,
  `--ground-backend codex --ground-prompt-file config/vision_prompt.txt`,
  `--eval-backend codex`가 있어야 한다. `codex login status`도 성공해야 한다.
- DGX의 `/usr/bin/node`는 v12라 현재 Codex CLI를 실행할 수 없다. 저장소의
  서비스 유닛처럼 nvm Node v22 경로를 PATH 맨 앞에 둔다.
- `/health`의 HTTP 상태는 워치독이 담당하는 대화 백엔드 기준이다. 응답의
  `components.grounding`과 `components.evaluation`에서 Codex 준비 상태를 따로 본다.
