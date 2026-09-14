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

- 워치독: cron 이 `~/reachy-ops/dgx/broker_watchdog.sh`(실행 심 - 실제 내용은
  `ops/dgx/broker_watchdog.sh`) 를 돌려 `/health` 503 이면 Ollama/브로커를
  살린다. crontab 은 긴 경로를 잘라먹은 적이 있어 심을 쓴다.
- 서비스 유닛에는 `--motion-backend ollama --motion-model qwen2.5:7b`,
  `--ground-backend codex --ground-prompt-file config/vision_prompt.txt`,
  `--eval-backend codex`가 있어야 한다. `codex login status`도 성공해야 한다.
- DGX의 `/usr/bin/node`는 v12라 현재 Codex CLI를 실행할 수 없다. 저장소의
  서비스 유닛처럼 nvm Node v22 경로를 PATH 맨 앞에 둔다.
- `/health`의 HTTP 상태는 워치독이 담당하는 대화 백엔드 기준이다. 응답의
  `components.grounding`과 `components.evaluation`에서 Codex 준비 상태를 따로 본다.
