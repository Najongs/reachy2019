# DGX 브로커 (broker/llm_broker.py)

한 프로세스가 세 종류의 언어모델을 HTTP 로 중개한다. Pi 와 시뮬 양쪽이
같은 브로커를 쓴다. 인증은 `X-Auth-Token`.

## 엔드포인트

| 경로 | 백엔드 | 용도 | 반환 |
|---|---|---|---|
| `/reply` | Ollama EXAONE 3.5 7.8B | 대화 답변 (페르소나 적용) | `{"reply": ...}` |
| `/motion` | Claude opus CLI (영속 세션) | 동작 키프레임 JSON 생성 | 모션 JSON |
| `/vision` | Claude opus CLI (전용 세션) | 이미지 접지·비평 (원문 그대로) | 텍스트 |
| `/health` | - | Ollama 살아있나 (워치독용) | 200/503 |
| `/stats` | - | 호출 수·실패·지연 집계 | JSON |

- **Ollama 컨텍스트**: `num_ctx=8192` 고정. 기본값 2048 이면 페르소나가
  잘려 자기를 EXAONE 이라고 한다 (실제로 겪음).
- **CLI 백엔드**: `ClaudeCliBackend` 가 세션별 영속 프로세스를 유지한다.
  vision 은 motion 과 별도 세션 + 별도 시스템 프롬프트
  (`config/vision_prompt.txt` - 번호 박스 접지, JSON 한 줄 규약).
- 클라이언트는 `robot/llm_client.py` 의 `BrokerClient`
  (`ask`/`ask_motion`/`ask_vision`).

## 운영

```bash
systemctl --user status reachy-broker     # 서비스 (DGX)
bash ops/status.sh                        # 전체 요약에 포함됨
curl -s http://127.0.0.1:8080/stats | python3 -m json.tool
```

- 워치독: cron 이 `~/reachy-ops/broker_watchdog.sh`(실행 심 - 실제 내용은
  `ops/broker_watchdog.sh`) 를 돌려 `/health` 503 이면 Ollama/브로커를
  살린다. crontab 은 긴 경로를 잘라먹은 적이 있어 심을 쓴다.
- 서비스 유닛 인자에 `--vision-model opus --vision-prompt-file
  config/vision_prompt.txt` 가 있어야 /vision 이 뜬다.
