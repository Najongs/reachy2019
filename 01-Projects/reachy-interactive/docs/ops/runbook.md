# 장애 대응 (증상 → 볼 곳)

```bash
bash ops/status.sh        # [DGX] 전부 한 화면 - 대개 이거면 끝
bash ops/status.sh -v     # 최근 로그·워치독 기록까지
```

| 증상 | 먼저 볼 것 |
|---|---|
| 대답을 아예 안 한다 | `ops/status.sh` → `voice_chat` 과 `pi_tunnel` 이 ● 인가 |
| "음, 잘 모르겠어요" 만 한다 | `/health` 503 → Ollama. 워치독이 2분 안에 살린다 |
| 자기를 EXAONE 이라고 한다 | 페르소나 잘림 → 브로커 `--ollama-ctx` (8192 이상) |
| 움직이지 않는다 | 모터 전원 (퇴근 때 끈다). 로그 `움직임 없이 소리만 내는 모드` |
| 밤인데 소리를 낸다 | `--quiet-hours` / `--dark-below` (voice_chat.service) |
| 낮인데 조용하다 | 심장박동이 `자는 중` 인가 → 밝기 문턱 `ops/brightness_report.sh` |
| 왼쪽 그리퍼 오류 | 오류 아님 - 모터가 원래 없다 |
| 시뮬 루프가 이상하다 | [../sim/improve-loop.md](../sim/improve-loop.md) '망가졌을 때' |

## 자주 쓰는 명령

```bash
# 로봇에게 직접 말 걸기 (스피커로 나온다)
curl -s -H 'X-Auth-Token: reachy2019' -H 'Content-Type: application/json' \
     http://127.0.0.1:8080/reply -d '{"text":"너 이름이 뭐야?"}'

curl -s http://127.0.0.1:8080/stats | python3 -m json.tool   # 사용량/실패
ssh -p 2222 pi@localhost                                     # Pi 접속
bash ops/deploy.sh                                           # 배포(md5 대조)
systemctl --user restart reachy-broker                       # 브로커 재시작
tail -f ~/reachy_logs/voice_chat.log                         # [Pi] 라이브 로그
```
