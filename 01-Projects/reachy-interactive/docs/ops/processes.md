# 프로세스 목록

## Pi (로봇)

| 이름 | 방식 | 역할 |
|---|---|---|
| voice_chat.service | systemd --user | 본체 (STT→분류→응답, 수면 모드 포함) |
| pi_tunnel.service | systemd --user | SSH 터널 -L 8080 (DGX 브로커로) |
| pi_viewer.service | systemd --user | 화면 뷰어 |
| camera_tune.service | systemd | 카메라 파라미터 |
| respeaker_gain.service | systemd | 마이크 게인 |
| pi_watchdog.sh | cron | 서비스 감시·재시작 |

Pi 는 armv7l / Raspbian Buster - 파이썬·패키지 오래됨을 전제.

## DGX

| 이름 | 방식 | 역할 |
|---|---|---|
| reachy-broker | systemd --user | 브로커 :8080 (/reply /motion /vision) |
| ollama | 시스템 | EXAONE 3.5 (GPU0) |
| broker_watchdog | cron (~/reachy-ops/ 심 → ops/dgx/broker_watchdog.sh) | /health 감시 |
| sim_director.py | **cron 야간 전용** (20:10~07:55 KST) | 시뮬 학습 감독 - 실물과 같은 opus CLI 를 쓰므로 로봇 수면시간에만 돈다 |
| daily_update.sh | cron | 로그 수집·리포트 |

## 규칙

- GPU0 만 사용 (Ollama + MuJoCo EGL). GPU1-7 은 다른 프로젝트.
- 장기 루프는 반드시 `setsid` 분리, 로그는 `sim_data/improve_10h.log`.
- 중지는 `pgrep -f "sim_improve[.]py"` 브래킷 패턴으로만
  ([../sim/lessons.md](../sim/lessons.md) 8번).

---

# 상세 (README 에서 이동)

## 프로세스 관리 (두 대를 어떻게 살려 두는가)

```
bash ops/status.sh        # DGX + 로봇 전체 상태 한 화면
bash ops/status.sh -v     # 최근 로그·워치독 기록까지
```

| | DGX | 로봇(Pi) |
|---|---|---|
| 서비스 | `ollama.service`, `reachy-broker.service` (`--user` + linger) | `voice_chat`, `pi_tunnel`, `pi_viewer`, `respeaker_gain` |
| 죽으면 | systemd `Restart=always` | systemd `Restart=always` |
| 멎으면 | `ops/dgx/broker_watchdog.sh` (cron 2분) | `ops/pi_watchdog.sh` (cron 5분) |
| 로그 회전 | 워치독이 스스로 자름 | `ops/logrotate-reachy` → `/etc/logrotate.d/reachy` |

**'죽는 것'과 '멎는 것'은 다르다.** systemd 는 프로세스가 사라져야 반응한다.
실제로 겪은 고장은 대부분 떠 있는데 멎은 쪽이었다 — 그래서 감시가 따로 있다.

**`/health` 는 진짜로 확인한다.** 예전에는 브로커가 떠 있으면 무조건 200 이라,
Ollama 가 죽어도 워치독이 아무것도 하지 않고 로봇은 하루 종일 캔 답변만 했다.
지금은 Ollama 연결·모델 설치·모델 적재까지 보고 실패하면 **503** 을 준다. 워치독은
그 내용을 읽고 브로커를 살릴지 Ollama 를 살릴지 고른다. 모델이 내려가 있으면
미리 데워 둔다(안 그러면 다음 방문객이 40초 적재를 기다린다).

```
GET /health   {"ok":true,"backend":"OllamaBackend","model":"exaone3.5:7.8b","loaded":true}
GET /stats    {"counts":{"stream":212,"stream_error":3},
               "latency_s":{"stream":{"p50":0.41,"p90":0.68}}}
```

**`/stats` 는 로그를 안 뒤져도 문제를 보이게 한다.** 대화 8.7%가 통째로 실패하던
시절, 그 사실은 나중에 로그를 파고 나서야 드러났다. 이제 `status.sh` 가 건수와
실패율과 첫소리 지연을 바로 보여 준다.

**로봇의 심장박동.** `voice_chat` 이 5분마다 `심장박동:` 줄을 남긴다. 마이크
스트림이 물리거나 카메라가 프레임을 안 주면 프로세스는 살아 있어도 이 줄이
끊긴다 — `pi_watchdog.sh` 가 15분 침묵을 보고 서비스를 다시 올린다. 두 워치독 다
**쿨다운(10분)** 이 있다: 재시작으로 안 낫는 고장을 2분마다 영원히 흔드는 것이 더
나쁘다.

## 실행

평소엔 **전부 서비스로 자동**(전원만 켜면 됨). 아래는 손으로 돌릴 때.

```bash
# [Pi] 복도 데모 (서비스가 이걸 실행한다)
python3 voice_chat.py --motions --hallway --stt auto --fixed-energy 900 \
  --gaze-tilt 0 --camera-index main --collect-people \
  --url http://127.0.0.1:8080 --token <TOKEN>

# [Pi] 조용한 모드(불러야 반응) / STT 튜닝
python3 voice_chat.py --motions --url ... --token ...
python3 voice_chat.py --stt-test            # 들은 것 + captured RMS 만 출력

# [DGX] 브로커 (서비스: systemctl --user restart reachy-broker)
python3 broker/llm_broker.py --host 127.0.0.1 --port 8080 \
  --backend ollama --ollama-model exaone3.5:7.8b --token <TOKEN> \
  --system-prompt-file config/persona.txt \
  --motion-model opus --motion-prompt-file config/motion_prompt.txt

# [DGX] 로그 당겨와 분석·개선 (사람 사진 가져오기 포함)
bash ops/daily_update.sh

# [DGX] 사람 데이터만 따로 (daily_update 가 이 순서를 자동으로 돈다)
python3 ops/data/sync_persons.py                      # Pi -> 여기로 가져와 DB 합치기
$VENV/bin/python3 ops/data/backfill_person_boxes.py --write   # 사람 인식 DB 생성
python3 ops/data/persons_export.py --out --crop-persons       # 학습용으로 뽑기
python3 ops/data/sync_persons.py --stats              # 현황만 보기
python3 ops/data/sync_persons.py --purge-remote       # 옮긴 뒤 Pi 원본 삭제(SD 확보)
```

`$VENV` = `/home/kiro-ai/NAJY/trossen-ai-simulation/.venv` (torch 가 여기 있다)

**브로커 엔드포인트** (토큰은 `X-Auth-Token` 헤더로 보낸다 — 본문에 넣으면 401)

| 경로 | 하는 일 |
|---|---|
| `POST /reply` | 대화. `{"text": "...", "session": "..."}` |
| `POST /motion` | 동작 생성(opus). `{"text": "..."}` → `{say, preset, moves}` |
| `POST /reset` | 그 세션의 대화 맥락 비우기 |
| `GET /health` | 살아 있는지 + 어떤 백엔드인지 |

```bash
curl -s http://127.0.0.1:8080/reply -H 'X-Auth-Token: <TOKEN>' \
  -H 'Content-Type: application/json' -d '{"text":"안녕하세요"}'
```

## ops/ 파일

| 파일 | 하는 일 | 어디서 |
|---|---|---|
| `deploy.sh` | 로봇 코드·설정을 Pi 로 보낸다 | DGX |
| `daily_update.sh` | **하루 한 번 이것만 돌리면 된다.** 아래 순서를 자동으로 | DGX |
| `sync_persons.py` | Pi 사진 → 중앙 DB 로 가져오기·합치기 | DGX |
| `backfill_person_boxes.py` | 가져온 사진에서 사람 인식 DB 생성 (Faster R-CNN) | DGX |
| `persons_export.py` | 조건 걸어 학습용으로 뽑기 (사람 크롭 포함) | DGX |
| `log_digest.py` | 활동 다이제스트 (무엇이 얼마나 일어났나) | DGX |
| `build_tts_cache_list.py` | 자주 나온 답변 → 미리 합성할 목록 | DGX |
| `voice_chat.service` 외 | Pi 의 systemd 유닛들 | Pi |
| `99-reachy-cameras.rules` | 카메라 이름 고정 (main/sub) | Pi |
| `70-wifi-powersave-off.rules` | 와이파이 절전 끄기 (끊김 원인이었다) | Pi |
| `camera_tune.sh` / `respeaker_gain.py` | 카메라·마이크 설정 고정 | Pi |

**설치 스크립트**(1회): `ops/install_ollama.sh`(DGX 로컬 LLM),
`ops/pi/install_vosk.sh`(오프라인 STT), `ops/pi/install_object_vision.sh`(물체 검출 모델).
대용량 모델은 git 제외.

Pi 의존성: `pip3 install gTTS edge-tts SpeechRecognition vosk` + `sudo apt install mpg123 python3-pyaudio flac`
