# Robot Ops — 운영 문서

Reachy 로봇의 하드웨어 특성, 네트워크, 부팅, 배포를 정리. 코드는 `../../01-Projects/reachy-interactive/`.

## 하드웨어 (이 로봇 실물)

- **없는 모터** (2026-09-02 버스 스캔): 오른팔 wrist_roll(dxl16), 왼팔 wrist_roll(dxl26)·gripper(dxl27)·힘센서. 기본 손 클래스는 연결 시 죽으므로 `robot/custom_hands.py` 사용
- **왼손 = 3D 프린팅 보관함(트레이)**, 오른손 = 힘센서 그리퍼. pick&place: 오른손이 집고 왼쪽 트레이가 받음
- 테이블 면 ≈ z -0.27 (토르소 기준). 팔 늘어뜨리면 손이 상판 아래로
- 관절: 사용자 프레임. shoulder_pitch 음수=앞/위(들 땐 팔꿈치 굽힐 것 — 편 채 들면 스톨), elbow_pitch 음수=굽힘
- **상완이 몸통과 rest 에서 ~3cm** 뿐 → SAFE_LIMITS 로 shoulder_roll 좁힘
- **목 = Orbita**(3디스크 병렬). 각도 매핑 불가. 아래를 오래 응시하면 과열로 힘빠짐 → GAZE_TILT 완화 + neck 하트비트
- compliant 상태에선 goal_position 무시 → stiffen 직후 seed 재기록(팔 튐 방지)
- Pi: **라즈베리파이 4, armv7l(32bit), Buster, Python 3.7**. 최신 Anthropic SDK 설치 불가 → llm_client 는 stdlib urllib

## 네트워크

- Pi(192.168.0.20) → 공유기(WAN 172.16.211.x) → 인터넷
- DGX: 인터넷에서 **210.183.240.125:20411** (호스트키 검증됨)
- **Pi ↔ DGX 직접 라우팅 안 됨** → Pi 가 나가는 SSH 에 포워딩 실어보냄:
  - `-L 8080` : Pi 127.0.0.1:8080 = DGX 브로커
  - `-R 2222` : DGX 에서 `ssh -p 2222 pi@localhost` = Pi (역방향 배포)
  - `-R 8090/6171/6172` : 뷰어·미러 확인
- `ops/pi_tunnel.service` (systemd, 자동 재연결). 키: `~/.ssh/dgx_tunnel`

## 부팅 (2026-09-02 정리 후)

전원만 켜면 자동:
- 와이파이 자동연결 (dhcpcd AP 잔재 제거 — 백업 `/etc/dhcpcd.conf.bak-ap-mode`)
- DGX 터널 자동 (`pi_tunnel.service`)
- 뷰어 서빙 자동 (`pi_viewer.service`, 8090)
- demo.py 자동실행 없음 (demo.service disabled)

로봇 실행은 이것만: `cd ~/Documents && python3 voice_chat.py --motions --url http://127.0.0.1:8080 --token <TOKEN>`

## 배포 (DGX → Pi)

역터널(2222) 경유. `ops/deploy.sh` 한 번이면 전 파일 전송 + 버전 대조.
```bash
cd 01-Projects/reachy-interactive/ops && ./deploy.sh
```
Pi 파일은 `~/Documents/` 에 평면 배치(경로 고정). 브로커는 **DGX 에서 수동 실행**(재부팅 시 재기동 필요; `curl localhost:8080/health` 로 확인).

## 로그

- Pi: `~/reachy_logs/<세션>/events.jsonl` + `frames/` — 대화·비전·동작(생성 JSON, 실행결과, 잡기성공, 스톨)
- DGX: `python3 broker/export_logs.py` → 읽기 좋은 형태로 (git 제외)
