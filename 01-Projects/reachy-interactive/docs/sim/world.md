# MuJoCo 월드

## 기구학은 실물과 같다

MJCF 를 손으로 쓰지 않고 `motion_exec.CHAINS`(실물 검증기와 같은 체인)에서
생성한다 (`sim_mujoco.build_mjcf`). FK 불일치 0.0000000m 검증됨.
**여기서 안전하면 실물에서도 안전하다** 가 성립하는 근거.

- **qpos 는 항상 라디안**. `compiler angle="degree"` 는 XML 속성에만
  적용된다. 도 단위를 qpos 에 넣으면 40도가 40rad 이 된다 (겪었다).
- 관절 인덱스는 **모델의 모든 관절**을 돌아 만든다. 팔 관절만 돌리면
  눈 관절이 안 움직이는데 서보는 "동작" 한다 (겪었다).

## 목 (Orbita)

실물 `orbita` 패키지의 역기구학 그대로 (Pc_z=[0,0,23], R=35.9,
R0=rot(z,60)@rot(y,10)). pan/tilt+roll 힌지 근사가 방향·롤 오차 0.000°.

- **시선은 순간이동하지 않는다**: `World.glide_gaze`(목표까지 나눠 돌기),
  `World.nudge_gaze`(한 스텝만 - 추적용). 스텝 각도는
  `gaze_step_deg` 파라미터. 실물 목 속도 상한과 같은 원칙.
- 팔이 움직일 때: 들어올리기·서보 모두 **손-목표 중간점**을 본다. 손이
  다가갈수록 시선이 목표로 수렴하고, 손과 목표가 같이 화면에 담긴다
  (visual servoing 요건이자 품질의 '시야' 성분).

## 카메라

- `robot_eye`: 머리에 붙어 목을 따라간다. **xyaxes 로 완전 지정**해야
  한다 - zaxis 만 주면 롤이 임의라 90도 돌아간 그림이 나온다 (겪었다).
- 제3자: `sim_servo._third_person` 고정 카메라 (azimuth 150). 평가
  전용. azimuth 180 = 정면.

## 외형과 충돌

- 시각 메시: `sim/extract_meshes.py` 가 reachy.glb 를 관절 조상 기준으로
  쪼갠 `sim/meshes/*.obj`.
- 충돌은 캡슐 유지: 실측 반지름 상완 4.4 / 전완 3.7 / 손 4.4cm
  (`measure_arm.py`), 관절 여유 p95 5cm.
- 투과 검사: `World.clearance` 가 mj_geomDistance 로 팔 전체 vs 물체들의
  최소 표면거리. 프레임마다 재서 경로 전체를 본다.

## 장면

테이블 상판 z=-0.27. 물체 원형: 컵(주황 원기둥)·블록(파랑)·공(노랑)·
캔(회색)·쟁반. `sim_tasks._place_objects` 가 겹치지 않게 흩는다.
시작 자세는 실물 휴식(전부 0) - 예전 '대기' 자세는 손이 테이블 밑이라
서보가 상판을 뚫고 올라왔다.
