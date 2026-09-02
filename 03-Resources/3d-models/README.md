# 3D 모델 (reachy.glb)

시뮬 뷰어가 쓰는 실물 CAD 모델. **용량(8MB)이라 git 제외** — 아래로 재생성.

## 재생성

1. Pollen 공식 Blender 모델 다운로드:
   `https://github.com/pollen-robotics/reachy-blender/releases/download/v0.1/models.zip`
2. Blender(3.6+) 헤드리스로 Z-up glTF 내보내기 (나사·자석 제거 + 데시메이트 0.35):
   ```
   blender -b models/reachy_full_assembly.blend --python export.py -- reachy.glb
   ```
   export.py: 잔부품(screw/magnet/nut/bearing/washer/insert) 삭제 →
   2000폴리곤↑ 메시에 DECIMATE 0.35 →
   `export_scene.gltf(export_format='GLB', export_apply=True, export_yup=False)`
3. 결과 `reachy.glb` 를 `sim/` 옆에 배치(뷰어가 상대경로 로드) 또는 서버 디렉토리에.

## 관절 노드

glTF 에 `<side>_<joint>_joint` 19개 노드. 뷰어는 전부 **로컬 z축** 회전으로 구동,
부호는 실물 대조로 확정 (`sim/sim_viewer.html` J 테이블). Z-up 이므로 뷰어에서 root.rotation.x=-90.
