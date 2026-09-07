"""reachy.glb 에서 링크별 메시를 뽑아 MuJoCo 가 쓸 수 있게 만든다.

HTML 뷰어가 실제 로봇처럼 보이는 이유는 CAD 메시를 쓰기 때문이다. 그 메시가
이미 저장소에 있고(03-Resources/3d-models/reachy.glb), 노드 그래프에 관절
이름(right_shoulder_pitch_joint 등)이 그대로 박혀 있다 - 뷰어는 그 이름으로
노드를 찾아 돌린다. 같은 이름으로 메시를 갈라내면 MuJoCo 에도 쓸 수 있다.

**기구학은 손대지 않는다.** 메시는 motion_exec.CHAINS 로 만든 바디에 붙이기만
한다. CAD 의 관절 노드 위치가 내 사슬과 몇 군데 다르지만(shoulder_pitch 6.6cm,
arm_yaw 7.6cm, forearm_yaw 7.9cm), 전부 **자기 회전축 위에서 피벗이 미끄러진
것**이라 축 자체는 같다 - shoulder_pitch 는 y축인데 차이도 y 성분뿐이다.
기구학적으로 동일하므로 붙여도 어긋나지 않는다.

쉬는 자세에서는 모든 회전이 항등이므로, 메시를 월드 좌표로 펴고 그 바디의
휴식 위치를 빼면 바로 바디 로컬 좌표가 된다.

사용:
    python3 sim/extract_meshes.py            # sim/meshes/*.obj 로 저장
    python3 sim/extract_meshes.py --report   # 무엇이 어디에 붙는지만 출력
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'robot'))

import motion_exec as me                          # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import para  # PARA 기준 경로
GLB = os.path.join(para.RESOURCES, '3d-models',
                   'reachy.glb')
OUT = os.path.join(HERE, 'meshes')

# glTF 관절 노드 -> 이 프로젝트의 MJCF 바디 이름 (sim_mujoco._body_name 과 같은 규칙).
# 뷰어(sim_viewer.html)의 J 표와 같은 대응이다.
NODE_TO_BODY = {
    'right_shoulder_pitch_joint': 'right_arm_shoulder_pitch',
    'right_shoulder_roll_joint': 'right_arm_shoulder_roll',
    'right_arm_yaw_joint': 'right_arm_arm_yaw',
    'right_elbow_pitch_joint': 'right_arm_elbow_pitch',
    'right_forearm_yaw_joint': 'right_arm_hand_forearm_yaw',
    'right_wrist_pitch_joint': 'right_arm_hand_wrist_pitch',
    # 이 로봇에 없는 관절(wrist_roll)은 손목에 고정된 것으로 본다.
    'right_wrist_roll_joint': 'right_arm_hand_wrist_pitch',
    # 그리퍼는 실물처럼 회전하는 별도 바디다 (viewer 의 right_gripper_joint).
    'right_gripper_joint': 'right_arm_hand_gripper',
    'left_shoulder_pitch_joint': 'left_arm_shoulder_pitch',
    'left_shoulder_roll_joint': 'left_arm_shoulder_roll',
    'left_arm_yaw_joint': 'left_arm_arm_yaw',
    'left_elbow_pitch_joint': 'left_arm_elbow_pitch',
    'left_forearm_yaw_joint': 'left_arm_hand_forearm_yaw',
    'left_wrist_pitch_joint': 'left_arm_hand_wrist_pitch',
    'left_wrist_roll_joint': 'left_arm_hand_wrist_pitch',
    'left_gripper_joint': 'left_arm_hand_wrist_pitch',
    'left_ear_joint': 'left_antenna',
    'right_ear_joint': 'right_antenna',
    # 머리 전체(neck_joint 서브트리)는 목을 따라 도는 별도 바디다 -
    # 통짜 world 에 구우면 카메라만 돌고 머리는 정지해 보인다.
    'neck_joint': 'head',
}


def body_rest_positions():
    """각 MJCF 바디의 휴식 자세 월드 위치. 메시를 로컬로 옮길 때 쓴다."""
    rest = {}
    for side, chain in me.CHAINS.items():
        for i, (jname, _t, _a, _l) in enumerate(chain):
            body = jname.replace('.', '_')
            rest[body] = me.forward_kinematics(chain, {}, upto=i + 1)
    # 안테나는 사슬 밖이다. sim_mujoco 가 쓰는 위치와 같게 둔다.
    z = me.HEAD_SPHERE_CENTER[2] + me.HEAD_SPHERE_RADIUS - 0.01
    rest['left_antenna'] = (-0.02, 0.06, z)
    rest['right_antenna'] = (-0.02, -0.06, z)
    # 머리 바디는 eye_tilt 밑에 붙는다 (sim_mujoco 의 팬/틸트 사슬 휴식
    # 위치: eye_pan (0,0,0.09) + eye_tilt (0.06,0,0.02)).
    rest['head'] = (0.06, 0.0, 0.11)
    rest['world'] = (0.0, 0.0, 0.0)
    return rest


def collect(glb=GLB):
    """(바디 이름 -> 합쳐진 메시) 와 어느 노드가 어디에 붙었는지."""
    import numpy as np
    import trimesh

    scene = trimesh.load(glb)
    parents = scene.graph.transforms.parents
    groups, owners = {}, {}

    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        geom = scene.geometry[gname]
        if max(geom.extents) >= 1.0:
            continue                                # 배경 실린더
        # 가장 가까운 '아는 관절' 조상을 찾는다. 없으면 몸통(world).
        owner, cur = 'world', node
        seen = 0
        while cur in parents and seen < 60:
            cur = parents[cur]
            seen += 1
            if cur in NODE_TO_BODY:
                owner = NODE_TO_BODY[cur]
                break
        m = geom.copy()
        m.apply_transform(T)                        # 월드 좌표로
        groups.setdefault(owner, []).append(m)
        owners.setdefault(owner, []).append(node)

    rest = body_rest_positions()
    merged = {}
    for owner, parts in groups.items():
        m = trimesh.util.concatenate(parts)
        m.apply_translation(-np.array(rest.get(owner, (0, 0, 0))))
        merged[owner] = m
    return merged, owners


def export(glb=GLB, out=OUT):
    os.makedirs(out, exist_ok=True)
    merged, owners = collect(glb)
    written = {}
    for owner, mesh in merged.items():
        path = os.path.join(out, owner + '.obj')
        mesh.export(path)
        written[owner] = path
    return written, merged, owners


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--glb', default=GLB)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    if args.report:
        merged, owners = collect(args.glb)
        for owner in sorted(merged):
            m = merged[owner]
            print('  %-30s 메시 %2d개, 삼각형 %6d, 크기 %.2f x %.2f x %.2f m'
                  % (owner, len(owners[owner]), len(m.faces), *m.extents))
        return

    written, merged, owners = export(args.glb, args.out)
    total = sum(len(m.faces) for m in merged.values())
    print('링크 %d개, 삼각형 %d개 -> %s' % (len(written), total, args.out))
    for owner in sorted(written):
        print('  %-30s %6d 삼각형  %.1fKB' % (
            owner, len(merged[owner].faces),
            os.path.getsize(written[owner]) / 1024))


if __name__ == '__main__':
    main()
