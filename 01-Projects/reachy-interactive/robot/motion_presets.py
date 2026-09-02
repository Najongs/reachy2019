"""Named, hand-authored gestures for the motion pipeline.

Same schema as LLM-generated moves, and validated through the same safety
layer (validate_all() runs at broker/robot startup) - one execution path,
no privileged presets.

Sign conventions (user frame): negative shoulder_pitch = arm forward/up
(0 = hanging, -90 = pointing forward, -150 = up); negative elbow_pitch =
bend; negative right shoulder_roll / positive left shoulder_roll = arm
swings outward, away from the torso.
"""

from motion_exec import REST_POSE, validate, ValidationError, ALL_LIMITS


PRESETS = {
    'wave': {
        'description': '오른팔을 옆으로 들어 손을 흔든다',
        'moves': [
            {'pose': {'right_arm.shoulder_pitch': -20, 'right_arm.shoulder_roll': -60,
                      'right_arm.elbow_pitch': -100, 'right_arm.hand.forearm_yaw': 0},
             'duration': 1.6},
            {'pose': {'right_arm.hand.forearm_yaw': 30}, 'duration': 0.6},
            {'pose': {'right_arm.hand.forearm_yaw': -30}, 'duration': 0.6},
            {'pose': {'right_arm.hand.forearm_yaw': 30}, 'duration': 0.6},
            {'pose': {'right_arm.hand.forearm_yaw': 0}, 'duration': 0.6},
        ],
    },
    'bow': {
        'description': '안테나를 숙이며 가볍게 인사한다 (팔 안 씀)',
        'moves': [
            {'pose': {'head.left_antenna': 110, 'head.right_antenna': -110},
             'duration': 1.2},
            {'pose': {'head.left_antenna': 110, 'head.right_antenna': -110},
             'duration': 0.8},
            {'pose': {'head.left_antenna': 0, 'head.right_antenna': 0},
             'duration': 1.2},
        ],
    },
    'greet': {
        'description': '인사 - 안테나 꾸벅 후 오른손을 살짝 흔든다',
        'moves': [
            {'pose': {'head.left_antenna': 100, 'head.right_antenna': -100},
             'duration': 1.0},
            {'pose': {'head.left_antenna': 0, 'head.right_antenna': 0,
                      'right_arm.shoulder_pitch': -20, 'right_arm.shoulder_roll': -55,
                      'right_arm.elbow_pitch': -95}, 'duration': 1.6},
            {'pose': {'right_arm.hand.forearm_yaw': 25}, 'duration': 0.5},
            {'pose': {'right_arm.hand.forearm_yaw': -25}, 'duration': 0.5},
            {'pose': {'right_arm.hand.forearm_yaw': 0}, 'duration': 0.5},
        ],
    },
    'point_forward': {
        'description': '오른팔로 정면을 가리킨다',
        'moves': [
            {'pose': {'right_arm.shoulder_pitch': -70, 'right_arm.elbow_pitch': -15,
                      'right_arm.hand.wrist_pitch': 0, 'right_arm.hand.gripper': -40},
             'duration': 1.8},
            {'pose': {'right_arm.shoulder_pitch': -70}, 'duration': 1.2},
        ],
    },
    'hug_open': {
        'description': '양팔을 벌려 환영한다',
        'moves': [
            {'pose': {'right_arm.shoulder_pitch': -35, 'right_arm.shoulder_roll': -55,
                      'right_arm.elbow_pitch': -35,
                      'left_arm.shoulder_pitch': -35, 'left_arm.shoulder_roll': 55,
                      'left_arm.elbow_pitch': -35}, 'duration': 2.0},
            {'pose': {'right_arm.shoulder_roll': -55, 'left_arm.shoulder_roll': 55},
             'duration': 1.0},
        ],
    },
    'cheer': {
        'description': '만세 - 양팔을 번쩍 든다',
        'moves': [
            {'pose': {'right_arm.shoulder_pitch': -130, 'right_arm.shoulder_roll': -20,
                      'right_arm.elbow_pitch': -20,
                      'left_arm.shoulder_pitch': -130, 'left_arm.shoulder_roll': 20,
                      'left_arm.elbow_pitch': -20,
                      'head.left_antenna': 40, 'head.right_antenna': -40},
             'duration': 2.2},
            {'pose': {'right_arm.elbow_pitch': -45, 'left_arm.elbow_pitch': -45},
             'duration': 0.6},
            {'pose': {'right_arm.elbow_pitch': -20, 'left_arm.elbow_pitch': -20},
             'duration': 0.6},
        ],
    },
    'handshake': {
        'description': '오른손을 앞으로 내밀어 악수를 청한다',
        'moves': [
            {'pose': {'right_arm.shoulder_pitch': -60, 'right_arm.elbow_pitch': -30,
                      'right_arm.hand.wrist_pitch': 0, 'right_arm.hand.gripper': -30},
             'duration': 1.8},
            {'pose': {'right_arm.shoulder_pitch': -60}, 'duration': 2.0},
            {'pose': {'right_arm.shoulder_pitch': -55}, 'duration': 0.5},
            {'pose': {'right_arm.shoulder_pitch': -60}, 'duration': 0.5},
        ],
    },
    'pick_to_tray': {
        'description': '앞의 물건을 집어 왼쪽 보관함에 넣는다 (FK 계산 자세 기반)',
        'moves': [
            # 물건 상공에서 그리퍼 벌리고 대기 (몸 앞 x0.31, y-0.14, z-0.16)
            {'pose': {'right_arm.shoulder_pitch': 0, 'right_arm.shoulder_roll': -20,
                      'right_arm.arm_yaw': 20, 'right_arm.elbow_pitch': -115,
                      'right_arm.hand.gripper': -45}, 'duration': 2.2},
            # 테이블 높이로 하강 (z-0.27)
            {'pose': {'right_arm.elbow_pitch': -95}, 'duration': 1.6},
            {'grasp': 'close'},
            # 들어올림
            {'pose': {'right_arm.elbow_pitch': -115}, 'duration': 1.4},
            # 보관함 상공으로 이동 (x0.29, y+0.14, z-0.18)
            {'pose': {'right_arm.shoulder_pitch': -25, 'right_arm.shoulder_roll': 5,
                      'right_arm.arm_yaw': 60, 'right_arm.elbow_pitch': -85},
             'duration': 2.6},
            {'grasp': 'open'},
            # 물러나기
            {'pose': {'right_arm.shoulder_pitch': 0, 'right_arm.shoulder_roll': -20,
                      'right_arm.arm_yaw': 20, 'right_arm.elbow_pitch': -115},
             'duration': 2.2},
        ],
    },
    'antenna_dance': {
        'description': '안테나만 신나게 흔든다',
        'moves': [
            {'pose': {'head.left_antenna': 60, 'head.right_antenna': -60}, 'duration': 0.5},
            {'pose': {'head.left_antenna': -60, 'head.right_antenna': 60}, 'duration': 0.5},
            {'pose': {'head.left_antenna': 60, 'head.right_antenna': -60}, 'duration': 0.5},
            {'pose': {'head.left_antenna': -60, 'head.right_antenna': 60}, 'duration': 0.5},
            {'pose': {'head.left_antenna': 0, 'head.right_antenna': 0}, 'duration': 0.6},
        ],
    },
}


def preset_lines():
    """One-line-per-preset summary for the motion system prompt."""
    return '\n'.join(
        '- {}: {}'.format(name, info['description'])
        for name, info in sorted(PRESETS.items())
    )


def validate_all(available_joints=None):
    """Every preset must pass the same safety layer as generated moves.

    Returns {name: segments}. Raises if any preset fails - presets are code,
    a failure here is a bug to fix, not a runtime condition to tolerate.
    """
    if available_joints is None:
        available_joints = list(ALL_LIMITS)

    seed = dict(REST_POSE)
    seed.update({'head.left_antenna': 0, 'head.right_antenna': 0})

    validated = {}
    for name, info in PRESETS.items():
        try:
            validated[name] = validate(info['moves'], seed, available_joints)
        except ValidationError as e:
            raise AssertionError('Preset {!r} fails validation: {}'.format(name, e))
    return validated


if __name__ == '__main__':
    result = validate_all()
    for name, segments in sorted(result.items()):
        total = sum(d for kind, d in segments if kind != 'grasp')
        grasps = sum(1 for kind, _ in segments if kind == 'grasp')
        note = ' (+{} grasp)'.format(grasps) if grasps else ''
        print('{:15s} {:2d} segments, {:.1f}s{}'.format(name, len(segments), total, note))
    print('all presets pass the safety layer')
