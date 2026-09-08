"""Hand classes matching this robot's actual hardware.

Probing the buses (2026-09-02) showed both wrist_roll motors are physically
gone, and the left gripper too:

    right arm: dxl_10-15, dxl_17 + force sensor   (no dxl_16 / wrist_roll)
    left arm:  dxl_20-25                          (no dxl_26 / wrist_roll, no dxl_27 / gripper)

The stock hands ('force_gripper', 'empty_hand') look those motors up at
connection time and die with LuosModuleNotFoundError, so these classes mirror
what is really there. The missing wrist_roll's link offset (-0.0325 m) is
folded into the next link so forward/inverse kinematics stay correct.

Usage:
    from custom_hands import register_hands
    register_hands()
    arm = parts.RightArm(io=..., hand='gripper_no_wrist_roll')
    arm = parts.LeftArm(io=..., hand='wrist_pitch_only')
"""

from collections import OrderedDict

from reachy.parts.hand import ForceGripper, Hand


class RightForceGripperNoWristRoll(ForceGripper):
    """Right force gripper without the (missing) wrist_roll motor.

    Same as RightForceGripper minus dxl_16. The gripper link translation grows
    by wrist_roll's -0.0325 m so the end effector stays where it really is.
    """

    dxl_motors = OrderedDict([
        ('forearm_yaw', {
            'id': 14, 'offset': 0.0, 'orientation': 'indirect',
            'angle-limits': [-100, 100],
            'link-translation': [0, 0, 0], 'link-rotation': [0, 0, 1],
        }),
        ('wrist_pitch', {
            'id': 15, 'offset': 0.0, 'orientation': 'indirect',
            'angle-limits': [-45, 45],
            'link-translation': [0, 0, -0.25], 'link-rotation': [0, 1, 0],
        }),
        ('gripper', {
            'id': 17, 'offset': 0.0, 'orientation': 'direct',
            'angle-limits': [-69, 20],
            'link-translation': [0, -0.01, -0.1075], 'link-rotation': [0, 0, 0],
        }),
    ])


class LeftForceGripperNoWristRoll(ForceGripper):
    """왼손 그리퍼 (wrist_roll 없음). 2026-09-08 왼손 모듈 교체로 그리퍼 장착.

    스톡 LeftForceGripper 에서 dxl_26(wrist_roll)만 뺐다. 사라진 링크
    오프셋(-0.0325m)은 그리퍼 링크에 접어 넣어 기구학이 맞다.
    한계 [-20, 69]: 오른손([-69, 20])의 거울 - 양수=벌림.
    """

    dxl_motors = OrderedDict([
        ('forearm_yaw', {
            'id': 24, 'offset': 0.0, 'orientation': 'indirect',
            'angle-limits': [-100, 100],
            'link-translation': [0, 0, 0], 'link-rotation': [0, 0, 1],
        }),
        ('wrist_pitch', {
            'id': 25, 'offset': 0.0, 'orientation': 'indirect',
            'angle-limits': [-45, 45],
            'link-translation': [0, 0, -0.25], 'link-rotation': [0, 1, 0],
        }),
        ('gripper', {
            'id': 27, 'offset': 0.0, 'orientation': 'direct',
            'angle-limits': [-20, 69],
            'link-translation': [0, -0.01, -0.1075], 'link-rotation': [0, 0, 0],
        }),
    ])


class LeftWristPitchOnlyHand(Hand):
    """Left hand with just forearm_yaw + wrist_pitch.

    wrist_roll (dxl_26) and the gripper (dxl_27) are physically absent, and so
    is the force sensor - hence a bare Hand, not a ForceGripper.
    """

    fans = {'wrist_fan': 'hand.wrist_pitch'}

    dxl_motors = OrderedDict([
        ('forearm_yaw', {
            'id': 24, 'offset': 0.0, 'orientation': 'indirect',
            'angle-limits': [-100, 100],
            'link-translation': [0, 0, 0], 'link-rotation': [0, 0, 1],
        }),
        ('wrist_pitch', {
            'id': 25, 'offset': 0.0, 'orientation': 'indirect',
            'angle-limits': [-45, 45],
            'link-translation': [0, 0, -0.25], 'link-rotation': [0, 1, 0],
        }),
    ])

    def __init__(self, root, io):
        """Create the two-axis left hand."""
        Hand.__init__(self, root=root, io=io)

        dxl_motors = OrderedDict({
            name: dict(conf)
            for name, conf in self.dxl_motors.items()
        })
        self.attach_dxl_motors(dxl_motors)


def register_hands():
    """Make the custom hands usable as Arm(hand=...) names."""
    from reachy.parts import arm

    arm.hands['gripper_no_wrist_roll'] = {'right': RightForceGripperNoWristRoll,
                                          'left': LeftForceGripperNoWristRoll}
    arm.hands['wrist_pitch_only'] = {'left': LeftWristPitchOnlyHand}
