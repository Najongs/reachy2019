"""Reachy 2019 base (rest) pose.

The upstream package does not define a rest pose anywhere; this module holds the
convention used by the test notebooks: every joint at 0, elbows at -90.

Usage:
    python base_pose.py                  # both arms + head, real hardware
    python base_pose.py --no-head        # skip the head (avoids the camera/opencv deps)
    python base_pose.py --io ws          # websocket io (Unity sim), no hardware needed
    python base_pose.py --release        # go to base pose, then turn every motor compliant
"""

import argparse
import logging


logger = logging.getLogger(__name__)


# Positions are in the user frame (offset / orientation already applied by
# reachy.parts.motor.DynamixelMotor), not the raw motor frame.
RIGHT_ARM_BASE = {
    'right_arm.shoulder_pitch': 0,
    'right_arm.shoulder_roll': 0,
    'right_arm.arm_yaw': 0,
    'right_arm.elbow_pitch': -90,
    'right_arm.hand.forearm_yaw': 0,
    'right_arm.hand.wrist_pitch': 0,
    'right_arm.hand.wrist_roll': 0,
    'right_arm.hand.gripper': 0,
}

LEFT_ARM_BASE = {
    'left_arm.shoulder_pitch': 0,
    'left_arm.shoulder_roll': 0,
    'left_arm.arm_yaw': 0,
    'left_arm.elbow_pitch': -90,
    'left_arm.hand.forearm_yaw': 0,
    'left_arm.hand.wrist_pitch': 0,
    'left_arm.hand.wrist_roll': 0,
    'left_arm.hand.gripper': 0,
}

HEAD_BASE = {
    'head.left_antenna': 0,
    'head.right_antenna': 0,
}

# Where the neck (Orbita) should point: straight ahead.
HEAD_LOOK_AT = (1, 0, 0)


POSE_DEFAULTS = {}
POSE_DEFAULTS.update(RIGHT_ARM_BASE)
POSE_DEFAULTS.update(LEFT_ARM_BASE)
POSE_DEFAULTS.update(HEAD_BASE)


def base_pose(reachy):
    """Build the base pose for the motors actually present on `reachy`.

    Keyed on the live motor list rather than the static tables, so a hand with
    a missing motor (e.g. this robot's left arm) just drops that entry instead
    of sending a goal to hardware that is not there.
    """
    return {
        m.name: POSE_DEFAULTS[m.name]
        for m in reachy.motors
        if m.name in POSE_DEFAULTS
    }


def stiffen(reachy):
    """Turn every motor stiff.

    Goal positions are silently ignored while a motor is compliant, so this has
    to run before any goto.
    """
    for m in reachy.motors:
        m.compliant = False

    if reachy.head is not None:
        reachy.head.compliant = False


def relax(reachy):
    """Turn every motor compliant (safe state to leave the robot in)."""
    for m in reachy.motors:
        m.compliant = True

    if reachy.head is not None:
        reachy.head.compliant = True


def goto_base_pose(reachy, duration=4, wait=True, interpolation_mode='minjerk'):
    """Move the whole robot to its base pose.

    Args:
        reachy (reachy.Reachy): robot to move
        duration (float): move duration (in seconds), keep it generous for the first move
        wait (bool): whether to block until the end of the motion
        interpolation_mode (str): 'linear' or 'minjerk'
    """
    stiffen(reachy)

    trajs = reachy.goto(
        goal_positions=base_pose(reachy),
        duration=duration,
        wait=wait,
        interpolation_mode=interpolation_mode,
    )

    # Reachy.goto only waits on the last motor of the dict, so wait on all of
    # them to make sure the whole pose is reached before returning.
    if wait:
        for t in trajs:
            t.wait()

    if reachy.head is not None:
        reachy.head.look_at(*HEAD_LOOK_AT, duration=min(duration, 2), wait=wait)

    return trajs


def connect(io='/dev/ttyUSB*', with_head=True,
            right_hand='gripper_no_wrist_roll',
            left_hand='gripper_no_wrist_roll'):
    """Connect to Reachy with both arms and, optionally, the head.

    실제 하드웨어 이력: 양쪽 wrist_roll 없음(2026-09-02 실측). 왼손은
    2026-09-08 모듈 교체로 그리퍼가 생겨 기본이 그리퍼 손이다. 혹시
    왼손 모듈이 옛 것(그리퍼 없음)으로 돌아오면 자동으로 구형 손으로
    한 번 더 시도한다 - 손 하나 때문에 로봇 전체가 소리만 모드로
    떨어지지 않게.
    """
    from reachy import Reachy, parts

    from custom_hands import register_hands
    register_hands()

    def _build(lh):
        kwargs = {
            'right_arm': parts.RightArm(io=io, hand=right_hand),
            'left_arm': parts.LeftArm(io=io, hand=lh),
        }
        if with_head:
            kwargs['head'] = parts.Head(io=io)
        return Reachy(**kwargs)

    try:
        return _build(left_hand)
    except Exception:
        if left_hand == 'wrist_pitch_only':
            raise
        logging.getLogger(__name__).warning(
            '왼손 그리퍼 연결 실패 - 구형 손(wrist_pitch_only)으로 재시도')
        return _build('wrist_pitch_only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--io', default='/dev/ttyUSB*',
                        help="io to use: serial port template or 'ws' for the websocket sim")
    parser.add_argument('--no-head', action='store_true', help='do not connect the head part')
    hand_choices = ['gripper_no_wrist_roll', 'wrist_pitch_only',
                    'force_gripper', 'orbita_wrist', 'empty_hand', 'none']
    parser.add_argument('--right-hand', default='gripper_no_wrist_roll', choices=hand_choices,
                        help="hand attached to the right arm (default matches this robot: "
                             "wrist_roll motor dxl_16 is missing)")
    parser.add_argument('--left-hand', default='gripper_no_wrist_roll', choices=hand_choices,
                        help="hand attached to the left arm (default matches this robot: "
                             "wrist_roll dxl_26 and gripper dxl_27 are missing)")
    parser.add_argument('--duration', type=float, default=4, help='move duration (in seconds)')
    parser.add_argument('--release', action='store_true',
                        help='turn all motors compliant once the base pose is reached')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    reachy = connect(
        io=args.io,
        with_head=not args.no_head,
        right_hand=None if args.right_hand == 'none' else args.right_hand,
        left_hand=None if args.left_hand == 'none' else args.left_hand,
    )

    try:
        goto_base_pose(reachy, duration=args.duration)
        logger.info('Base pose reached')

        if args.release:
            relax(reachy)
            logger.info('All motors compliant')
    finally:
        reachy.close()


if __name__ == '__main__':
    main()
