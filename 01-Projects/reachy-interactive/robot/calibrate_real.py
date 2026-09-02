"""Move ONE real joint back and forth so you can check it against the sim.

Run this INSTEAD of voice_chat (it needs the serial port). It powers only the
chosen arm, wiggles one joint slowly around its current pose, and streams the
live state to the web viewer via StateMirror (port 6171). Watch the viewer:
the real joint and the sim joint must move the same way.

    python3 calibrate_real.py --list
    python3 calibrate_real.py right_arm.shoulder_pitch
    python3 calibrate_real.py right_arm.elbow_pitch --amp 25 --io /dev/ttyUSB*

Ctrl-C to stop; the arm relaxes on exit.
"""
import argparse, time, logging

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('joint', nargs='?', help='e.g. right_arm.shoulder_pitch')
    ap.add_argument('--io', default='/dev/ttyUSB*')
    ap.add_argument('--amp', type=float, default=20, help='wiggle amplitude (deg)')
    ap.add_argument('--period', type=float, default=4, help='seconds per cycle')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    import numpy as np
    from base_pose import connect

    reachy = connect(io=args.io, with_head=True)
    from state_mirror import StateMirror
    StateMirror(reachy).start()
    print('미러 시작: 뷰어에서 ws://<Pi>:6171 [실물] 로 연결해 두세요.')

    names = [m.name for m in reachy.motors]
    if args.list or not args.joint:
        print('사용 가능 관절:')
        for n in names:
            print('  ', n)
        reachy.close(); return

    if args.joint not in names:
        print('그런 관절 없음:', args.joint); reachy.close(); return

    from operator import attrgetter
    motor = attrgetter(args.joint)(reachy)
    side = args.joint.split('.')[0]     # right_arm / left_arm
    arm = getattr(reachy, side, None)

    # power just this arm
    arm_motors = arm.motors if arm else [motor]
    for m in arm_motors:
        m.compliant = False
    time.sleep(0.1)
    for m in arm_motors:
        m.goal_position = m.present_position   # hold current, avoid jump
    time.sleep(0.1)

    center = motor.present_position
    print('관절 %s 중심 %.1f°, ±%.0f° 왕복. 실물과 시뮬 방향 비교하세요. (Ctrl-C 종료)'
          % (args.joint, center, args.amp))
    print('  → 슬라이더 + 방향 = 값이 커지는 방향입니다.')
    try:
        t0 = time.time()
        while True:
            t = time.time() - t0
            motor.goal_position = center + args.amp * np.sin(2*np.pi*t/args.period)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print('\n종료 - 팔 힘 뺍니다.')
    finally:
        for m in arm_motors:
            m.compliant = True
        reachy.close()

if __name__ == '__main__':
    main()
