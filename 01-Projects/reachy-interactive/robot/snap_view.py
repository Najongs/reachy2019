"""Point the head down at the table and capture what the robot sees.

Unpowered, the Orbita neck flops backwards so the camera stares at the
ceiling - this connects the head, tilts it toward the table, snaps a frame,
and leaves the head powered so consecutive shots stay aimed.

Usage (on the Pi; stop anything else using /dev/ttyUSB* first):
    python3 snap_view.py                          # look at table, save /tmp/setup_view.jpg
    python3 snap_view.py --z -0.5 --out /tmp/a.jpg   # look further down
    python3 snap_view.py --release                # let the head go afterwards
"""

import argparse
import logging
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='/tmp/setup_view.jpg')
    parser.add_argument('--io', default='/dev/ttyUSB*')
    parser.add_argument('--x', type=float, default=0.5, help='gaze forward (m)')
    parser.add_argument('--z', type=float, default=-0.35,
                        help='gaze height; more negative = further down')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--release', action='store_true',
                        help='make the head compliant again before exiting')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    import cv2
    from reachy import Reachy, parts

    reachy = Reachy(head=parts.Head(io=args.io))
    try:
        reachy.head.compliant = False
        time.sleep(0.2)
        reachy.head.look_at(args.x, 0, args.z, duration=1.5, wait=True)
        time.sleep(0.4)   # let auto-exposure settle on the new view

        capture = cv2.VideoCapture(args.camera_index)
        try:
            frame = None
            for _ in range(6):
                success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError('camera read failed')
            cv2.imwrite(args.out, frame)
            print('saved:', args.out)
        finally:
            capture.release()

        if args.release:
            reachy.head.compliant = True
    finally:
        reachy.close()


if __name__ == '__main__':
    main()
