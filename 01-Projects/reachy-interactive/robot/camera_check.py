"""카메라 초점·수평·얼굴검출 확인 도구 (Pi에서 직접 실행).

Reachy 머리에는 카메라가 둘(좌/우) 있다. C270 은 고정초점이라 소프트웨어로 초점을
못 맞추고, 렌즈 배럴을 손으로 돌려야 한다. 맨눈으로는 어디가 제일 선명한지 알기
어려우므로, 이 도구가 카메라마다 선명도를 숫자와 막대로 실시간 표시한다.
숫자가 가장 커지는 지점에서 링을 멈추면 된다.

사용법 (Pi 에서):
    python3 camera_check.py                 # 잡히는 카메라 전부 나란히 보기
    python3 camera_check.py --device 0      # 특정 카메라만
    python3 camera_check.py --text          # 숫자만 (창 없이)
    python3 camera_check.py --save /tmp/a   # 각 카메라를 한 장씩 저장하고 종료
    python3 camera_check.py --seconds 60    # 60초 뒤 자동 종료

창 모드 키:  q 종료 · s 사진 저장 · r 최고기록 초기화

카메라는 voice_chat 이 계속 쓰고 있으므로 기본적으로 잠시 멈췄다가
종료할 때 자동으로 다시 켠다 (--keep-service 로 끄지 않게 할 수 있다).
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time


def service_active(name='voice_chat'):
    try:
        out = subprocess.run(['systemctl', 'is-active', name],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() == 'active'
    except Exception:
        return False


def service(action, name='voice_chat'):
    try:
        subprocess.run(['sudo', 'systemctl', action, name], timeout=30)
        return True
    except Exception:
        return False


def capture_devices():
    """실제로 영상이 나오는 /dev/videoN 만 골라 인덱스로 돌려준다.

    요즘 UVC 드라이버는 카메라 한 대당 노드를 둘 만든다(하나는 메타데이터).
    그래서 /dev/video* 를 세면 카메라 수가 두 배로 보인다.
    """
    found = []
    for path in sorted(glob.glob('/dev/video[0-9]*'),
                       key=lambda p: int(re.sub(r'\D', '', p) or 0)):
        idx = int(re.sub(r'\D', '', path) or 0)
        if idx >= 10:          # video10+ 는 파이 내장 코덱
            continue
        try:
            out = subprocess.run(['v4l2-ctl', '-d', path, '--info'],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            continue
        # "Device Caps" 블록에 Video Capture 가 있어야 진짜 카메라
        caps = out.split('Device Caps')[-1]
        if 'Video Capture' in caps:
            bus = ''
            m = re.search(r'Bus info\s*:\s*(\S+)', out)
            if m:
                bus = m.group(1)
            found.append((idx, bus + role_of(idx)))
    return found


def role_of(index):
    """udev 로 고정한 역할 이름(main/sub)을 붙여 준다 (ops/99-reachy-cameras.rules)."""
    for role, link in (('main', '/dev/reachy-cam-main'),
                       ('sub', '/dev/reachy-cam-sub')):
        try:
            if os.path.realpath(link) == '/dev/video%d' % index:
                return '  [%s]' % role
        except Exception:
            pass
    return ''


def sharpness(gray, cv2):
    """라플라시안 분산 = 선명도. 초점이 맞을수록 커진다."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def tilt_hint(gray, cv2, np):
    """화면의 직선들이 몇 도 기울었는지 추정 (수평 맞출 때 참고용)."""
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                            minLineLength=90, maxLineGap=12)
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 25:                     # 수평에 가까운 선만
            angles.append(a)
    return float(np.median(angles)) if angles else None


class Cam(object):
    """열려 있는 카메라 하나 + 그 카메라의 측정값."""

    def __init__(self, index, bus, cv2, width=640, height=480):
        self.index = index
        self.bus = bus
        self.cap = cv2.VideoCapture(index)
        # 두 대를 동시에 열면 USB 대역이 빠듯하다. MJPG 로 두면 여유가 생긴다.
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except Exception:
            pass
        self.frame = None
        self.sharp = 0.0
        self.best = 0.0
        self.faces = []
        self.ok = self.cap.isOpened()

    def read(self, cv2, cascade):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False
        self.frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.gray = gray
        self.sharp = sharpness(gray, cv2)
        self.best = max(self.best, self.sharp)
        self.faces = cascade.detectMultiScale(
            cv2.equalizeHist(cv2.resize(gray, (320, 240))), 1.2, 5,
            minSize=(30, 30))
        return True

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', type=int, action='append',
                    help='특정 카메라만 (여러 번 쓰면 여러 대). 생략하면 전부')
    ap.add_argument('--text', action='store_true', help='창 없이 숫자만 출력')
    ap.add_argument('--seconds', type=float, default=0, help='이 시간 뒤 자동 종료')
    ap.add_argument('--save', help='각 카메라를 한 장씩 저장하고 종료 (접두사)')
    ap.add_argument('--keep-service', action='store_true',
                    help='voice_chat 을 멈추지 않는다')
    args = ap.parse_args()

    import cv2
    import numpy as np

    devices = capture_devices()
    if args.device:
        devices = [(i, b) for (i, b) in devices if i in args.device] or \
                  [(i, '') for i in args.device]
    if not devices:
        print('캡처 가능한 카메라를 찾지 못했습니다. (lsusb 로 연결 확인)')
        return 1

    print('카메라 %d대 발견: %s' % (
        len(devices), ', '.join('/dev/video%d' % i for i, _ in devices)))
    for i, b in devices:
        print('   /dev/video%d  %s' % (i, b))

    stopped = False
    if not args.keep_service and service_active():
        print('voice_chat 을 잠시 멈춥니다 (종료할 때 자동으로 다시 켭니다)...')
        stopped = service('stop')
        time.sleep(2)

    cams = [Cam(i, b, cv2) for i, b in devices]
    cams = [c for c in cams if c.ok]
    if not cams:
        print('카메라를 열 수 없습니다. 다른 프로그램이 쓰고 있는지 확인하세요.')
        if stopped:
            service('start')
        return 1

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # DISPLAY 가 비어 있어도 X 가 떠 있으면 그 화면에 창을 띄운다 (SSH/sudo 대응).
    gui = False
    if not args.text and not args.save:
        if not os.environ.get('DISPLAY'):
            try:
                socks = sorted(os.listdir('/tmp/.X11-unix'))
            except OSError:
                socks = []
            if socks:
                os.environ['DISPLAY'] = ':' + socks[0].lstrip('X')
                print('DISPLAY 가 없어서 %s 로 자동 설정했습니다.' % os.environ['DISPLAY'])
        if os.environ.get('DISPLAY'):
            try:
                probe = np.zeros((40, 80, 3), dtype=np.uint8)
                cv2.imshow('probe', probe)
                cv2.waitKey(1)
                cv2.destroyWindow('probe')
                gui = True
            except Exception as e:
                print('창을 띄울 수 없어 숫자 모드로 실행합니다: %s' % e)
        else:
            print('화면(X)을 찾지 못해 숫자 모드로 실행합니다.')

    t0 = time.time()
    last_print = 0.0
    try:
        while True:
            got = [c for c in cams if c.read(cv2, cascade)]
            if not got:
                time.sleep(0.05)
                continue

            if args.save:
                for c in got:
                    path = '%s_cam%d.jpg' % (args.save.rstrip('.jpg'), c.index)
                    cv2.imwrite(path, c.frame)
                    print('저장: %s  (선명도 %.0f, 얼굴 %d)'
                          % (path, c.sharp, len(c.faces)))
                break

            if gui:
                panels = []
                for c in got:
                    view = c.frame.copy()
                    h, w = view.shape[:2]
                    cv2.line(view, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
                    cv2.line(view, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
                    bar = int(min(1.0, c.sharp / 400.0) * (w - 40))
                    cv2.rectangle(view, (20, h - 40), (20 + bar, h - 22),
                                  (0, 220, 0), -1)
                    cv2.putText(view, 'cam%d%s  sharp %.0f (best %.0f)'
                                % (c.index, role_of(c.index), c.sharp, c.best),
                                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)
                    for (x, y, fw, fh) in c.faces:
                        s = view.shape[1] / 320.0
                        cv2.rectangle(view, (int(x * s), int(y * s)),
                                      (int((x + fw) * s), int((y + fh) * s)),
                                      (255, 120, 0), 2)
                    panels.append(view)

                if len(panels) > 1:
                    h = min(p.shape[0] for p in panels)
                    panels = [p[:h] for p in panels]
                    canvas = np.hstack(panels)
                else:
                    canvas = panels[0]
                cv2.putText(canvas, 'q:quit  s:save  r:reset', (20, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow('reachy cameras - turn each lens, watch "sharp"', canvas)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                if k == ord('s'):
                    stamp = int(time.time())
                    for c in got:
                        p = '/tmp/camera_check_%d_cam%d.jpg' % (stamp, c.index)
                        cv2.imwrite(p, c.frame)
                        print('저장:', p)
                if k == ord('r'):
                    for c in cams:
                        c.best = 0.0
            else:
                now = time.time()
                if now - last_print > 0.5:
                    last_print = now
                    parts = []
                    for c in got:
                        bars = int(min(1.0, c.sharp / 400.0) * 18)
                        parts.append('cam%d%s %4.0f(최고%4.0f)|%-18s|얼굴%d'
                                     % (c.index, role_of(c.index), c.sharp,
                                        c.best, '#' * bars, len(c.faces)))
                    tilt = tilt_hint(got[0].gray, cv2, np)
                    tail = ('기울기 %+.1f도' % tilt) if tilt is not None else ''
                    sys.stdout.write('\r' + '  '.join(parts) + '  ' + tail + '   ')
                    sys.stdout.flush()

            if args.seconds and time.time() - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for c in cams:
            c.release()
        if gui:
            cv2.destroyAllWindows()
        print()
        for c in cams:
            print('cam%d%s 최고 선명도: %.0f' % (c.index, role_of(c.index), c.best))
        print('  참고: 200 이상이면 양호, 400 이상이면 아주 선명합니다.')
        print('  렌즈 배럴을 조금씩 돌리면서 이 숫자가 가장 커지는 곳에서 멈추세요.')
        if stopped:
            print('voice_chat 을 다시 켭니다...')
            service('start')
    return 0


if __name__ == '__main__':
    sys.exit(main())
