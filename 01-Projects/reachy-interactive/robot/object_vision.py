"""Look at an object with the camera: detect it, then turn the neck toward it.

The head camera + the Orbita neck let the robot visually attend to whatever
someone shows it - the first step toward vision-guided grasping. A MobileNet-SSD
(Caffe, 20 VOC classes) runs locally on the Pi (~0.6s/frame, no network), finds
the most relevant object, and this servos the neck to center it, iterating a
couple of times so it actually converges. Fully offline.

Model files (not in git, ~23MB): mnssd.caffemodel + mnssd.prototxt next to this
script. Install via ops/install_object_vision.sh.
"""

import logging
import os
import time

logger = logging.getLogger('reachy.objectvision')

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
# Korean names for what it looked at (spoken back).
KO = {'bottle': '병', 'person': '사람', 'chair': '의자', 'diningtable': '탁자',
      'sofa': '소파', 'tvmonitor': '모니터', 'pottedplant': '화분', 'cat': '고양이',
      'dog': '강아지', 'car': '자동차', 'bicycle': '자전거', 'bird': '새'}
# Small objects a person is likely holding out - prefer these to attend to.
HELD = {'bottle'}
FURNITURE = {'diningtable', 'sofa', 'chair', 'tvmonitor', 'pottedplant'}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ObjectVision(object):
    """MobileNet-SSD object detector (lazy-loaded, offline)."""

    def __init__(self, proto, model, conf=0.35):
        self._proto = os.path.expanduser(proto)
        self._model = os.path.expanduser(model)
        self.conf = conf
        self._net = None

    def available(self):
        return os.path.exists(self._proto) and os.path.exists(self._model)

    def _load(self):
        if self._net is None:
            import cv2
            self._cv = cv2
            self._net = cv2.dnn.readNetFromCaffe(self._proto, self._model)
            logger.info('Object detector loaded')
        return self._net

    def detect(self, frame):
        """Return detections [{label, conf, cx, cy, area}] with cx/cy in [0,1]."""
        cv = __import__('cv2')
        net = self._load()
        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)),
                                    0.007843, (300, 300), 127.5)
        net.setInput(blob)
        det = net.forward()
        out = []
        for i in range(det.shape[2]):
            c = float(det[0, 0, i, 2])
            if c < self.conf:
                continue
            idx = int(det[0, 0, i, 1])
            x1, y1, x2, y2 = det[0, 0, i, 3:7]
            out.append({
                'label': VOC_CLASSES[idx] if idx < len(VOC_CLASSES) else '?',
                'conf': c,
                'cx': float((x1 + x2) / 2), 'cy': float((y1 + y2) / 2),
                'area': float(abs((x2 - x1) * (y2 - y1))),
                # 상자 자체도 넘긴다(비율 좌표). 얼굴 상자가 정말 이 사람 위에
                # 있는지 확인하는 데 쓴다 - 안 그러면 문틀을 얼굴로 잡은 크롭이
                # 사람 사진에 섞인다.
                'x1': float(x1), 'y1': float(y1),
                'x2': float(x2), 'y2': float(y2),
            })
        return out

    def pick(self, dets):
        """Choose the object to look at: held items > person > other > furniture."""
        if not dets:
            return None

        def score(d):
            s = d['conf']
            if d['label'] in HELD:
                s += 1.0
            elif d['label'] == 'person':
                s += 0.3
            elif d['label'] in FURNITURE:
                s -= 0.5
            return s

        return max(dets, key=score)


def look_at_object(reachy, grab, vision, sign=1.0, gain_y=0.42, gain_z=0.34,
                   iters=3, freq=50, seg=0.7):
    """Capture -> detect -> turn the neck toward the object; iterate to center.

    Returns the detection it settled on (dict), or None if nothing was found.
    `grab` is a callable returning a BGR frame; `sign` is IdleMotion.ATTEND_SIGN.
    """
    from say_and_move import point_head
    import numpy as np

    head = reachy.head
    try:
        head.compliant = False
        for m in head.motors:
            m.compliant = False
    except Exception:
        pass
    time.sleep(0.05)

    found = None
    for _ in range(iters):
        frame = grab()
        if frame is None:
            break
        target = vision.pick(vision.detect(frame))
        if target is None:
            break
        found = target

        ey = (target['cx'] - 0.5) * 2.0     # + = object on image right
        ez = (target['cy'] - 0.5) * 2.0     # + = object low in image
        y0, z0 = getattr(head, '_soft_gaze', (0.0, 0.0))
        ty = _clamp(y0 - sign * ey * gain_y, -0.5, 0.5)
        tz = _clamp(z0 - ez * gain_z, -0.35, 0.42)

        t0 = time.time()
        while True:
            t = (time.time() - t0) / seg
            if t >= 1:
                break
            k = 0.5 - 0.5 * np.cos(np.pi * t)
            point_head(reachy, 0.5, y0 + (ty - y0) * k, z0 + (tz - z0) * k,
                       tilt=False)
            time.sleep(1.0 / freq)
        point_head(reachy, 0.5, ty, tz, tilt=False)

        if abs(ey) < 0.12 and abs(ez) < 0.12:
            break                            # centered - done
        time.sleep(0.15)

    return found


def describe(det):
    """Short Korean line for what it looked at."""
    if det is None:
        return '음, 잘 안 보이네요. 좀 더 가까이 보여주실래요?'
    name = KO.get(det['label'], '그거')
    return '아, {} 보이네요!'.format(name)
