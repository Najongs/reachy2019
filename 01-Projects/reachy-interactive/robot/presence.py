"""Local, always-on people awareness from the head camera - no LLM, no cost.

A background thread grabs a frame every second or so and runs OpenCV's Haar
face detector (bundled with opencv-python, already on the Pi). It exposes:

  * person      - True while a face has been seen recently
  * last_seen   - timestamp of the last detection
  * face_error  - horizontal offset of the largest face in [-1, 1]
                  (-1 = far left of the image, +1 = far right)
  * face_yerr   - vertical offset of that face in [-1, 1] (-1 = top)

The conversation loop reads `person` so the robot knows it is talking to
someone, and IdleMotion reads `face_error` / `face_yerr` to gently turn the
head toward whoever is in front of it (opt-in via --attend).

Detection runs on a downscaled grayscale frame (~30-80 ms on a Pi 4), so it is
cheap enough to run continuously without disturbing the 50 Hz motion loops.
"""

import logging
import os
import time

logger = logging.getLogger('reachy.presence')

# 얼굴 없이 '움직임'만으로 촬영할 조건 (실측 기준값).
# 화면 변화 넓이가 이 사이에 있을 때만 사람으로 본다. 너무 작으면 센서 잡음,
# 너무 크면 조명 변화나 미처 못 거른 카메라 흔들림이다.
MOTION_MAX = 0.60      # 이보다 크게 변하면 조명 변화나 카메라 흔들림으로 본다
HEAD_SETTLE = 0.8      # 목이 멈춘 뒤 이만큼 지나야 화면을 믿는다(초)
MOTION_RECENT = 4.0    # 최근 이 시간 안에 움직임이 있었으면 아직 사람이 있다고 본다
DIAG_EVERY = 30.0      # 수집 조건이 왜 막혔는지 이따금 남긴다(튜닝용)


class PresenceWatcher(object):
    """Watch the camera for faces on a background thread.

    Args:
        grab (callable): grab() -> BGR ndarray (one frame) or None
        interval (float): seconds between detections
        forget (float): drop `person` this many seconds after the last face
        detect_width (int): downscale frames to this width before detecting
    """

    def __init__(self, grab, interval=1.2, forget=6.0, detect_width=320):
        import cv2

        self._cv = cv2
        self.grab = grab
        self.interval = interval
        self.forget = forget
        self.detect_width = detect_width

        path = os.path.join(cv2.data.haarcascades,
                            'haarcascade_frontalface_default.xml')
        self._cascade = cv2.CascadeClassifier(path)

        self.person = False
        self.last_seen = 0.0
        self.face_error = 0.0
        self.face_yerr = 0.0
        # Appearance bookkeeping for proactive behaviors (hallway greeting).
        self.appeared_at = 0.0   # when person last flipped False -> True
        self.left_at = 0.0       # when person last flipped True -> False

        # Scene-motion detection (frame differencing): notices passers-by who
        # never face the camera. motion_x is the horizontal centroid of the
        # change in [-1, 1] (-1 = left edge of the image).
        self.motion_level = 0.0
        self.motion_x = 0.0
        self.motion_at = 0.0
        # 변한 화면의 '넓이' 비율. 카메라가 움직이는 머리에 달려 있어서, 머리가
        # 돌면 화면 전체가 변한다(≈1.0). 사람 하나가 지나가는 것은 화면의 일부만
        # 바꾼다. 이 값으로 둘을 가른다 - 없으면 빈 복도 사진만 잔뜩 쌓인다.
        self.motion_area = 0.0
        self._prev_gray = None
        # 마지막으로 검출한 얼굴 박스(비율 좌표)와, 사람이 보일 때마다 호출할 콜백.
        # 콜백은 person_db 로 사진을 모으는 데 쓴다 (voice_chat 에서 연결).
        self.face_box_rel = None
        self.on_person = None
        # 머리가 멈춰 있는지 알려 주는 함수(say_and_move.head_still_for).
        # 없으면 얼굴이 보일 때만 촬영한다 - 오탐보다 놓치는 편이 낫다.
        self.head_still = None
        self._diag_at = 0.0

        self._stop = None
        self._thread = None

    def start(self):
        """Begin watching in the background (no-op if the cascade is missing)."""
        from threading import Event, Thread

        if self._cascade is None or self._cascade.empty():
            logger.warning('Face cascade not found - presence watcher disabled')
            return self

        self._stop = Event()
        self._thread = Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()
        logger.info('Presence watcher started (every %.1fs)', self.interval)
        return self

    def stop(self):
        if self._stop is not None and not self._stop.is_set():
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2)

    def seen_within(self, seconds):
        """True if a face was detected within the last `seconds`."""
        return self.person and (time.time() - self.last_seen) <= seconds

    def _to_gray(self, frame):
        """Downscale + grayscale a BGR frame for detection/differencing."""
        cv = self._cv
        h, w = frame.shape[:2]
        if w > self.detect_width:
            scale = self.detect_width / float(w)
            frame = cv.resize(frame, (int(w * scale), int(h * scale)))
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def _detect(self, frame):
        """Return (cx, cy) of the largest face in [0,1], or None."""
        gray = self._cv.equalizeHist(self._to_gray(frame))
        return self._detect_gray(gray)

    def _detect_gray(self, gray):
        faces = self._cascade.detectMultiScale(
            self._cv.equalizeHist(gray),
            scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None

        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        H, W = gray.shape[:2]
        # 검출은 축소본에서 하므로, 원본 프레임 좌표로 되돌려 기억해 둔다
        # (사람 데이터셋이 얼굴만 잘라 저장할 때 쓴다).
        self.face_box_rel = (x / float(W), y / float(H),
                             fw / float(W), fh / float(H))
        return (x + fw / 2.0) / W, (y + fh / 2.0) / H

    def _update_motion(self, gray):
        """Frame differencing: how much the scene changed, and where.

        카메라가 머리에 달려 있으므로, 목이 움직인 사이에 찍힌 두 프레임을 빼면
        장면이 바뀐 게 아니라 카메라가 움직인 것이다. 그걸 '움직임'으로 세면
        로봇은 자기가 고개를 돌린 것을 보고 사람이 지나갔다고 착각한다.
        그래서 목이 멎어 있는 동안 연속으로 찍힌 두 프레임만 비교한다.
        """
        cv = self._cv
        still = self.head_still() > HEAD_SETTLE if self.head_still else True
        prev = self._prev_gray
        # 목이 움직였으면 직전 프레임은 기준으로 못 쓴다. 버리고 다시 모은다.
        self._prev_gray = gray if still else None
        if prev is None or not still or prev.shape != gray.shape:
            return

        diff = cv.absdiff(gray, prev)
        level = float(diff.mean()) / 255.0
        self.motion_level = level
        self.motion_area = float((diff > 25).mean())
        if level < 0.02:            # sensor noise floor
            return

        cols = diff.sum(axis=0).astype('float64')
        total = cols.sum()
        if total > 0:
            import numpy as np
            cx = float((cols * np.arange(len(cols))).sum() / total) / len(cols)
            self.motion_x = (cx - 0.5) * 2.0
        self.motion_at = time.time()

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                frame = self.grab()
                if frame is None:
                    # 프레임이 안 오면 얼굴 검출도 수집도 전부 멈춘 것이다.
                    # 조용히 지나가면 안 되므로 이따금 알린다.
                    if time.time() - self._diag_at > DIAG_EVERY:
                        self._diag_at = time.time()
                        logger.warning('presence: 카메라 프레임을 못 받고 있습니다')
                if frame is not None:
                    gray = self._to_gray(frame)
                    self._update_motion(gray)
                    res = self._detect_gray(gray)
                    now = time.time()
                    if res is not None:
                        cx, cy = res
                        self.face_error = (cx - 0.5) * 2.0
                        self.face_yerr = (cy - 0.5) * 2.0
                        if not self.person:
                            logger.info('Person detected')
                            self.appeared_at = now
                        self.person = True
                        self.last_seen = now
                        # 사람이 보이는 동안 계속 알려 준다. 실제로 저장할지는
                        # 받는 쪽(person_db)이 간격·장수 제한으로 결정한다.
                        if self.on_person is not None:
                            try:
                                self.on_person(frame, self.face_box_rel)
                            except Exception:
                                logger.debug('on_person failed', exc_info=True)
                    else:
                        # 얼굴은 안 보이지만 화면이 움직인다 - 옆모습이나 뒷모습으로
                        # 지나가는 사람일 수 있다. 복도에서는 이쪽이 더 흔하다.
                        # 단, 머리가 도는 중이면 화면 전체가 흐르므로 믿지 않는다.
                        # '지금 움직이는가'가 아니라 '최근에 움직였는가'로 본다.
                        # 사람이 지나가면 idle 모션이 그쪽으로 고개를 돌리는데,
                        # 그 동안은 화면을 믿을 수 없다. 고개가 멎은 직후를 노려
                        # 찍어야 그 사람을 놓치지 않는다. 멈춰 선 사람도 잡힌다.
                        if (self.on_person is not None
                                and self.head_still is not None
                                and self.motion_area < MOTION_MAX
                                and now - self.motion_at < MOTION_RECENT
                                and self.head_still() > HEAD_SETTLE):
                            try:
                                self.on_person(frame, None)
                            except Exception:
                                logger.debug('on_person(motion) failed',
                                             exc_info=True)
                        elif (self.on_person is not None
                                and now - self._diag_at > DIAG_EVERY):
                            # 왜 안 찍혔는지 남겨 둔다. 이게 없으면 '조용해서'
                            # 안 찍힌 건지 '조건이 계속 막혀서'인지 알 수 없다.
                            self._diag_at = now
                            since = now - self.motion_at if self.motion_at else -1
                            still = self.head_still() if self.head_still else -1
                            logger.info(
                                '수집 대기: 변화넓이 %.3f · 마지막 움직임 %s · '
                                '목 멈춘 지 %.1fs',
                                self.motion_area,
                                ('%.1fs 전' % since) if since >= 0 else '없음',
                                min(still, 999.0))

                    if self.person and now - self.last_seen > self.forget:
                        logger.info('Person gone')
                        self.person = False
                        self.left_at = now
            except Exception:
                logger.exception('Presence detection failed')

            wait = self.interval - (time.time() - t0)
            if wait > 0:
                self._stop.wait(wait)
