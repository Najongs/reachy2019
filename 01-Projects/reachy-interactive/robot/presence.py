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
        self._prev_gray = None

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
        return (x + fw / 2.0) / W, (y + fh / 2.0) / H

    def _update_motion(self, gray):
        """Frame differencing: how much the scene changed, and where."""
        cv = self._cv
        prev = self._prev_gray
        self._prev_gray = gray
        if prev is None or prev.shape != gray.shape:
            return

        diff = cv.absdiff(gray, prev)
        level = float(diff.mean()) / 255.0
        self.motion_level = level
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
                    elif self.person and now - self.last_seen > self.forget:
                        logger.info('Person gone')
                        self.person = False
                        self.left_at = now
            except Exception:
                logger.debug('Presence detection failed', exc_info=True)

            wait = self.interval - (time.time() - t0)
            if wait > 0:
                self._stop.wait(wait)
