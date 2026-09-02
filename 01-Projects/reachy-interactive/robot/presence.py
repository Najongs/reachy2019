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

        self._running = None
        self._thread = None

    def start(self):
        """Begin watching in the background (no-op if the cascade is missing)."""
        from threading import Event, Thread

        if self._cascade is None or self._cascade.empty():
            logger.warning('Face cascade not found - presence watcher disabled')
            return self

        self._running = Event()
        self._running.set()
        self._thread = Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()
        logger.info('Presence watcher started (every %.1fs)', self.interval)
        return self

    def stop(self):
        if self._running is not None and self._running.is_set():
            self._running.clear()
            if self._thread is not None:
                self._thread.join(timeout=2)

    def seen_within(self, seconds):
        """True if a face was detected within the last `seconds`."""
        return self.person and (time.time() - self.last_seen) <= seconds

    def _detect(self, frame):
        """Return (cx, cy) of the largest face in [0,1], or None."""
        cv = self._cv
        h, w = frame.shape[:2]
        if w > self.detect_width:
            scale = self.detect_width / float(w)
            frame = cv.resize(frame, (int(w * scale), int(h * scale)))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None

        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        H, W = gray.shape[:2]
        return (x + fw / 2.0) / W, (y + fh / 2.0) / H

    def _loop(self):
        while self._running.is_set():
            t0 = time.time()
            try:
                frame = self.grab()
                if frame is not None:
                    res = self._detect(frame)
                    if res is not None:
                        cx, cy = res
                        self.face_error = (cx - 0.5) * 2.0
                        self.face_yerr = (cy - 0.5) * 2.0
                        if not self.person:
                            logger.info('Person detected')
                        self.person = True
                        self.last_seen = time.time()
                    elif self.person and time.time() - self.last_seen > self.forget:
                        logger.info('Person gone')
                        self.person = False
            except Exception:
                logger.debug('Presence detection failed', exc_info=True)

            wait = self.interval - (time.time() - t0)
            if wait > 0:
                self._running.wait(wait)
