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
MOTION_MIN = 0.03      # 이보다 작으면 센서 잡음. 사람이면 화면의 몇 %는 바뀐다
MOTION_MAX = 0.30      # 이보다 크면 사람이 아니라 화면 전체가 흔들린 것이다
HEAD_SETTLE = 0.8      # 목이 멈춘 뒤 이만큼 지나야 화면을 믿는다(초)
HEAD_MOVE_TOL = 0.3    # 목 디스크가 이보다 움직였으면 '움직인 것'(도).
                       # 실측: 정지 상태로 20회 읽었을 때 변동폭 최대 0.081도.
MOTION_RECENT = 4.0    # 최근 이 시간 안에 움직임이 있었으면 아직 사람이 있다고 본다
DIAG_EVERY = 30.0      # 수집 조건이 왜 막혔는지 이따금 남긴다(튜닝용)

# 얼굴로 인정하려면 이웃 검출이 몇 개나 겹쳐야 하는지.
#
# 기본값 5 는 이 복도에서 오검출이 잦다. 실측:
#   - 빈 복도 26장 중 5장에서 유리문 틀을 얼굴로 잡았고, 그 사진들이 그대로
#     데이터셋에 들어갔다. 7 이상에서는 그 26장에서 오검출 0.
#   - 예전 복도 프레임 24장에서는 바닥 표식을 얼굴로 잡았다(8 에서도 남고,
#     9 에서 사라진다).
# 오검출은 데이터셋을 더럽힐 뿐 아니라 로봇이 문을 보고 인사하게 만든다.
#
# 주의: 진짜 얼굴에 대한 재현율은 아직 검증하지 못했다. 가진 표본(위 50장)에
# 사람 얼굴이 한 장도 없었기 때문이다. 그래서 오검출을 줄이되 너무 올리지는
# 않는 선인 8 로 둔다. 사람 앞에서 확인한 뒤 조정할 것.
FACE_MIN_NEIGHBORS = 8

# 사람 검출(MobileNet-SSD)을 얼마나 자주 돌릴지(초). Pi 4 에서 한 번에 ~350ms 라
# 매 폴링마다 돌리면 코어 하나를 절반쯤 먹는다.
PERSON_NET_INTERVAL = 1.6
PERSON_NET_CONF = 0.5


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
        # 목 디스크의 '실측' 위치를 돌려주는 함수. 명령을 어디서 넣었는지
        # 쫓아다니는 대신, 카메라가 실제로 움직였는지를 본다. 명령 경로를
        # 계측하는 방식은 한 군데만 놓쳐도 조용히 틀린다(실제로 그랬다).
        # 없으면 얼굴이 보일 때만 촬영한다 - 오탐보다 놓치는 편이 낫다.
        # 사람 검출기(MobileNet-SSD). 복도에서는 얼굴 검출보다 훨씬 잘 맞는다.
        # 실측: 크게 찍힌 정면 얼굴을 Haar 는 어떤 설정으로도 못 잡았는데
        # (역광·안경·화면 기울기 13도), SSD 는 같은 사진을 0.98 로 잡았다.
        self.detect_person = None
        self._net_at = 0.0
        self.person_conf = 0.0
        self.head_pose = None
        self._pose_prev = None
        self._head_moved_at = 0.0
        self._still = False
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

    def _update_head_state(self):
        """목이 실제로 움직였는지 갱신한다 (디스크 실측 위치 비교)."""
        if self.head_pose is None:
            self._still = False        # 알 수 없으면 화면을 믿지 않는다
            return
        try:
            pose = self.head_pose()
        except Exception:
            self._still = False
            return
        now = time.time()
        if self._pose_prev is not None and any(
                abs(a - b) > HEAD_MOVE_TOL for a, b in zip(pose, self._pose_prev)):
            self._head_moved_at = now
        self._pose_prev = pose
        self._still = (now - self._head_moved_at) > HEAD_SETTLE

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
            scaleFactor=1.2, minNeighbors=FACE_MIN_NEIGHBORS, minSize=(30, 30))
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
        still = self._still
        prev = self._prev_gray
        # 목이 움직였으면 직전 프레임은 기준으로 못 쓴다. 버리고 다시 모은다.
        self._prev_gray = gray if still else None
        if prev is None or not still or prev.shape != gray.shape:
            return

        import numpy as np

        # 복도 끝이 유리문이라 역광이 세고, 카메라가 자동노출을 계속 조정한다.
        # 그러면 화면 전체 밝기가 통째로 출렁여 프레임 차이가 크게 나온다
        # (실측 0.52~0.67 - 사람이 지나갈 때 0.12~0.18 보다 훨씬 크다).
        # 밝기가 통째로 바뀐 만큼(중앙값)을 빼서, 국소적으로 변한 곳만 남긴다.
        d = gray.astype(np.int16) - prev.astype(np.int16)
        d -= int(np.median(d))
        diff = np.abs(d).astype('uint8')

        level = float(diff.mean()) / 255.0
        self.motion_level = level
        self.motion_area = float((diff > 25).mean())
        if level < 0.02:            # sensor noise floor
            return

        cols = diff.sum(axis=0).astype('float64')
        total = cols.sum()
        if total > 0:
            cx = float((cols * np.arange(len(cols))).sum() / total) / len(cols)
            self.motion_x = (cx - 0.5) * 2.0
        self.motion_at = time.time()

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._update_head_state()
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

                    # 얼굴이 안 잡혀도 사람일 수 있다. 무거운 검출기라 간격을 둔다.
                    body = False
                    if (res is None and self.detect_person is not None
                            and now - self._net_at > PERSON_NET_INTERVAL):
                        self._net_at = now
                        try:
                            conf = self.detect_person(frame)
                        except Exception:
                            conf = 0.0
                            logger.debug('사람 검출 실패', exc_info=True)
                        self.person_conf = conf
                        body = conf >= PERSON_NET_CONF

                    if body:
                        if not self.person:
                            logger.info('사람 검출 (몸 %.2f)', self.person_conf)
                            self.appeared_at = now
                        self.person = True
                        self.last_seen = now
                        self.face_box_rel = None
                        if self.on_person is not None:
                            try:
                                self.on_person(frame, None)
                            except Exception:
                                logger.debug('on_person(body) failed', exc_info=True)
                    elif res is not None:
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
                                and self.head_pose is not None
                                and MOTION_MIN < self.motion_area < MOTION_MAX
                                and now - self.motion_at < MOTION_RECENT
                                and self._still):
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
                            logger.info(
                                '수집 대기: 변화넓이 %.3f · 마지막 움직임 %s · '
                                '목 %s',
                                self.motion_area,
                                ('%.1fs 전' % since) if since >= 0 else '없음',
                                '정지' if self._still else '움직이는 중')

                    if self.person and now - self.last_seen > self.forget:
                        logger.info('Person gone')
                        self.person = False
                        self.left_at = now
            except Exception:
                logger.exception('Presence detection failed')

            wait = self.interval - (time.time() - t0)
            if wait > 0:
                self._stop.wait(wait)
