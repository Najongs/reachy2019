"""Voice conversation loop: mic -> STT -> LLM -> speech + head motion.

Runs on the Pi. Listens on the microphone, turns Korean speech into text
(Google Web Speech via the SpeechRecognition library), asks the LLM broker,
then answers out loud while moving the head. While the model is working the
robot plays its "thinking" motion instead of freezing.

Pi setup:
    sudo apt install -y python3-pyaudio flac
    pip3 install SpeechRecognition
    arecord -l          # a USB microphone must show up here

Usage:
    python3 voice_chat.py --list-mics
    python3 voice_chat.py --url http://<windows-ip>:8080 --token <secret>
    python3 voice_chat.py --no-robot --url ...        # STT + text reply only
    python3 voice_chat.py --stt-test                  # STT alone, no LLM

Say "그만" or "종료" to stop the loop.
"""

import argparse
import logging
import json
import os
import time

from contextlib import contextmanager


logger = logging.getLogger(__name__)


@contextmanager
def quiet_stderr():
    """Silence C-level stderr (ALSA/JACK config spam) during audio setup.

    PortAudio probes every ALSA device listed in the system config and prints
    pages of harmless 'Unknown PCM ...' noise. Python's own stderr is restored
    afterwards, so real errors still show.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


STOP_WORDS = ('그만', '종료', '잘가', '잘 가')

# When the reply arrives while a pondering hum is playing, let the hum run to
# its natural end (up to this long) instead of chopping it mid-word, then take
# a short breath before answering - that is what makes the hand-off feel human.
FILLER_TAIL_MAX = 1.2
FILLER_BREATH = 0.18

# When the sentence sounds like it is about what the robot can SEE,
# a camera frame is attached to the request.
# '누구'/'누가' alone are too broad ("누구세요"=identity, "누가 만들었어"=maker):
# only the clearly camera-flavored forms count as vision.
VISION_WORDS = ('보여', '보이', '뭐가 있', '이게 뭐', '이거 뭐', '저게 뭐', '저거 뭐',
                '누구 있', '누가 있', '누가 왔', '사람 누구',
                '몇 명이야', '몇명이야', '몇 명 있', '몇명 있',
                '무슨 색', '무슨색', '어떻게 생겼',
                '읽어봐', '읽어줘', '읽을 수', '앞에 뭐', '앞에뭐',
                '놓여', '놓여 있', '뭐 놓', '뭐가 보',
                '입었게', '입고 있게', '들었게', '보고 있')


# Motion routing: a body part must co-occur with an action verb, or a gesture
# noun with a command-ish tail. "팔 아파" has the noun but no verb -> chat.
BODY_WORDS = ('팔', '손', '고개', '안테나', '머리', '몸',
              # 어깨가 빠져 있어서 "왼쪽 어깨를 돌려 줘" 가 잡담으로 새고,
              # 대화 모델이 움직인 척 답했다 (실측 2026-09-08 아침).
              '어깨', '팔꿈치', '손목', '허리',
              '물건', '이거', '저거', '그거', '컵', '공', '펜', '상자', '보관함', '트레이')
# Single-syllable stems ('드','놓','숙','젖') matched inside ordinary words
# (만드는/놓고 가요/몸살/젖었어) - only multi-syllable, verb-shaped stems now.
ACTION_STEMS = ('들어', '들고', '들어서', '올려', '올리', '내려', '내리', '흔들',
                '움직', '벌려', '벌리', '접어', '돌려', '돌리', '가리켜', '가리키',
                '집어', '집게', '놓아', '놓고', '옮겨', '건네', '넣어', '꺼내',
                '굴려', '빙글', '잡아',
                '펴', '뻗', '세워', '펼쳐', '펼', '휘둘', '휘저', '뻗쳐',
                '틀어', '틀', '젖혀', '숙여', '까딱', '꺾',
                # 정렬 요청 ("어깨를 일직선으로/수평으로 맞춰") - 신체 단어와
                # 함께일 때만 동작이 되므로 과잉 발동 위험이 낮다
                '일직선', '수평', '정렬', '맞춰', '맞추', '반듯')
# Phrases that LOOK like body+verb but are everyday hallway talk, never a
# command to the robot ("먼저 들어가볼게요", "그거 드셨어요", "몸살 나서").
NOT_MOTION = ('들어가', '들어오', '들어온', '들어간', '들어갈', '드셨', '드세요',
              '드실', '만드는', '몸살', '팔레트', '학수고대', '컵라면', '놓고가')
# Gesture nouns strong enough to trigger on their own (no verb needed) - a
# bare "만세" or "만세 다시" is unambiguously a motion request.
# In the hallway, a bare overheard fragment ("박수", clapping talk nearby)
# must not trigger a gesture - 박수/손뼉 need a command tail now.
GESTURE_WORDS = ('인사', '만세', '하이파이브', '브이', '악수')
GESTURE_WEAK = ('춤', '환영', '포즈', '박수', '손뼉', '가리켜', '가리키')
# Compounds that contain a gesture noun but are not motion requests
# (office talk: 인사이동/인사팀, 티브이/브이로그 contain 브이, etc).
GESTURE_STOPWORDS = ('인사말', '인사법', '인사드', '박수갈채', '만세력',
                     '인사이동', '인사팀', '인사평가', '인사과',
                     '티브이', '브이로그', '만세운동', '하는것', '하는 것',
                     '무슨뜻', '뜻이야', '뜻이', '그만', '시끄', '멈춰', '멈추')
# Bare '자' matched inside 자연스럽다/자리 - removed. For GESTURE_WEAK the
# tail must also come AFTER the gesture word, close by (see wants_motion).
COMMAND_TAILS = ('해봐', '해 봐', '해줘', '해 줘', '하자', '해라', '춰봐', '쳐봐',
                 '쳐줘', '해보', '볼래', '할래', '수있', '다시', '또', '봐', '줘',
                 '취해', '취하', '지어')


# 직전 턴이 동작일 때, 문맥상 이전 동작을 가리키는 후속 발화 (대명사·수식어)
FOLLOWUP_WORDS = ('다시', '또', '한번더', '한 번 더', '더크게', '더 크게', '더작게',
                  '반대', '천천히', '처음부터', '계속', '아까처럼', '방금', '그거',
                  '이번엔', '한번', '해볼래')
# ...but a question or a comment about the motion is NOT a repeat request:
# "방금 뭐 한 거야?", "그거 진짜 신기하다", "천천히 말해줘".
FOLLOWUP_BLOCK = ('뭐', '왜', '어떻', '누구', '말', '신기', '대박', '우와', '멋')


# Head/neck movement. The neck is a motorised Orbita mechanism (gaze-driven,
# not keyframe joints), so these run LOCALLY via run_head_gesture - offline,
# no broker. A head word + a direction/action maps to a gesture.
HEAD_WORDS = ('고개', '목', '머리', '얼굴')
# Unambiguous head gestures - trigger even without a 고개/목 word ('도리도리
# 해봐'). '까딱' is NOT here (it collides with wrist gestures).
_HEAD_STRONG = [
    (('끄덕끄덕', '끄덕여', '끄덕', '네네'), 'nod'),
    (('도리도리', '절레절레', '절레'), 'shake'),
]
# With a head word, a direction/action cue. Directions come BEFORE 'roll' so
# "왼쪽으로 돌려" = look left, while a bare "돌려/한바퀴" = full roll.
_HEAD_DIR = [
    (('끄덕', '까딱'), 'nod'),
    (('도리도리', '절레', '좌우로', '흔들'), 'shake'),
    (('위로', '위', '올려', '쳐들'), 'up'),
    (('아래', '밑', '숙여', '숙이', '내려'), 'down'),
    (('왼쪽', '왼', '좌측', '좌로'), 'left'),
    (('오른쪽', '오른', '우측', '우로'), 'right'),
    (('젖혀', '젖히', '뒤로'), 'tilt_back'),
    (('돌려', '돌리', '한바퀴', '빙', '돌아'), 'roll'),
    (('정면', '앞', '가운데', '중앙', '똑바로', '원위치', '제자리'), 'center'),
]


# "Look at that / the object" - point the camera+neck at a detected object.
_LOOK_OBJ = ('저거 봐', '저것 봐', '이거 봐', '이것 봐', '그거 봐', '물건 봐',
             '물체 봐', '저기 봐', '저 물건', '저 물체', '이 물건', '이 물체',
             '물건을 봐', '물체를 봐', '저거 찾', '물건 찾', '물체 찾', '뭐 있나 봐',
             '저거 뭐', '이거 집', '그거 집', '물건 집', '저거 잡', '물건 잡',
             # "바닥에 있는 물건 확인해 봐", "카메라로 물체 확인" 류
             '물건 확인', '물체 확인', '뭐가 있는지 확인', '확인해 봐', '확인해봐',
             '바닥에 있는', '앞에 있는 물', '카메라로')


# STT junk seen repeatedly in the logs. Vosk (used whenever Google STT is
# unreachable) turns corridor noise and half-heard words into confident-looking
# but meaningless text - "야 고사양" appeared 8 times in one day, and each one
# cost an LLM call and a puzzled reply. These are LOUD, so an audio-level
# threshold cannot catch them; they have to be filtered as text.
_STT_JUNK = {
    '야고사양', 'a고사양', '야록고사양', '호소야', '호소야아', '호소야야',
    '아호소야햐', '하얀자', '하얀', '이다', '노는', '보스', '음', '으음',
    '어', '아', '그', '저', '네네네',
}


_SHORT_REAL = {'안녕', '하이', '만세', '인사', '악수', '박수', '고마워', '미안',
               '누구', '이름', '뭐야', '왜', '응', '네', '노래', '춤'}


def looks_like_noise(text, engine=None):
    """True for transcripts that are almost certainly mis-heard noise.

    Only consulted at the very END of routing: anything meaningful (a command,
    a note, a real question) has already matched by then, so filtering here
    cannot swallow a real request. Keeps the robot from answering the corridor.
    """
    compact = (text or '').replace(' ', '')
    if not compact:
        return True
    if compact.lower() in _STT_JUNK:
        return True
    # A one- or two-syllable leftover that matched nothing is noise - EXCEPT
    # for real short words people actually say to a robot.
    if compact in _SHORT_REAL:
        return False
    if len(compact) <= 2:
        return True
    # Offline Vosk is much noisier than Google; be stricter with its output.
    if engine == 'vosk' and len(compact) <= 3 and not any(
            c in compact for c in ('안녕', '고마', '미안', '누구', '이름')):
        return True
    return False


def wants_object_look(text):
    """True if the user wants the robot to look at / find an object."""
    compact = text.replace(' ', '')
    return any(w.replace(' ', '') in compact for w in _LOOK_OBJ)


def wants_head_gesture(text):
    """Return a head-gesture name for a neck/head movement request, else None."""
    compact = text.replace(' ', '')
    for cues, name in _HEAD_STRONG:
        if any(c in compact for c in cues):
            return name
    # '목' must not be the '목' inside '손목'(wrist) - that is an arm gesture.
    has_head = ('고개' in compact or '머리' in compact or '얼굴' in compact
                or ('목' in compact and '손목' not in compact))
    if not has_head:
        return None
    # Needs a movement cue, not just a mention ("머리 아파" -> chat).
    for cues, name in _HEAD_DIR:
        if any(c in compact for c in cues):
            return name
    return None


def wants_motion(text, last_kind=None):
    """Check if the request asks for a physical gesture.

    last_kind='motion' 이면, 이전 동작을 가리키는 짧은 후속 발화도 동작으로 본다.
    """
    compact = text.replace(' ', '')

    # Everyday phrases that merely CONTAIN body/verb fragments never route
    # to motion, whatever else matches.
    if any(w in compact for w in NOT_MOTION):
        return False

    if any(b in compact for b in BODY_WORDS) and any(a in compact for a in ACTION_STEMS):
        return True
    # Strong gesture nouns trigger alone; weak ones still need a command tail
    # positioned shortly AFTER the gesture word ("포즈 취해봐" yes,
    # "포즈가 아주 자연스럽네요" / "환영회 언제인지 알아봐줘" no).
    if any(g in compact for g in GESTURE_WORDS) \
            and not any(w.replace(' ', '') in compact for w in GESTURE_STOPWORDS):
        return True
    for g in GESTURE_WEAK:
        gi = compact.find(g)
        if gi < 0:
            continue
        after = compact[gi + len(g):gi + len(g) + 8]
        if any(t.replace(' ', '') in after for t in COMMAND_TAILS):
            return True
    # 직전이 동작이면, 문맥상 그 동작을 이어가는 짧은 후속 요청도 동작으로
    # (단, 동작에 대한 질문/감상은 제외)
    if last_kind == 'motion' and len(compact) <= 14:
        if any(w.replace(' ', '') in compact for w in FOLLOWUP_WORDS) \
                and not any(b in compact for b in FOLLOWUP_BLOCK):
            return True
    return False


class TurnLogger(object):
    """Append-only dataset of what happened each turn.

    One JSONL line per event under <root>/events.jsonl; attached camera
    frames go to <root>/frames/. Grows into training/analysis data over time.
    """

    def __init__(self, root):
        # Each run gets its own session directory, so runs never mix and
        # frame files can never overwrite an earlier session's.
        session = time.strftime('%Y%m%d-%H%M%S')
        self.root = os.path.join(os.path.expanduser(root), session)
        self.frames = os.path.join(self.root, 'frames')
        os.makedirs(self.frames, exist_ok=True)
        self.path = os.path.join(self.root, 'events.jsonl')
        # 지금 누가 앞에 있는지 알려 주는 함수(person_db.visit_id). 모든 이벤트에
        # 방문 id 를 붙여, 나중에 군집으로 그 방문이 '누구'인지 알아냈을 때
        # 그 사람이 무슨 말을 했는지까지 따라오게 한다. 이게 없으면 사진과
        # 대화가 끊겨, 사람을 알아봐도 아는 게 얼굴뿐이다.
        self.visit_of = None
        self._seq = 0
        self._disk_checked = 0.0
        self._disk_ok = True

    def _frames_ok(self, min_free_mb=500):
        """Stop writing frames if the SD card is running low (checked hourly)."""
        now = time.time()
        if now - self._disk_checked > 3600:
            self._disk_checked = now
            try:
                st = os.statvfs(self.root)
                free_mb = st.f_bavail * st.f_frsize / (1024 * 1024)
                self._disk_ok = free_mb > min_free_mb
                if not self._disk_ok:
                    logger.warning('Only %.0fMB free - not saving more frames',
                                   free_mb)
            except Exception:
                self._disk_ok = True
        return self._disk_ok

    def log(self, kind, data, image_b64=None):
        """Record one event; never raises."""
        import base64
        import json

        try:
            entry = {'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'kind': kind}
            if self.visit_of is not None:
                try:
                    visit = self.visit_of()
                except Exception:
                    visit = None
                if visit:
                    entry['visit_id'] = visit
            entry.update(data)

            if image_b64 and self._frames_ok():
                self._seq += 1
                name = 'frame-{:04d}-{}.jpg'.format(self._seq, kind)
                frame_path = os.path.join(self.frames, name)
                with open(frame_path, 'wb') as f:
                    f.write(base64.b64decode(image_b64))
                entry['frame'] = os.path.join('frames', name)

            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception:
            logger.exception('Turn logging failed')


class MotionHandler(object):
    """Route a motion request through the motion model and execute it safely.

    All the untrusted-JSON handling lives here: preset resolution, safety
    validation, execution, and speaking the result. Never raises.
    """

    ACK_LINES = ['좋아요, 어떻게 움직일지 생각해볼게요.', '음, 한번 해볼게요.']
    FAIL_LINE = '미안해요, 지금은 동작을 만들지 못했어요.'

    def __init__(self, reachy, client, speech, turn_logger=None):
        from motion_exec import MotionExecutor, validate, ValidationError
        from motion_presets import PRESETS, validate_all

        validate_all()   # presets are code; fail loudly at startup if broken

        self.reachy = reachy
        self.client = client
        self.speech = speech
        self.turn_logger = turn_logger
        self._last_snap = None

        self.executor = MotionExecutor(reachy)
        self.presets = PRESETS
        self._validate = validate
        self._validation_error = ValidationError

    def _snapshot(self):
        """Grab one camera frame mid-gesture (kept for the turn log)."""
        import base64

        try:
            image = capture_view(self.reachy)
            if image:
                self._last_snap = image
                with open('/tmp/motion_view.jpg', 'wb') as f:
                    f.write(base64.b64decode(image))
                logger.info('Mid-motion frame saved to /tmp/motion_view.jpg')
        except Exception:
            logger.exception('Mid-motion snapshot failed')

    def _log(self, text, motion, outcome, extra=None):
        if self.turn_logger is None:
            return
        data = {'text': text, 'motion': motion, 'outcome': outcome}
        if extra:
            data.update(extra)
        self.turn_logger.log('motion', data, image_b64=self._last_snap)
        self._last_snap = None

    def handle(self, text, image, head):
        """Full motion turn: ack -> Qwen -> validate -> speak + move."""
        import random

        from say_and_move import say_and_move

        if self.speech is not None:
            self.speech.start(text=random.choice(self.ACK_LINES))

        if head is not None:
            head.setup()
            head.start_thinking()

        motion = self.client.ask_motion(text, image=image)

        if head is not None:
            head.stop_thinking()
        if self.speech is not None:
            self.speech.wait()

        if motion is None:
            self._log(text, None, 'broker_failed')
            say_and_move(self.reachy, text=self.FAIL_LINE, speech=self.speech, head=head)
            return

        say = motion.get('say') or ''
        moves = None

        preset = motion.get('preset')
        if preset and preset in self.presets:
            moves = self.presets[preset]['moves']
            logger.info('Using preset %r', preset)
        elif motion.get('moves'):
            moves = motion['moves']
            logger.info('Generated motion: %d keyframes', len(moves))

        if moves is None:
            # Not actually a motion request (keyword false positive) -
            # the motion model answered conversationally; just speak it.
            self._log(text, motion, 'null_moves')
            if say:
                say_and_move(self.reachy, text=say, speech=self.speech, head=head)
            return

        # Hanging-arm seed for validation; the executor ramps from the real
        # pose it reads at execution time.
        seed = {j: 0.0 for j in self.executor.available_joints()}

        try:
            segments = self._validate(moves, seed, self.executor.available_joints())
        except self._validation_error as e:
            logger.warning('Motion rejected: %s', e)
            self._log(text, motion, 'rejected', {'reason': str(e)})
            say_and_move(self.reachy, text=str(e), speech=self.speech, head=head)
            return

        # Speak while moving; the executor owns the head during the motion.
        if say and self.speech is not None:
            self.speech.start(text=say)

        # Snapshot mid-gesture so we can verify the hand is really in frame.
        from threading import Timer

        total = sum(d for kind, d in segments if kind != 'grasp')
        snap_timer = Timer(1.5 + total * 0.5, self._snapshot)
        snap_timer.start()

        ok, reason = self.executor.execute(segments)
        snap_timer.cancel()

        # 파지류 동작이면 라벨 수집 창을 연다 - 이어지는 "잡았어/놓쳤어"
        # 한마디가 실전 파지 성패 데이터가 된다 (지식 원장의 병목은
        # 시도 수가 아니라 결과 라벨이다).
        def _uses_gripper(seg):
            head_ = seg[0]
            if head_ == 'grasp':
                return True
            joints = head_.keys() if isinstance(head_, dict) else (
                head_ if isinstance(head_, (list, tuple, set)) else ())
            return any(str(j).endswith('.gripper') for j in joints)
        if any(_uses_gripper(seg) for seg in segments):
            self.last_grasp_at = time.time()
            self.last_grasp_text = text

        self._log(text, motion, 'executed' if ok else 'failed',
                  {'reason': reason,
                   'grasp_ok': getattr(self.executor, 'last_grasp_ok', None),
                   'max_divergence_deg': round(getattr(self.executor, 'max_divergence', 0), 1)})

        if not ok and reason:
            if self.speech is not None:
                self.speech.wait()
            say_and_move(self.reachy, text=reason, speech=self.speech, head=head)
            return

        # If it is still talking after the gesture, let the head talk along.
        if head is not None and self.speech is not None and self.speech.is_playing:
            head.setup()
            head.run_while(lambda: self.speech.is_playing, timeout=30)
            head.home()


def wants_vision(text):
    """Check if the request is about what the robot sees."""
    compact = text.replace(' ', '')
    return any(w.replace(' ', '') in compact for w in VISION_WORDS)


_CAMERA_LOCK = None

# Stable camera names from ops/99-reachy-cameras.rules. /dev/videoN numbering
# swaps on boot and the two head cameras are NOT equivalent (one is sharply
# focused, the other badly out of focus), so resolve the pinned symlink to an
# index instead of trusting the number.
CAMERA_ALIASES = {'main': '/dev/reachy-cam-main', 'sub': '/dev/reachy-cam-sub'}


def resolve_camera(name_or_index, default=0):
    """'main'/'sub'/'/dev/videoN'/int -> a V4L2 index OpenCV can open."""
    if isinstance(name_or_index, int):
        return name_or_index
    value = str(name_or_index)
    path = CAMERA_ALIASES.get(value, value)
    if path.startswith('/dev/'):
        try:
            real = os.path.realpath(path)
            digits = ''.join(c for c in os.path.basename(real) if c.isdigit())
            if digits:
                return int(digits)
        except Exception:
            pass
        logger.warning('Camera %s not found - falling back to index %d',
                       path, default)
        return default
    try:
        return int(value)
    except ValueError:
        return default



def grab_frame(reachy, side='left', camera_index=0):
    """Return one BGR frame of the robot's view (ndarray), or None.

    Different reachy checkouts expose the camera differently, so this tries
    them in order: left/right_camera objects, head.get_image(), and finally
    opening the video device directly with OpenCV. Shared by the vision
    request path and the background presence watcher; a module lock keeps the
    two from opening /dev/video0 at the same moment.
    """
    import cv2 as cv

    global _CAMERA_LOCK
    if _CAMERA_LOCK is None:
        from threading import Lock
        _CAMERA_LOCK = Lock()

    with _CAMERA_LOCK:
        return _grab_frame_locked(cv, reachy, side, camera_index)


SDK_CAMERA_WARMUP = 4.0   # 연결 직후 첫 프레임을 기다려 주는 최대 시간(초)
_SDK_CAMERA_READY = False  # SDK 카메라에서 한 번이라도 프레임을 받았는가


def check_sdk_camera(camera_index='main'):
    """SDK 가 여는 장치가 우리가 쓰려던 카메라와 같은지 확인하고 알린다.

    SDK 는 /dev/video0 을 고정으로 연다(BackgroundVideoCapture(0)). udev 로
    붙인 이름(reachy-cam-main)이 다른 노드를 가리키면, 로봇은 아무 말 없이
    흐린 쪽 카메라로 보게 된다. 두 카메라의 선명도는 실측 212 대 62 로 차이가
    크므로 조용히 넘어가면 안 된다.
    """
    link = '/dev/reachy-cam-%s' % camera_index if camera_index in ('main', 'sub') \
        else None
    if link is None or not os.path.exists(link):
        return
    target = os.path.realpath(link)
    if target == '/dev/video0':
        logger.info('카메라 확인: SDK 가 여는 video0 = %s (%s)', link, camera_index)
    else:
        logger.warning(
            '카메라 불일치: SDK 는 /dev/video0 을 여는데 %s 는 %s 입니다. '
            '로봇이 의도한 카메라로 보고 있지 않습니다 (USB 포트를 바꿔 꽂으세요).',
            link, target)


def _grab_frame_locked(cv, reachy, side, camera_index):
    head = reachy.head
    img = None

    # SDK 카메라가 먼저다. connect() 가 이미 /dev/video0 을 열어 백그라운드로
    # 스트리밍하고 있어서, 같은 장치를 우리가 또 열면 open 이 영영 돌아오지
    # 않는다(실측: VideoCapture 생성에서 무한 대기). 그러면 presence 스레드가
    # 첫 프레임에서 멎어 얼굴 검출·복도 인사·사람 수집이 전부 조용히 죽는다.
    # 이미 열려 있는 스트림을 그대로 쓰는 것이 유일하게 안전한 길이다.
    global _SDK_CAMERA_READY
    camera = getattr(head, 'camera', None)
    if camera is not None and hasattr(camera, 'read'):
        success, frame = camera.read()
        if success and frame is not None:
            _SDK_CAMERA_READY = True
            return frame
        # 연결 직후에는 백그라운드 스레드가 아직 첫 프레임을 못 받았을 수 있다
        # (실측 약 2.2초). 그때 한 번만 기다린다. 한 번이라도 받은 뒤에는
        # 기다리지 않는다 - 매 호출마다 4초씩 멈추면 사람이 지나가는 동안
        # 감시 스레드가 사실상 멎어 버린다.
        if not _SDK_CAMERA_READY:
            for _ in range(int(SDK_CAMERA_WARMUP / 0.3)):
                time.sleep(0.3)
                success, frame = camera.read()
                if success and frame is not None:
                    _SDK_CAMERA_READY = True
                    return frame
        logger.warning('SDK 카메라에서 프레임을 못 받았습니다')
        return None

    # 1) left_camera / right_camera attributes (this repo's head.py)
    other = 'right' if side == 'left' else 'left'
    for name in (side, other):
        camera = getattr(head, name + '_camera', None)
        if camera is None:
            continue
        success, frame = camera.read()
        if success and frame is not None:
            return frame
        logger.warning('Camera read failed (%s)', name)

    # 2) head.get_image() (the variant installed on the robot's Pi)
    if hasattr(head, 'get_image'):
        try:
            frame = head.get_image()
            if frame is not None and len(frame) > 0:
                return frame
        except Exception:
            logger.warning('head.get_image() failed', exc_info=True)

    # 3) 장치를 직접 연다. SDK 카메라가 있으면 그 장치는 이미 SDK 것이므로
    #    절대 열지 않는다 (열면 블록된다). camera_check.py 처럼 SDK 없이 쓰는
    #    경우에만 여기까지 온다.
    if camera is not None:
        return img
    capture = cv.VideoCapture(resolve_camera(camera_index))
    try:
        frame = None
        success = False
        # First frames can be dark while auto-exposure settles.
        for _ in range(5):
            success, frame = capture.read()
        if success and frame is not None:
            img = frame
    finally:
        capture.release()

    return img


def capture_view(reachy, side='left', width=640, quality=70, camera_index=0):
    """Grab one frame of the robot's view as base64 jpeg, or None on failure."""
    import base64

    try:
        import cv2 as cv

        img = grab_frame(reachy, side=side, camera_index=camera_index)
        if img is None:
            logger.warning('No camera image available')
            return None

        h, w = img.shape[:2]
        if w > width:
            img = cv.resize(img, (width, int(h * width / w)))

        success, buf = cv.imencode('.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return None

        return base64.b64encode(buf.tobytes()).decode('ascii')
    except Exception:
        logger.exception('Camera capture failed')
        return None

class Listener(object):
    """Microphone + speech-to-text.

    Args:
        language (str): recognition language (e.g. 'ko-KR')
        mic_index (int): microphone device index (None = system default)
        energy (int): initial energy threshold; higher = less sensitive
        phrase_limit (float): max seconds per utterance
    """

    def __init__(self, language='ko-KR', mic_index=None, energy=200, phrase_limit=10,
                 pause=0.5, energy_floor=100, energy_ceil=400,
                 fixed_energy=None, stt='auto', vosk_model=None):
        with quiet_stderr():
            import speech_recognition as sr

        self.sr = sr
        self.language = language
        self.phrase_limit = phrase_limit

        # STT backend: 'google' (online), 'vosk' (offline), 'auto' (vosk if a
        # model loads, else google - offline-first for a flaky hallway).
        self.stt = stt
        self.last_engine = None   # which STT engine produced the last text
        self.last_rms = None      # captured audio level of the last utterance
        self._vosk = None
        if stt in ('vosk', 'auto') and vosk_model:
            self._vosk = self._load_vosk(vosk_model)
        if stt == 'vosk' and self._vosk is None:
            logger.warning('Vosk requested but model failed to load - '
                           'falling back to Google STT')
            self.stt = 'google'
        # The dynamic threshold is clamped to this band each turn. A low floor
        # matters in a quiet hallway where people speak softly - too high and
        # the recognizer never hears them; the ceiling stops it going deaf if
        # a burst of noise pushes it up.
        self.energy_floor = energy_floor
        self.energy_ceil = energy_ceil

        if mic_index is None:
            mic_index = self.pick_best_mic()
            if mic_index is not None:
                logger.info('Auto-selected microphone index %d', mic_index)

        self._check_input_device(mic_index)

        self.recognizer = sr.Recognizer()
        # Bound the Google STT HTTP call: the default is None (unbounded
        # urlopen), so a half-open connection would freeze the loop forever.
        self.recognizer.operation_timeout = 10
        # Fixed threshold = deterministic sensitivity: nothing below it is ever
        # treated as speech. Best for a stationary robot in a steady room - the
        # dynamic threshold otherwise drifts DOWN in quiet and starts catching
        # micro-noise (RMS ~75-280) below the floor. --fixed-energy N pins it.
        self.fixed_energy = fixed_energy
        if fixed_energy is not None:
            self.recognizer.energy_threshold = fixed_energy
            self.recognizer.dynamic_energy_threshold = False
        else:
            self.recognizer.energy_threshold = energy
            self.recognizer.dynamic_energy_threshold = True
        # How much silence marks the end of an utterance. The 0.8s default
        # adds nearly half a second of dead air to every single turn.
        self.recognizer.pause_threshold = pause
        self.recognizer.non_speaking_duration = min(0.5, pause)

        with quiet_stderr():
            self.microphone = sr.Microphone(device_index=mic_index)

            # Calibrate for room noise once, so speech detection is reliable.
            # Skip it entirely when --fixed-energy pins the threshold, otherwise
            # ambient calibration OVERWRITES the pinned value (that was the bug:
            # --fixed-energy 250 ended up at ~600 after calibration).
            if fixed_energy is None:
                with self.microphone as source:
                    logger.info('Calibrating for ambient noise...')
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)

        # The stream is opened once and kept open across turns (see listen()).
        # Re-opening it every turn added stream-startup latency, so the first
        # syllable of the next utterance was clipped.
        self._source = None

        logger.info('Energy threshold: %d', self.recognizer.energy_threshold)
        if self.recognizer.energy_threshold < 60:
            logger.warning('Energy threshold is suspiciously low - '
                           'the microphone may not be picking up any sound.')

    def _check_input_device(self, mic_index):
        """Fail with a clear message when there is no usable microphone."""
        import pyaudio

        with quiet_stderr():
            pa = pyaudio.PyAudio()
        try:
            if mic_index is not None:
                info = pa.get_device_info_by_index(mic_index)
            else:
                try:
                    info = pa.get_default_input_device_info()
                except (IOError, OSError):
                    raise RuntimeError(
                        'No microphone found. The Pi has no built-in mic - plug in '
                        'a USB microphone and check "arecord -l" / --list-mics.')

            if int(info.get('maxInputChannels', 0)) < 1:
                raise RuntimeError(
                    'Device [{}] "{}" has no input channels - it is not a '
                    'microphone. Pick another index from --list-mics.'.format(
                        info.get('index'), info.get('name')))

            logger.info('Microphone: [%s] %s', info.get('index'), info.get('name'))
        finally:
            pa.terminate()

    @staticmethod
    def list_microphones():
        """List (index, name, is_input) for every audio device."""
        import pyaudio

        with quiet_stderr():
            pa = pyaudio.PyAudio()
        try:
            return [
                (i,
                 pa.get_device_info_by_index(i).get('name', '?'),
                 int(pa.get_device_info_by_index(i).get('maxInputChannels', 0)) > 0)
                for i in range(pa.get_device_count())
            ]
        finally:
            pa.terminate()

    @classmethod
    def pick_best_mic(cls):
        """Pick the most robot-appropriate microphone automatically.

        Preference: a ReSpeaker array (far-field, made for voice) > any other
        USB input device > the system default (None).
        """
        inputs = [(i, name) for i, name, is_input in cls.list_microphones() if is_input]

        for i, name in inputs:
            if 'respeaker' in name.lower():
                return i
        for i, name in inputs:
            lowered = name.lower()
            if 'usb' in lowered and 'bcm2835' not in lowered:
                return i
        return None

    def _load_vosk(self, model_path):
        """Load a Vosk model + recognizer, or None on failure."""
        model_path = os.path.expanduser(model_path)
        if not os.path.isdir(model_path):
            logger.warning('Vosk model not found at %s', model_path)
            return None
        try:
            import vosk
            vosk.SetLogLevel(-1)   # silence kaldi chatter
            model = vosk.Model(model_path)
            rec = vosk.KaldiRecognizer(model, 16000)
            self._vosk_model = model   # keep a ref alive
            logger.info('Vosk offline STT loaded from %s', model_path)
            return rec
        except Exception:
            logger.exception('Could not load Vosk model')
            return None

    def _ensure_stream(self):
        """Open the mic stream once and keep it open across turns."""
        if self._source is None:
            with quiet_stderr():
                self._source = self.microphone.__enter__()

    def _flush(self):
        """Discard audio buffered while the robot was speaking/thinking.

        The stream stays open across turns (so the leading syllable of the
        next utterance is never lost to stream-startup lag), but that means
        the robot's own TTS leaks into the buffer. Drain it right before we
        start listening so we begin clean without a stale, self-heard head.
        """
        try:
            raw = self._source.stream.pyaudio_stream
            chunk = self._source.CHUNK
            avail = raw.get_read_available()
            while avail > 0:
                raw.read(min(avail, chunk), exception_on_overflow=False)
                avail = raw.get_read_available()
        except Exception:
            pass

    def close(self):
        """Release the persistent mic stream (call once at shutdown)."""
        if self._source is not None:
            try:
                with quiet_stderr():
                    self.microphone.__exit__(None, None, None)
            except Exception:
                pass
            self._source = None

    def listen(self):
        """Wait for one utterance and return its text.

        Returns:
            str: recognized text
            None: heard something but could not understand it
            '': network/service problem (treat as offline)
        """
        try:
            self._ensure_stream()
            with quiet_stderr():
                self._flush()
                # The dynamic threshold drifts UP after any noise (going deaf to
                # soft voices) and DOWN in dead silence (hair-trigger). Clamp
                # it to the configured band every turn - lower the floor for a
                # quiet room with soft speakers (--energy-floor). Skipped when
                # --fixed-energy pins the threshold outright.
                if self.fixed_energy is None:
                    self.recognizer.energy_threshold = min(
                        max(self.recognizer.energy_threshold, self.energy_floor),
                        self.energy_ceil)
                source = self._source
                logger.info('Listening...')
                try:
                    audio = self.recognizer.listen(
                        source, timeout=20, phrase_time_limit=self.phrase_limit)
                except self.sr.WaitTimeoutError:
                    # Nothing said for a while; keep the stream open, wait again.
                    logger.info('(no speech for 20s)')
                    return None
        except Exception:
            # A device hiccup (USB mic brownout, ALSA stream error) must never
            # kill the whole program: drop the broken stream so the next call
            # reopens it fresh, breathe, and treat this turn as silence.
            logger.exception('Audio capture failed - resetting the mic stream')
            self.close()
            time.sleep(2)
            return None

        # Log the captured level vs the threshold - the fastest way to tune
        # --energy-floor/--energy-ceil on site (soft speech should read clearly
        # above the threshold; if 'could not understand' with a low RMS, the
        # voice is too quiet/far - raise AGC or move closer).
        try:
            import audioop
            rms = audioop.rms(audio.get_raw_data(), 2)
            self.last_rms = rms
            logger.info('captured RMS %d (threshold %d)',
                        rms, int(self.recognizer.energy_threshold))
        except Exception:
            pass

        text = self._recognize(audio)
        if text is None:
            return None
        if text == '':
            return ''

        text = _normalize_name(text)
        logger.info('Heard: %s', text)
        return text

    def _recognize(self, audio):
        """Turn captured audio into text. Returns str, None (unclear), '' (err).

        Backends: offline Vosk (no internet) and/or online Google. In a hallway
        with flaky wifi, Vosk keeps the robot listening when Google can't be
        reached; Google is more accurate when the connection is good.
        """
        # Pure offline: Vosk only, never touches the network.
        if self.stt == 'vosk':
            self.last_engine = 'vosk'
            return self._recognize_vosk(audio)

        # 'google' and 'auto' both try Google first (far more accurate on real,
        # distant hallway speech than the small Vosk model). 'auto' then falls
        # back to Vosk only when Google is unreachable, so a wifi blip does not
        # deafen the robot. (Vosk-first was a mistake: it returns confident
        # garbage like "[90]" that auto then trusted instead of asking Google.)
        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            self.last_engine = 'google'
            return text
        except self.sr.UnknownValueError:
            logger.info('Heard something, could not understand it')
            return None
        except self.sr.RequestError as e:
            logger.warning('Google STT unavailable: %s', e)
            if self.stt == 'auto' and self._vosk is not None:
                logger.info('Falling back to offline Vosk')
                self.last_engine = 'vosk'
                return self._recognize_vosk(audio)
            return ''

    def _recognize_vosk(self, audio):
        import json

        try:
            data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            self._vosk.AcceptWaveform(data)
            text = (json.loads(self._vosk.FinalResult()).get('text') or '').strip()
            self._vosk.Reset()
            if not text:
                logger.info('Vosk heard nothing intelligible')
                return None
            # Vosk emits space-separated tokens; keep as-is (Korean STT normalize
            # handles the rest downstream).
            return text
        except Exception:
            logger.exception('Vosk recognition failed')
            return None


# STT often mangles the wake-name '리치'. Only fix clear name-mishears; leave
# real words like '위치'(position) alone unless they stand alone as address.
# STT 가 흔히 틀리는 동작 단어 (문맥 안전한 것만)
# '학수'(학수고대)/'아까 사'(아까 사무실) proved unsafe as blind replaces -
# the LLM-side persona hints still cover those mishears in context.
_WORD_FIXES = [('안대나', '안테나'), ('보간함', '보관함'), ('방수', '박수'),
               ('만새', '만세')]
# '유치하'(childish) and '이치'(reason) are real words - handled by the
# sentence-start position guard below, not blind-replaced here.
_NAME_MISHEARS = ('다비치', '리치야', '리치아', '루치아', '유치아',
                  '니치', '릿지', '리찌', '리취', '리치가', '리치는')


def load_stt_corrections(path):
    """Merge extra STT mishear fixes from a JSON note file into the built-ins.

    Lets the correction lists grow from conversation logs without editing code
    (config/stt_corrections.json). Schema:
      {"word_fixes": [["양파","양팔"], ...], "name_mishears": ["이츠야", ...]}
    Deterministic word_fixes run before routing, so they also fix which turns
    are recognized as motion commands - not just what the LLM sees.
    """
    import json

    global _WORD_FIXES, _NAME_MISHEARS
    try:
        with open(os.path.expanduser(path), encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        logger.info('STT corrections: none loaded from %s', path)
        return 0

    added = 0
    have_fix = {tuple(p) for p in _WORD_FIXES}
    for pair in data.get('word_fixes', []):
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] and pair[1]:
            t = (pair[0], pair[1])
            if t not in have_fix:
                _WORD_FIXES.append(t)
                have_fix.add(t)
                added += 1

    names = list(_NAME_MISHEARS)
    have_name = set(names)
    for w in data.get('name_mishears', []):
        if w and w not in have_name:
            names.append(w)
            have_name.add(w)
            added += 1
    _NAME_MISHEARS = tuple(names)

    logger.info('STT corrections: +%d (%d word fixes, %d name mishears)',
                added, len(_WORD_FIXES), len(_NAME_MISHEARS))
    return added


def _normalize_name(text):
    for w in _NAME_MISHEARS:
        text = text.replace(w, '리치')
    # Address-position mishears at sentence start -> '리치'. These are real
    # words (위치=position, 지하=basement, 유치원=kindergarten, 몇시야=what
    # time...), so two guards: they must OPEN the utterance the way the name
    # would, AND the rest must contain a robot-ish cue (a command/greeting) -
    # otherwise "위치 알려줘" or "몇시야 지금" are real questions, not the name.
    # These stand-alone words are real questions ("위치가 어디야", "몇 시야"),
    # but the SAME word followed by anything else is almost always the name
    # being misheard - people say "리치, <명령>". So: alone (or with a
    # time/place question word) = keep; followed by other content = the name.
    # An explicit cue list kept missing cases (노래/춤/소개/이야기...), which is
    # why "몇 시야 노래 불러줘" stayed misheard.
    # Short fragments like '이야' are unsafe here ('이야기 해줘' would match).
    _REAL_QUESTION = ('지금', '몇시', '몇 시', '시간', '어디', '어때', '뭐야',
                      '뭐예요', '몇 분', '몇분')
    stripped = text.strip()
    for w in ('위치', '유치원', '유치하', '유치', '지하', '몇 시야', '몇시야',
              '비치', '이치'):
        if stripped == w or stripped.startswith(w + ' '):
            rest = stripped[len(w):].strip()
            if not rest:
                break                      # bare word = the real question
            compact_rest = rest.replace(' ', '')
            if any(q.replace(' ', '') in compact_rest for q in _REAL_QUESTION):
                break                      # "위치가 어디야", "몇 시야 지금"
            text = text.replace(w, '리치', 1)
            break
    for a, b in _WORD_FIXES:
        text = text.replace(a, b)
    return text


def is_stop_word(text):
    """Check if the user asked to end the conversation.

    Near-exact only: a stop word buried in a longer sentence is almost always
    overheard hallway talk ("이제 그만 가자", "회사 그만두고 싶다", "종료
    버튼이 어디에 있어요") - substring matching here used to KILL the whole
    program on those. The utterance must essentially BE the stop phrase,
    with at most a short polite tail ("그만할게요", "잘가요").
    """
    compact = text.replace(' ', '')
    for w in STOP_WORDS:
        w = w.replace(' ', '')
        if compact.startswith(w) and len(compact) - len(w) <= 3:
            return True
    return False


def prepare_fillers(speech, cache_dir='~/.cache/reachy_fillers'):
    """Pre-synthesize short acknowledgements ("um...") for instant playback.

    Played when the LLM+TTS wait drags on, so it feels like pondering instead
    of lag. The cache key includes the engine and voice: switching TTS engines
    regenerates the fillers instead of replaying stale ones.
    """
    # Neutral pondering hums, chained back-to-back during a long wait so
    # there is no dead air. A wide, varied set so a long wait never repeats
    # the same sound twice in a row and it reads as thinking, not a loop.
    phrases = [
        '음...', '어디 보자...', '그러니까...', '음, 잠깐만요.', '네...',
        '아, 네...', '흠...', '그게...', '음, 그러니까요...', '잠시만요.',
        '아하...', '으음...', '한번 볼게요.', '네, 네...', '그렇군요...',
        '음, 어디 보자.', '오...', '그거는...',
    ]

    # Persistent (NOT /tmp): the Pi's /tmp is tmpfs, so a reboot wiped the
    # cache and re-synthesis needed network right at boot - if the wifi was
    # late, the whole day ran with no fillers (dead air on every LLM wait).
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    voice = getattr(speech, 'edge_voice', '') if speech.engine == 'edge' else speech.voice
    slug = '{}-{}'.format(speech.engine, ''.join(c for c in voice if c.isalnum()))

    import hashlib

    paths = []
    for phrase in phrases:
        # Hash the phrase into the filename so editing the wording regenerates
        # the file instead of replaying a stale cache from an earlier list.
        tag = hashlib.md5(phrase.encode('utf-8')).hexdigest()[:8]
        path = os.path.join(cache_dir, 'filler_{}_{}.mp3'.format(slug, tag))
        if os.path.exists(path) or speech.synthesize_to_file(phrase, path):
            paths.append(path)

    logger.info('%d filler sounds ready (%s)', len(paths), slug)
    return paths


RECACHE_TRIES = 10      # 연결이 돌아오길 기다리며 다시 시도하는 횟수
RECACHE_WAIT = 60.0     # 시도 사이 간격(초)


def precache_common_lines(speech, notes=None):
    """자주 하는 말을 미리 합성해 둔다.

    이 Pi 는 Buster(glibc 2.28) 라 오프라인 신경망 TTS(piper)를 못 쓴다 - 받아
    보니 어느 릴리스든 GLIBC 2.29~2.30 을 요구한다. 자연스러운 목소리(edge)는
    인터넷이 필요하므로, 복도 와이파이가 끊기면 결국 espeak(기계음)까지
    내려간다. 그래서 '늘 하는 말'만이라도 미리 만들어 두면, 끊긴 동안에도
    인사와 단골 답변은 자연스러운 목소리로 나온다. 덤으로 합성 대기도 없다.
    """
    if speech is None or not hasattr(speech, 'precache'):
        return 0

    from threading import Thread

    phrases = []
    try:
        # 인사말·권유말에 더해, 캠페인 멘트가 붙은 조합까지 전부. 붙은 문장은
        # 통째로 한 번에 합성되므로 조합을 캐시해야 실제로 캐시가 맞는다.
        from hallway import all_spoken_lines
        phrases.extend(all_spoken_lines())
    except Exception:
        logger.debug('복도 인사말을 읽지 못했습니다', exc_info=True)

    # QuickNotes 는 (패턴들, 답변, 종류) 로 컴파일해 들고 있다.
    for _pats, reply, _kind in getattr(notes, '_compiled', None) or []:
        phrases.append(reply)

    # 로그에서 뽑은, 실제로 되풀이된 답변들 (ops/build_tts_cache_list.py 가 만든다).
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, 'cached_lines.json'),
                 os.path.join(here, '..', 'config', 'cached_lines.json')):
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as fh:
                    phrases.extend(json.load(fh).get('lines') or [])
            except Exception:
                logger.debug('cached_lines.json 을 읽지 못했습니다', exc_info=True)
            break

    phrases = [p for p in dict.fromkeys(phrases) if p]
    if not phrases:
        return 0

    # 백그라운드에서 채운다. 합성에는 인터넷이 필요한데 복도 와이파이는 부팅
    # 직후에 특히 잘 끊긴다(실제로 39개 중 7개가 그때 이름 풀이 실패로 빠졌다).
    # 기동을 붙잡아 두지 않고, 연결이 돌아오면 나머지를 채우도록 되풀이한다.
    def _fill():
        for attempt in range(RECACHE_TRIES):
            missing = [ph for ph in phrases if speech.cached_path(ph) is None]
            if not missing:
                if attempt:
                    logger.info('자주 쓰는 말 %d개 모두 준비됐습니다', len(phrases))
                return
            try:
                made = speech.precache(missing)
            except Exception:
                made = 0
                logger.debug('미리 합성 실패', exc_info=True)
            left = len([ph for ph in phrases if speech.cached_path(ph) is None])
            logger.info('자주 쓰는 말 %d개 중 %d개 준비 (이번에 %d개 추가) - '
                        '인터넷이 끊겨도 이 말들은 자연스러운 목소리로 나옵니다',
                        len(phrases), len(phrases) - left, made)
            if not left:
                return
            time.sleep(RECACHE_WAIT)
        logger.info('자주 쓰는 말 일부를 아직 못 만들었습니다 (인터넷 확인)')

    t = Thread(target=_fill)
    t.daemon = True
    t.start()
    return len(phrases)


# 첫 문장을 기다리는 한계. 모델이 1.2초면 답하므로 넉넉하다. 이걸 넘으면
# 스트리밍을 포기하고 평소의 한 번에 받는 경로로 내려간다.
STREAM_FIRST_WAIT = 25.0
STREAM_NEXT_WAIT = 20.0

# 살아 있다는 줄을 이만큼마다 남긴다. 감시 쪽(ops/pi/pi_watchdog.sh)은 이보다
# 넉넉히 기다렸다가 판단한다.
HEARTBEAT_EVERY = 300.0


def stream_reply(client, text, image=None):
    """브로커에 스트리밍으로 묻는다. (첫 문장, 나머지 이터레이터, 결과칸).

    첫 문장이 닿을 때까지만 막는다. 그 뒤 문장들은 오는 대로 흘려 주므로,
    부르는 쪽은 첫 문장을 말하는 동안 나머지를 받을 수 있다.

    스트리밍을 쓸 수 없거나 한 문장도 못 받으면 (None, None, 결과칸).
    """
    if not hasattr(client, 'ask_stream'):
        return None, None, None

    from queue import Empty, Queue
    from threading import Thread

    q = Queue()
    box = {'reply': None}

    def producer():
        try:
            box['reply'] = client.ask_stream(text, q.put, image=image)
        except Exception:
            logger.exception('스트리밍 요청이 실패했습니다')
        finally:
            q.put(None)

    thread = Thread(target=producer)
    thread.daemon = True
    thread.start()

    try:
        first = q.get(timeout=STREAM_FIRST_WAIT)
    except Empty:
        logger.warning('첫 문장이 %.0f초 안에 오지 않았습니다', STREAM_FIRST_WAIT)
        first = None

    if not first:
        return None, None, box

    def rest():
        while True:
            try:
                item = q.get(timeout=STREAM_NEXT_WAIT)
            except Empty:
                logger.warning('다음 문장을 기다리다 끊었습니다')
                return
            if item is None:
                return
            yield item

    return first, rest(), box


def _sentence_stream(first, rest):
    """첫 문장과 나머지를 하나의 흐름으로 잇는다."""
    yield first
    if rest is not None:
        for item in rest:
            yield item


def run_loop(listener, client, reachy=None, speech=None, head=None, fillers=(),
             ack_delay=0.8, idle=None, vision=False, camera_side='left',
             camera_index=0, motion_handler=None, turn_logger=None, notes=None,
             hallway=None, objvis=None, music_player=None, sleeper=None):
    """Main conversation loop. Blocks until a stop word or Ctrl-C."""
    import random

    from say_and_move import say_and_move, say_sentences_and_move

    def speak(line):
        if reachy is not None:
            say_and_move(reachy, text=line, speech=speech, head=head)
        elif speech is not None:
            speech.start(text=line)
            speech.wait()
        if hallway is not None:
            hallway.note_activity()   # keep the greeter quiet mid-conversation

    try:
        _run(listener, client, reachy, speech, head, fillers, ack_delay, idle,
             say_and_move, random, speak, vision, camera_side, camera_index,
             motion_handler, turn_logger, notes, hallway, objvis,
             music_player, say_sentences_and_move, sleeper)
    finally:
        # Order matters: silence the greeter FIRST so its finally-block can't
        # restart idle after we stop it (then throw against a closed robot).
        if hallway is not None:
            hallway.stop()
        if idle is not None:
            idle.stop()
        listener.close()


MOTOR_RECOVERY_IO = None   # main 이 채운다 (--motions 시 io 경로)


def _run(listener, client, reachy, speech, head, fillers, ack_delay, idle,
         say_and_move, random, speak, vision=False, camera_side='left',
         camera_index=0, motion_handler=None, turn_logger=None, notes=None,
         hallway=None, objvis=None, music_player=None,
         say_sentences_and_move=None, sleeper=None):
    from threading import Event, Thread

    last_kind = None   # 직전 턴 종류 (motion/chat) — 문맥 라우팅용
    offline_mute_until = 0.0   # STT 불가 안내 백오프 (한 번 말하고 점점 조용히)
    offline_backoff = 60.0
    was_asleep = False   # 잠드는 '순간'을 알아야 팔을 한 번만 내린다
    # 살아 있다는 표시를 이따금 남긴다. 프로세스가 떠 있는 것과 대화 루프가
    # 실제로 돌고 있는 것은 다른 얘기다 - 마이크나 카메라에서 멎으면 systemd
    # 는 아무것도 알아채지 못한다. ops/pi/pi_watchdog.sh 가 이 줄이 끊기면
    # 서비스를 다시 올린다.
    heartbeat_at = 0.0
    motor_probe_at = 0.0
    turns = 0
    while True:
        # 모터 자동 복구: 연결은 시작 때 한 번뿐이라, 꺼진 채 부팅되면
        # 나중에 전원을 켜도 재시작 전까지 영영 '소리만' 모드였다.
        # 소리만 모드에서 90초마다 모터를 조용히 찔러 보고, 응답이 오면
        # 스스로 내려간다 - systemd 가 즉시 되살리며 완전한 동작 모드로
        # 부팅된다 (사람이 전원만 켜면 2분 안에 팔이 돌아온다).
        if (reachy is None and MOTOR_RECOVERY_IO
                and time.time() - motor_probe_at >= 90.0):
            motor_probe_at = time.time()
            try:
                from base_pose import connect as _probe_connect
                _r = _probe_connect(io=MOTOR_RECOVERY_IO, with_head=True)
                logger.info('모터 전원 감지 - 동작 모드로 재시작합니다')
                try:
                    speak('모터가 켜졌네요. 잠깐만요, 기지개 켜고 올게요.')
                except Exception:
                    pass
                os._exit(0)
            except Exception:
                pass                      # 아직 꺼져 있음 - 조용히 계속
        # 카메라 자가 복구: 장치가 뽑히거나 죽으면(VIDIOC 오류 스팸)
        # 감시가 조용히 눈을 감은 채 며칠을 보낸다 (실측: 09-11~14,
        # 65시간 수집 공백). 프레임이 10분 넘게 안 오면 스스로 내려가
        # systemd 재기동으로 장치를 다시 연다 - 모터 자동복구와 같은 꼴.
        _w = getattr(sleeper, 'watcher', None) if sleeper is not None else None
        if (_w is not None
                and getattr(_w, 'last_tick', 0)
                and time.time() - getattr(_w, 'last_frame_at',
                                          time.time()) > 600.0):
            logger.error('카메라 프레임 10분 기아 - 재시작으로 장치를 다시 엽니다')
            os._exit(0)
        if time.time() - heartbeat_at >= HEARTBEAT_EVERY:
            heartbeat_at = time.time()
            logger.info('심장박동: %s, 턴 %d건, 마지막 밝기 %s',
                        '자는 중' if (sleeper is not None and sleeper.asleep)
                        else '깨어 있음', turns,
                        '%.3f' % sleeper.brightness()
                        if sleeper is not None and sleeper.brightness() is not None
                        else '없음')
        # Re-assert neck stiffness each loop: the Orbita disks can thermally
        # cut torque while holding the head, and go limp until re-gripped.
        # 자는 동안에는 건너뛴다 - 절전(토크 차단)을 매 바퀴 되살리면
        # 모터 전력을 끊은 의미가 없다.
        if (reachy is not None and getattr(reachy, 'head', None) is not None
                and not (sleeper is not None and sleeper.asleep)):
            try:
                reachy.head.compliant = False
            except Exception:
                pass

        # Back at the top of the loop = the previous turn (if any) is done.
        if hallway is not None:
            hallway.end_turn()

        # 불이 꺼지고 사람도 없으면 잔다. 빈 사무실에서 로봇이 혼자 움직이고
        # 소리를 내면 무섭다. 자는 동안에도 마이크·카메라·터널·브로커는 그대로
        # 살아 있다 - 멈추는 것은 '스스로 시작하는' 움직임과 소리뿐이다.
        asleep = sleeper.update() if sleeper is not None else False
        if asleep:
            if idle is not None:
                idle.stop()
            if motion_handler is not None:
                # 팔을 내리는 것은 '잠드는 순간' 한 번뿐이다. 자는 동안 매
                # 바퀴 부르면 이미 내려간 팔에 대고 모터를 켰다 껐다 하며
                # 밤새 소리를 낸다 - 없애려던 바로 그 소리다.
                try:
                    motion_handler.executor.stop_idle_arms()
                    if not was_asleep:
                        motion_handler.executor._cancel_settle()
                        motion_handler.executor._settle_quietly()
                except Exception:
                    logger.debug('자는 동안 팔 내리기 실패', exc_info=True)
            if not was_asleep and reachy is not None:
                # 절전: 팔을 내린 뒤 모든 모터 토크를 끊는다 (모터 전력
                # 절약 - 사용자 지정). 깨면 목은 루프의 재강직이, 팔은
                # hold_ready/idle 경로가 다시 힘을 준다.
                try:
                    for m in getattr(reachy, 'motors', []) or []:
                        m.compliant = True
                    if getattr(reachy, 'head', None) is not None:
                        reachy.head.compliant = True
                    logger.info('절전: 모터 토크 차단 (모든 모터 compliant)')
                except Exception:
                    logger.debug('모터 토크 차단 실패', exc_info=True)
        else:
            # Small random motions while waiting keep the robot looking alive:
            # the head glances/breathes and (with arms) the held pose breathes
            # too, so the robot looks relaxed and welcoming rather than frozen.
            # (While the hallway greeter is mid-speech it owns the head - skip.)
            if idle is not None and not (hallway is not None and hallway.busy):
                idle.start()
            if motion_handler is not None:
                motion_handler.executor.start_idle_arms()

        was_asleep = asleep
        text = listener.listen()

        if text is None:
            # Silence timeout or not understood - just keep listening quietly.
            continue

        if text == '':
            # STT service unreachable. In a noisy hallway this path fires on
            # every captured noise - announce once, then back off (60s
            # doubling to 10min) instead of repeating the line all hour.
            now = time.time()
            if asleep:
                # 자는 중에는 이 안내도 하지 않는다. 아무도 없는 캄캄한
                # 복도에서 로봇이 혼자 말하는 것이 바로 피하려던 일이다.
                time.sleep(2)
                continue
            if now >= offline_mute_until:
                speak('지금 인터넷이 불안정한 것 같아요.')
                offline_mute_until = now + offline_backoff
                offline_backoff = min(offline_backoff * 2, 600.0)
            time.sleep(2)
            continue

        offline_backoff = 60.0   # network is back - reset the backoff

        # The far-field mic hears the robot's own hallway greeting; drop
        # recognized text that is just our recent words coming back.
        if hallway is not None and hallway.is_echo(text):
            logger.info('Dropping self-echo: %r', text)
            continue

        if asleep:
            # 자는 중에 들린 소리. 사람이 정말 말을 건 것만 깨우고, 잡음은
            # 못 들은 척한다 - 어두운 복도에서 로봇이 잡음에 반응해 혼자
            # 말하기 시작하는 것이 무서운 부분이다.
            if looks_like_noise(text, getattr(listener, 'last_engine', None)):
                logger.info('자는 중 잡음은 지나칩니다: %r', text)
                continue
            if not sleeper.wake('말을 걸었습니다'):
                # 근무시간 밖 무조건 절전 - 말을 걸어도 자는 채로 둔다
                logger.info('근무시간 밖이라 계속 잡니다: %r', text[:30])
                continue
            asleep = False

        if hallway is not None:
            hallway.begin_turn()   # hard-mute proactive greeting for this turn

        if idle is not None:
            idle.stop()

        # Immediate "I heard you": antennas prick up the moment the words are
        # recognised, before any model is called. Without it there is a silent
        # dead zone between the person finishing and the robot reacting.
        if head is not None:
            try:
                head.acknowledge()
            except Exception:
                logger.debug('acknowledge failed', exc_info=True)

        # Bring the arms back up if they auto-settled. After SETTLE_AFTER (3
        # min) idle the arms power down and _arms_held goes False - and nothing
        # re-armed them, so every conversational gesture (talk_accent, idle
        # breathing) was a silent no-op for the rest of the day. Raising them
        # here runs in the background and overlaps the model wait, and the
        # settle timer is re-armed so it cannot fire mid-conversation.
        if motion_handler is not None:
            ex = motion_handler.executor
            try:
                ex._cancel_settle()
                if not ex._arms_held:
                    Thread(target=ex.hold_ready).start()
                else:
                    ex._schedule_settle()
            except Exception:
                logger.debug('arm re-arm failed', exc_info=True)
        if motion_handler is not None:
            motion_handler.executor.stop_idle_arms()

        turns += 1
        print('나:', text)

        # Music: play a short clip and dance to it (offline, local mp3s).
        if music_player is not None:
            from music import wants_music, wants_music_stop
            if wants_music_stop(text):
                if music_player.playing:
                    music_player.stop()
                    speak('네, 음악 끌게요.')
                    last_kind = 'chat'
                    continue
            elif wants_music(text):
                if hallway is not None:
                    hallway.begin_turn()
                if idle is not None:
                    idle.stop()
                if motion_handler is not None:
                    motion_handler.executor.stop_idle_arms()
                speak('네, 한 곡 틀어드릴게요!')
                clip = music_player.play()
                if clip is None:
                    speak('음악 파일이 없네요.')
                else:
                    music_player.wait(timeout=40)
                if turn_logger is not None:
                    turn_logger.log('music', {'text': text, 'clip': clip})
                last_kind = 'chat'
                continue

        # Look at an object: detect it with the camera and turn the neck toward
        # it (offline). First step toward vision-guided grasping.
        if objvis is not None and reachy is not None and wants_object_look(text):
            from object_vision import look_at_object, describe
            if hallway is not None:
                hallway.begin_turn()
            if idle is not None:
                idle.stop()
            if speech is not None:
                speech.start(text='어디 볼까요...')
                speech.wait()
            det = None
            try:
                det = look_at_object(
                    reachy, lambda: grab_frame(reachy, side=camera_side,
                                               camera_index=camera_index), objvis,
                    sign=getattr(idle, 'ATTEND_SIGN', 1.0))
            except Exception:
                logger.exception('look_at_object failed')
            line = describe(det)
            print('리치:', line)
            speak(line)
            if turn_logger is not None:
                turn_logger.log('object_look',
                                {'text': text,
                                 'object': det.get('label') if det else None,
                                 'cx': det.get('cx') if det else None})
            last_kind = 'motion'
            continue

        # Head/neck movement: the Orbita neck moves via gaze, not arm keyframes,
        # so run it LOCALLY (offline, no broker) instead of saying it can't.
        head_g = wants_head_gesture(text) if reachy is not None and head is not None else None
        if head_g is not None:
            from say_and_move import run_head_gesture
            if hallway is not None:
                hallway.begin_turn()
            if idle is not None:
                idle.stop()
            logger.info('Head gesture: %s', head_g)
            if speech is not None:
                speech.start(text='네')
            try:
                run_head_gesture(reachy, head_g)
            except Exception:
                logger.exception('Head gesture failed')
            if speech is not None:
                speech.wait()
            if turn_logger is not None:
                turn_logger.log('head_gesture', {'text': text, 'gesture': head_g})
            last_kind = 'motion'
            continue

        # 실전 파지 라벨: 파지류 동작 후 90초 안의 성패 발화를 데이터로.
        # 판정은 어휘 규칙이면 충분하다 - 오탐이 나도 라벨은 원장에서
        # 사람이 검토한다.
        if (motion_handler is not None and turn_logger is not None
                and time.time() - getattr(motion_handler, 'last_grasp_at', 0)
                < 90.0):
            compact_lbl = text.replace(' ', '')
            good = any(w in compact_lbl for w in
                       ('잡았', '집었', '들었', '성공', '잘했', '옮겼'))
            bad = any(w in compact_lbl for w in
                      ('놓쳤', '떨어', '실패', '못잡', '못집', '안잡', '빠졌'))
            if good != bad:                      # 둘 다면 모호 - 버린다
                turn_logger.log('grasp_label', {
                    'label': 'success' if good else 'fail',
                    'said': text,
                    'command': getattr(motion_handler, 'last_grasp_text', ''),
                    'delay_s': round(time.time()
                                     - motion_handler.last_grasp_at, 1)})
                motion_handler.last_grasp_at = 0.0
                speak('네, 기록해 둘게요!' if good else
                      '아쉽네요, 기록해 두고 연습할게요.')
                last_kind = 'chat'
                continue
        if motion_handler is not None and wants_motion(text, last_kind):
            image = None
            if vision and reachy is not None and wants_vision(text):
                image = capture_view(reachy, side=camera_side, camera_index=camera_index)
            try:
                motion_handler.handle(text, image, head)
            except Exception:
                logger.exception('Motion handling crashed')
                speak('동작을 처리하다가 문제가 생겼어요.')
            last_kind = 'motion'
            continue

        if is_stop_word(text):
            speak('네, 다음에 또 얘기해요!')
            if hallway is not None:
                # Unattended demo: a goodbye ends the conversation, never
                # the program - go back to waiting for the next visitor.
                last_kind = None
                continue
            break

        # Instant path: simple greetings / fixed patterns answer from the note
        # file with no broker round trip at all, so the most common exchanges
        # feel immediate instead of waiting on the CLI.
        if notes is not None:
            note_reply = notes.lookup(text)
            if note_reply:
                logger.info('QuickNote hit - answering instantly')
                if turn_logger is not None:
                    turn_logger.log('note', {'text': text, 'reply': note_reply,
                                             'engine': getattr(listener, 'last_engine', None),
                                             'rms': getattr(listener, 'last_rms', None)})
                print('리치:', note_reply)
                if motion_handler is not None and random.random() < 0.5:
                    try:
                        motion_handler.executor.talk_accent(duration=1.8)
                    except Exception:
                        pass
                speak(note_reply)
                last_kind = 'chat'
                continue

        # Everything meaningful has been routed by now. What is left and looks
        # like mis-heard noise must NOT reach the model: the logs showed the
        # robot earnestly explaining "고사양" eight times to a corridor.
        if looks_like_noise(text, getattr(listener, 'last_engine', None)):
            logger.info('Ignoring noise-like transcript: %r', text)
            if head is not None:
                try:
                    head.acknowledge(duration=0.35)   # 짧게 "응?" 정도만
                except Exception:
                    pass
            if turn_logger is not None:
                turn_logger.log('noise', {'text': text,
                                          'engine': getattr(listener, 'last_engine', None),
                                          'rms': getattr(listener, 'last_rms', None)})
            last_kind = None
            continue

        # Fill the whole wait with pondering sounds, not just one blip: after
        # ack_delay, play fillers back-to-back until the reply arrives, so a
        # slow (4-10s) CLI turn feels like the robot thinking, not lag.
        filler_stop = Event()
        filler_started = Event()
        filler_thread = None
        if fillers and speech is not None:
            ack = type(speech)(voice=speech.voice)

            def filler_loop():
                if filler_stop.wait(ack_delay):
                    return                       # reply came fast, no filler
                filler_started.set()
                last = None
                while not filler_stop.is_set():
                    pick = random.choice(fillers)
                    if len(fillers) > 1:
                        while pick == last:      # never repeat back-to-back
                            pick = random.choice(fillers)
                    last = pick
                    ack.start(wav=pick)
                    # When the reply lands mid-hum, LET THE HUM FINISH instead
                    # of chopping it off - a person says "음..." to the end and
                    # then answers. Cutting mid-word was the jarring part.
                    cut_at = None
                    while ack.is_playing:
                        if filler_stop.is_set():
                            if cut_at is None:
                                cut_at = time.time()
                            # ...but never hold the answer for long.
                            if time.time() - cut_at > FILLER_TAIL_MAX:
                                ack.stop()
                                return
                        time.sleep(0.05)
                    if filler_stop.is_set():
                        return
                    if filler_stop.wait(0.4):     # short gap between hums
                        return

            filler_thread = Thread(target=filler_loop)
            filler_thread.daemon = True
            filler_thread.start()

        image = None
        if vision and reachy is not None and wants_vision(text):
            logger.info('Vision request - capturing a frame')
            image = capture_view(reachy, side=camera_side, camera_index=camera_index)

        if head is not None:
            head.setup()
            head.start_thinking()
        # A slow arm accent while the model works reads as "I am on it" from
        # across the corridor, where antennas alone can be missed.
        if motion_handler is not None:
            try:
                motion_handler.executor.talk_accent(kind='both_open', duration=2.6)
            except Exception:
                logger.debug('thinking accent failed', exc_info=True)

        def finish_fillers():
            """뜸 들이는 소리를 끊고 답할 준비를 한다."""
            if head is not None:
                head.stop_thinking()
            filler_stop.set()
            if filler_thread is not None:
                # Wait for the hum to finish its word (bounded by
                # FILLER_TAIL_MAX inside the loop), then a short beat so the
                # answer does not start on top of it.
                filler_thread.join(timeout=FILLER_TAIL_MAX + 0.5)
                if filler_started.is_set():
                    time.sleep(FILLER_BREATH)

        t_ask = time.time()

        # 문장이 완성되는 대로 받아 바로 말한다. 답을 다 기다렸다 합성하면
        # 생성(1.2초)과 음성 합성(1.5초)이 통째로 직렬이 되는데, 첫 문장은
        # 0.4초면 나오므로 첫 소리까지가 1초쯤 빨라진다.
        # 로봇에 연결되지 않았어도(모터 전원 내림) 스트리밍은 쓴다 -
        # say_sentences_and_move 가 머리 없이 말만 하는 길을 안다.
        first = rest = box = None
        if speech is not None and say_sentences_and_move is not None:
            first, rest, box = stream_reply(client, text, image=image)

        if first is not None:
            latency = round(time.time() - t_ask, 1)
            finish_fillers()
            # Gesture a little while talking - a talking head alone reads as
            # stiff. Fires on the first sentence, so it overlaps the speech.
            if motion_handler is not None and random.random() < 0.45:
                try:
                    motion_handler.executor.talk_accent(duration=2.6)
                except Exception:
                    logger.debug('talk accent failed', exc_info=True)

            spoken = say_sentences_and_move(
                reachy, _sentence_stream(first, rest), speech=speech, head=head,
                on_sentence=lambda line: print('리치:', line))
            reply = ((box or {}).get('reply') or ' '.join(spoken)).strip() or first
            if hallway is not None:
                hallway.note_activity()
        else:
            # 스트리밍이 안 되면(구형 브로커, 첫 문장 지연, 로봇 없이 실행)
            # 예전처럼 한 번에 받아 말한다.
            reply = client.ask_or_fallback(text, image=image)
            latency = round(time.time() - t_ask, 1)
            finish_fillers()
            print('리치:', reply)
            if motion_handler is not None and len(reply) > 12 and random.random() < 0.45:
                try:
                    motion_handler.executor.talk_accent(
                        duration=min(4.0, 1.4 + len(reply) / 55.0))
                except Exception:
                    logger.debug('talk accent failed', exc_info=True)
            speak(reply)

        if turn_logger is not None:
            # latency_s 는 이제 '첫 소리까지', total_s 는 '말을 마칠 때까지'.
            # streamed 플래그가 있어야 예전 로그와 섞어 읽어도 헷갈리지 않는다.
            turn_logger.log('vision_chat' if image else 'chat',
                            {'text': text, 'reply': reply, 'latency_s': latency,
                             'total_s': round(time.time() - t_ask, 1),
                             'streamed': first is not None,
                             'engine': getattr(listener, 'last_engine', None),
                             'rms': getattr(listener, 'last_rms', None)},
                            image_b64=image)
        last_kind = 'chat'


def startup_greeting(reachy, speech, head):
    """Wake up visibly: settle into the home pose, then say hello.

    The arms stay compliant (hanging); only the head and antennas move.
    """
    from say_and_move import head_ready, say_and_move

    try:
        # 모터가 꺼져 있으면 자세를 잡는 건 건너뛰고 인사말만 한다.
        # say_and_move 가 안에서 같은 판단을 한 번 더 한다.
        if head_ready(head):
            head.home(duration=1.5)
        say_and_move(reachy, text='안녕하세요, 리치예요! 편하게 말 걸어주세요.',
                     speech=speech, head=head)
    except Exception:
        logger.exception('Startup greeting failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080', help='broker base url')
    parser.add_argument('--token', help='broker shared secret')
    parser.add_argument('--session', default='voice', help='broker conversation id')
    parser.add_argument('--direct', action='store_true',
                        help='call the Claude API directly instead of the broker')
    parser.add_argument('--api-key', help='Anthropic API key for --direct')
    parser.add_argument('--workspace-id', help='workspace id for --direct')
    parser.add_argument('--language', default='ko-KR', help='STT language')
    parser.add_argument('--mic', type=int, help='microphone device index (see --list-mics)')
    parser.add_argument('--energy', type=int, default=200,
                        help='initial mic sensitivity threshold (auto-calibrated too)')
    parser.add_argument('--energy-floor', type=int, default=100,
                        help='lowest the speech threshold may fall to (default 100)')
    parser.add_argument('--energy-ceil', type=int, default=400,
                        help='highest the speech threshold may rise to; LOWER '
                             'for a quiet room so soft voices still trigger, '
                             'RAISE if it self-triggers on noise (default 400)')
    parser.add_argument('--fixed-energy', type=int, default=None,
                        help='pin the speech threshold to this exact value and '
                             'disable auto-adjust (deterministic sensitivity). '
                             'RAISE to ignore more micro-noise, LOWER to catch '
                             'softer speech. Tune with --stt-test (watch RMS). '
                             'Try ~250 if it picks up too much background noise.')
    parser.add_argument('--stt', default='auto', choices=['auto', 'google', 'vosk'],
                        help="speech engine: 'vosk' offline (flaky internet), "
                             "'google' online (more accurate), 'auto' = vosk if "
                             "its model loads else google (default)")
    parser.add_argument('--vosk-model', default='~/vosk-ko-model',
                        help='path to the offline Vosk Korean model directory')
    parser.add_argument('--phrase-limit', type=float, default=10,
                        help='max seconds per utterance')
    parser.add_argument('--pause', type=float, default=0.5,
                        help='silence (s) that ends an utterance; lower = snappier')
    parser.add_argument('--no-ack', action='store_true',
                        help="don't play the short acknowledgement while thinking")
    parser.add_argument('--ack-delay', type=float, default=0.8,
                        help='seconds to wait before the first pondering hum '
                             'plays; a reply that arrives sooner skips it. '
                             'Hums then chain until the reply is ready.')
    parser.add_argument('--voice', default='ko', help='TTS language')
    parser.add_argument('--io', default='/dev/ttyUSB*', help="serial port template, or 'ws'")
    parser.add_argument('--no-robot', action='store_true',
                        help='no head motion, voice only')
    parser.add_argument('--no-vision', action='store_true',
                        help="don't attach camera frames to see-related questions")
    parser.add_argument('--camera-side', default='left', choices=['right', 'left'],
                        help='which head camera to try first (left = /dev/video0)')
    parser.add_argument('--camera-index', default='main',
                        help="which head camera: 'main' (the sharp one, pinned "
                             "by udev), 'sub' (spare), or a raw V4L2 index")
    parser.add_argument('--collect-people', action='store_true',
                        help='복도를 지나가는 사람을 촬영해 데이터셋으로 모은다 '
                             '(~/reachy_logs/persons, 30일 후 자동 삭제)')
    parser.add_argument('--collect-days', type=int, default=30,
                        help='사람 사진 보관 기간(일). 0이면 삭제 안 함')
    parser.add_argument('--no-presence', action='store_true',
                        help='disable the local face-detection people watcher')
    parser.add_argument('--no-sleep', action='store_true',
                        help='밤에 불이 꺼져도 재우지 않는다 (기본은 재운다)')
    parser.add_argument('--dark-below', type=float, default=0.15,
                        help='이보다 어두우면 불이 꺼진 것으로 본다 (0~1)')
    parser.add_argument('--light-above', type=float, default=0.25,
                        help='이보다 밝으면 불이 켜진 것으로 본다 (0~1)')
    parser.add_argument('--sleep-after', type=float, default=180.0,
                        help='어둡고 사람도 없는 상태가 이만큼 이어지면 잠든다(초)')
    parser.add_argument('--quiet-hours', default='18-9',
                        help="사람이 없으면 밝기와 무관하게 자는 시간대 '시작-끝' "
                             "(기본 20-8 = 근무시간 8~20시 밖). 'off' 면 밝기만 "
                             '본다. 모터를 내려 두면 카메라도 없으므로 이때는 '
                             '이 규칙만 남는다')
    parser.add_argument('--attend', action='store_true',
                        help='turn the head to face detected people while idle '
                             '(needs the presence watcher; check ATTEND_SIGN)')
    parser.add_argument('--hallway', action='store_true',
                        help='unattended demo mode: greet people who appear, '
                             'invite them to talk, react to passers-by '
                             '(implies --attend)')
    parser.add_argument('--gaze-tilt', type=float, default=None,
                        help='baseline vertical gaze offset in m at 0.5m; '
                             'negative looks down (default -0.08)')
    parser.add_argument('--motions', action='store_true',
                        help='connect the arms and let voice commands trigger '
                             'gestures (arms are powered only while moving)')
    parser.add_argument('--no-mirror', action='store_true',
                        help="don't broadcast robot state for the web viewer")
    parser.add_argument('--log-dir', default='~/reachy_logs',
                        help="accumulate turns/motions/frames here ('off' disables)")
    parser.add_argument('--notes', default='auto',
                        help="instant-answer note file for simple patterns; "
                             "'auto' = quick_notes.json next to this script, "
                             "'off' disables")
    parser.add_argument('--stt-corrections', default='auto',
                        help="STT mishear fix note file; 'auto' = "
                             "stt_corrections.json next to this script, "
                             "'off' disables")
    parser.add_argument('--list-mics', action='store_true', help='list microphones and exit')
    parser.add_argument('--stt-test', action='store_true',
                        help='loop STT only: print what was heard, no LLM, no robot')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    if args.list_mics:
        for index, name, is_input in Listener.list_microphones():
            mark = '*' if is_input else ' '
            print('{:3d} {} {}'.format(index, mark, name))
        print("('*' = 입력 가능. --mic 없이 실행하면 ReSpeaker > USB 순으로 자동 선택)")
        return

    listener = Listener(language=args.language, mic_index=args.mic,
                        energy=args.energy, phrase_limit=args.phrase_limit,
                        pause=args.pause, energy_floor=args.energy_floor,
                        energy_ceil=args.energy_ceil, fixed_energy=args.fixed_energy,
                        stt=args.stt, vosk_model=args.vosk_model)

    if args.stt_test:
        print('말해보세요 (Ctrl-C 로 종료):')
        while True:
            text = listener.listen()
            print('->', repr(text))

    from llm_client import BrokerClient, DirectClient
    from say_and_move import Speech, TalkingHead

    if args.direct:
        client = DirectClient(api_key=args.api_key, workspace_id=args.workspace_id)
    else:
        # Above the broker's worst case (60s CLI turn + one retry on a fresh
        # process): at 30s the Pi spoke an offline line while the broker was
        # still successfully answering, then the next request queued behind it.
        # 모델이 1.2초에 답하는데 140초를 기다릴 이유가 없다. 터널이 반쯤
        # 열린 채 멈추면 그 시간만큼 로봇이 통째로 얼어붙는다(로그 최대
        # 90.1초). 20초면 가장 긴 답도 넉넉하고, 넘어가면 곧바로 캔 답변으로
        # 내려가 사람을 세워 두지 않는다. 동작 생성(Qwen)은 ask_motion 이
        # 따로 150초를 쓴다.
        client = BrokerClient(url=args.url, token=args.token, session=args.session,
                              timeout=20)
        if client.health() is None:
            logger.warning('Broker is not answering; replies will be canned lines.')

    speech = Speech(voice=args.voice)
    logger.info('TTS engine: %s', speech.engine)

    fillers = () if args.no_ack else prepare_fillers(speech)

    notes = None
    if args.notes and args.notes != 'off':
        notes_path = args.notes
        if notes_path == 'auto':
            notes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'quick_notes.json')
        if os.path.exists(os.path.expanduser(notes_path)):
            from quick_notes import QuickNotes
            notes = QuickNotes.load(notes_path)
            logger.info('QuickNotes: %d instant-answer patterns from %s',
                        len(notes), notes_path)
        else:
            logger.info('QuickNotes: no note file at %s, skipping', notes_path)

    # 자주 하는 말은 미리 합성해 둔다 (인터넷이 끊겨도 자연스러운 목소리로).
    precache_common_lines(speech, notes)

    if args.stt_corrections and args.stt_corrections != 'off':
        corr_path = args.stt_corrections
        if corr_path == 'auto':
            corr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'stt_corrections.json')
        load_stt_corrections(corr_path)

    reachy = None
    head = None
    idle = None
    neck_hold = None
    watcher = None
    sleeper = None
    hallway = None
    objvis = None
    music_player = None
    person_db = None

    motion_handler = None

    turn_logger = None
    if args.log_dir and args.log_dir != 'off':
        turn_logger = TurnLogger(args.log_dir)
        logger.info('Turn log: %s', turn_logger.path)

    # 밤에는 재운다. 빈 사무실에서 로봇이 혼자 인사하고 팔을 움직이면 무섭다.
    # 통신(터널·브로커·ssh·뷰어·로그)은 자는 동안에도 그대로 열려 있다.
    #
    # 카메라(watcher)는 나중에 붙인다. 모터 전원을 내려 두면 로봇 연결이
    # 통째로 실패해 카메라도 없는데, 그때가 바로 재워야 할 밤이다. 그래서
    # 눈이 없어도 시간대만으로 잘 수 있게 먼저 만들어 둔다.
    if not args.no_sleep:
        from sleep_mode import SleepWatcher
        quiet_hours = None if args.quiet_hours == 'off' else tuple(
            int(x) for x in args.quiet_hours.split('-'))
        sleeper = SleepWatcher(None,
                               dark_below=args.dark_below,
                               light_above=args.light_above,
                               quiet_for=args.sleep_after,
                               quiet_hours=quiet_hours)
        logger.info('수면 모드 켬 (근무시간 %s. 그 밖이거나 밝기 %.2f 아래인데 '
                    '사람이 %.0f분 없으면 잠들고, 불이 켜지거나 사람이 보이거나 '
                    '말을 걸면 깹니다. 통신은 자는 동안에도 그대로입니다)',
                    '%d~%d시' % (quiet_hours[1], quiet_hours[0])
                    if quiet_hours else '(시간대 안 씀)',
                    args.dark_below, args.sleep_after / 60.0)

    if not args.no_robot:
        from reachy import Reachy, parts

        import say_and_move
        from say_and_move import IdleMotion

        if args.gaze_tilt is not None:
            say_and_move.GAZE_TILT = args.gaze_tilt

        # 모터 전원을 내린 채로 두는 운용이라(퇴근 시), 이 연결은 실패할 수
        # 있다. Luos 게이트(USB)는 살아 있어도 다이나믹셀이 응답하지 않으면
        # LuosModuleNotFoundError 가 난다. 예전에는 그대로 main() 을 뚫고
        # 나가 서비스가 12초마다 재시작을 되풀이했다(밤새 로그가 그것으로
        # 찼다). 이제는 소리만 내는 모드로 내려가 조용히 계속 돈다.
        try:
            if args.motions:
                # 소리만 모드에서의 모터 자동복구가 쓸 설정 (_run 은 args
                # 스코프 밖이라 전역으로 넘긴다)
                global MOTOR_RECOVERY_IO
                if not args.no_robot:
                    MOTOR_RECOVERY_IO = args.io
                # Arms + head, with this robot's custom hands. The arms stay
                # compliant; the executor powers them per-gesture.
                from base_pose import connect
                reachy = connect(io=args.io, with_head=True)
            else:
                reachy = Reachy(head=parts.Head(io=args.io))
        except Exception as e:
            logger.warning('로봇에 연결하지 못했습니다 (%s: %s). 모터가 꺼져 '
                           '있는 것으로 보고, 움직임 없이 소리만 내는 모드로 '
                           '계속합니다.', type(e).__name__, e)
            reachy = None

    if reachy is not None:
        head = TalkingHead(reachy)
        head.setup()
        idle = IdleMotion(reachy)

        # 목을 계속 붙잡는다. connect() 직후 목은 풀린 채로 오고(실측: 디스크
        # 셋 다 compliant), 풀린 목은 중력에 처졌다가 누가 다시 굳히는 순간
        # 마지막 목표값으로 확 돌아간다 - 대기 중에 이따금 '팍' 하고 움직이는
        # 정체가 이것이다. 대화 루프도 강성을 다시 잡지만 한 바퀴에 한 번뿐이라,
        # 아무도 말을 걸지 않는 복도에서는 그 간격이 십수 초까지 벌어진다.
        from say_and_move import NeckHold
        try:
            neck_hold = NeckHold(reachy).start()
        except Exception:
            logger.exception('목 붙잡기를 시작하지 못했습니다')

        # Local people awareness: a background face watcher so the robot knows
        # someone is there (free, no LLM). With --attend it also turns the head
        # to face them while idle.
        if not args.no_presence and not args.no_vision:
            try:
                from presence import PresenceWatcher
                watcher = PresenceWatcher(
                    lambda: grab_frame(reachy, side=args.camera_side,
                                       camera_index=args.camera_index),
                    interval=0.8 if args.hallway else 1.2)
                watcher.start()
                if args.attend or args.hallway:
                    idle.watcher = watcher
                if sleeper is not None:
                    sleeper.watcher = watcher   # 이제 밝기와 사람을 볼 수 있다
                    logger.info('Head will attend to detected people')

                # 카메라가 머리에 달려 있어서, 목이 도는 동안에는 화면 전체가
                # 흐른다(사람이 지나가는 것보다도 크게). 목 디스크의 '실측'
                # 위치를 넘겨주면, 화면이 안정된 순간에만 움직임을 사람으로 본다.
                # 명령을 넣는 코드 경로를 계측하는 방식은 한 군데만 놓쳐도
                # 조용히 틀린다 - 실제로 그래서 빈 복도 사진이 쌓였다.
                try:
                    disks = reachy.head.neck.disks
                    watcher.head_pose = lambda: tuple(d.rot_position for d in disks)
                except Exception:
                    logger.warning('목 위치를 읽을 수 없어 움직임 촬영은 끕니다',
                                   exc_info=True)
                check_sdk_camera(args.camera_index)

                # 지나가는 사람을 데이터셋으로 모은다. 저장 여부(간격·장수·용량·
                # 보관기간)는 PersonDB 가 스스로 판단하므로 여기서는 넘겨만 준다.
                if args.collect_people:
                    try:
                        from person_db import PersonDB
                        person_db = PersonDB(
                            retention_days=args.collect_days,
                            # 버스트 최상 선택용 추가 프레임 (감시 스레드에서만
                            # 불린다 - 카메라 락이 겹침을 막는다)
                            grab=lambda: grab_frame(reachy, side=args.camera_side,
                                                    camera_index=args.camera_index))
                        # 대화 기록에 '지금 앞에 있는 방문'을 붙인다. 사진과
                        # 대화가 이어져야, 나중에 사람을 알아봤을 때 그 사람이
                        # 전에 무슨 말을 했는지도 알 수 있다.
                        if turn_logger is not None:
                            turn_logger.visit_of = (
                                lambda _db=person_db: _db.visit_id)

                        def _collect(frame, box_rel, person_rel=None,
                                     conf=None, still=True):
                            if frame is None:
                                return
                            h, w = frame.shape[:2]
                            face = None
                            if box_rel is not None:
                                x, y, bw, bh = box_rel
                                face = (x * w, y * h, bw * w, bh * h)
                            # 사람 상자는 (x1, y1, x2, y2) 비율로 온다.
                            person = None
                            if person_rel is not None:
                                x1, y1, x2, y2 = person_rel
                                person = (max(0.0, x1) * w, max(0.0, y1) * h,
                                          (min(1.0, x2) - max(0.0, x1)) * w,
                                          (min(1.0, y2) - max(0.0, y1)) * h)
                            person_db.add(frame, face=face, person=person,
                                          person_conf=conf, still=still)

                        watcher.on_person = _collect
                        st = person_db.stats()
                        logger.info('People collection ON (모은 사진 %d장, 보관 %d일)',
                                    st.get('photos', 0), args.collect_days)
                    except Exception:
                        logger.exception('People collection failed to start')
            except Exception:
                logger.exception('Presence watcher failed to start')
                watcher = None

        # Object attention: look at what someone shows the robot (offline DNN).
        if not args.no_vision:
            here = os.path.dirname(os.path.abspath(__file__))
            try:
                from object_vision import ObjectVision
                ov = ObjectVision(os.path.join(here, 'mnssd.prototxt'),
                                  os.path.join(here, 'mnssd.caffemodel'))
                if ov.available():
                    objvis = ov
                    logger.info('Object vision enabled')

                    # 복도에서는 얼굴 검출보다 사람 전체 검출이 훨씬 잘 맞는다.
                    # 실측: 크게 찍힌 정면 얼굴을 Haar 는 어떤 설정으로도 못
                    # 잡았는데(역광·안경·화면 기울기 13도), 같은 사진을 SSD 는
                    # 0.98 로 잡았다. 모델은 이미 물체 인식용으로 올라와 있으니
                    # 그대로 나눠 쓴다.
                    if watcher is not None and (args.collect_people or args.hallway):
                        def _person_conf(frame, _ov=ov):
                            """가장 확실한 사람의 (신뢰도, 상자) — 없으면 (0, None)."""
                            best, box = 0.0, None
                            for d in _ov.detect(frame):
                                if d['label'] == 'person' and d['conf'] > best:
                                    best = d['conf']
                                    box = (d['x1'], d['y1'], d['x2'], d['y2'])
                            return best, box

                        watcher.detect_person = _person_conf
                        logger.info('사람 검출(SSD)을 감시에 연결했습니다')
                else:
                    logger.info('Object vision model not found - skipping '
                                '(install with ops/pi/install_object_vision.sh)')
            except Exception:
                logger.exception('Object vision failed to init')

        # Music: short royalty-free clips the robot can play and dance to.
        try:
            from music import MusicPlayer, available as music_available
            if music_available():
                music_player = MusicPlayer(reachy=reachy)
                logger.info('Music enabled (%d clips)', len(music_available()))
            else:
                logger.info('No music clips found - skipping')
        except Exception:
            logger.exception('Music init failed')

        if args.motions:
            if not hasattr(client, 'ask_motion'):
                parser.error('--motions requires the broker (not --direct)')
            motion_handler = MotionHandler(reachy, client, speech,
                                           turn_logger=turn_logger)
            logger.info('Motion handling enabled')
            if music_player is not None:
                music_player.executor = motion_handler.executor

        if args.io != 'ws' and not args.no_mirror:
            # Real hardware: broadcast commanded pose so sim_viewer can show
            # a live twin. (With io='ws' the sim server owns port 6171.)
            try:
                from state_mirror import StateMirror
                StateMirror(reachy).start()
            except Exception:
                logger.exception('State mirror failed to start (viewer only)')

        # 이미 캄캄한 사무실에서 부팅했다면 켜자마자 인사하지 않는다.
        if sleeper is not None and sleeper.update():
            logger.info('조용히 기동합니다 (기동 인사 생략)')
        else:
            startup_greeting(reachy, speech, head)

        if motion_handler is not None:
            # Arms rise into the ready stance (slightly forward, soft elbow
            # bend) and hold it at low torque; they settle down on idle.
            motion_handler.executor.hold_ready()

        if args.hallway:
            if watcher is None:
                parser.error('--hallway needs the presence watcher '
                             '(do not combine with --no-presence/--no-vision)')
            from hallway import HallwayGreeter

            wave_segments = None
            executor = None
            if motion_handler is not None:
                executor = motion_handler.executor
                try:
                    from motion_exec import validate
                    from motion_presets import PRESETS
                    wave_segments = validate(
                        PRESETS['wave']['moves'],
                        {j: 0.0 for j in executor.available_joints()},
                        executor.available_joints())
                except Exception:
                    logger.exception('Greeting wave unavailable (voice-only greeting)')

            hallway = HallwayGreeter(
                watcher, speech, head, idle,
                person_db=person_db,
                executor=executor,
                wave_segments=wave_segments,
                turn_logger=turn_logger,
                # Photograph visitors as they arrive so the logs accumulate
                # real scenes for later analysis/improvement.
                sleeper=sleeper,
                snap=(lambda: capture_view(reachy, side=args.camera_side,
                                           camera_index=args.camera_index))
                if not args.no_vision else None)
            hallway.start()
            logger.info('Hallway demo mode on - will greet visitors')

    try:
        run_loop(listener, client, reachy=reachy, speech=speech, head=head,
                 fillers=fillers, ack_delay=args.ack_delay, idle=idle,
                 vision=not args.no_vision, camera_side=args.camera_side,
                 camera_index=args.camera_index, motion_handler=motion_handler,
                 turn_logger=turn_logger, notes=notes, hallway=hallway,
                 objvis=objvis, music_player=music_player, sleeper=sleeper)
    except KeyboardInterrupt:
        print()
    finally:
        if neck_hold is not None:
            neck_hold.stop()
        if hallway is not None:
            hallway.stop()
        if watcher is not None:
            watcher.stop()
        if motion_handler is not None:
            motion_handler.executor.shutdown()
        if reachy is not None:
            reachy.close()


if __name__ == '__main__':
    main()
