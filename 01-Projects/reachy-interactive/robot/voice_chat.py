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

# When the sentence sounds like it is about what the robot can SEE,
# a camera frame is attached to the request.
VISION_WORDS = ('보여', '보이', '뭐가 있', '이게 뭐', '저게 뭐', '누구', '누가',
                '몇 명', '몇명', '무슨 색', '어떻게 생겼', '읽어', '앞에')


# Motion routing: a body part must co-occur with an action verb, or a gesture
# noun with a command-ish tail. "팔 아파" has the noun but no verb -> chat.
BODY_WORDS = ('팔', '손', '고개', '안테나', '머리', '몸',
              '물건', '이거', '저거', '그거', '컵', '공', '펜', '상자', '보관함', '트레이')
ACTION_STEMS = ('들어', '들고', '들어서', '드', '올려', '올리', '내려', '내리', '흔들',
                '움직', '벌려', '벌리', '접어', '돌려', '돌리', '가리켜', '가리키',
                '집어', '집게', '놓', '옮겨', '건네', '넣어', '꺼내', '굴려', '빙글',
                '펴', '뻗', '세워', '펼쳐', '펼', '휘둘', '휘저', '뻗쳐')
# Gesture nouns strong enough to trigger on their own (no verb needed) - a
# bare "만세" or "만세 다시" is unambiguously a motion request.
GESTURE_WORDS = ('인사', '만세', '박수', '손뼉', '하이파이브', '브이', '악수')
GESTURE_WEAK = ('춤', '환영')   # need a command tail (춤=대화 맥락도 흔함)
# Compounds that contain a gesture noun but are not motion requests.
GESTURE_STOPWORDS = ('인사말', '인사법', '인사드', '박수갈채', '만세력')
COMMAND_TAILS = ('해봐', '해 봐', '해줘', '해 줘', '하자', '해라', '춰봐', '해보',
                 '볼래', '할래', '수있', '다시', '또', '봐', '줘', '자')


def wants_motion(text):
    """Check if the request asks for a physical gesture."""
    compact = text.replace(' ', '')

    if any(b in compact for b in BODY_WORDS) and any(a in compact for a in ACTION_STEMS):
        return True
    # Strong gesture nouns trigger alone; weak ones still need a command tail.
    if any(g in compact for g in GESTURE_WORDS) \
            and not any(w in compact for w in GESTURE_STOPWORDS):
        return True
    if any(g in compact for g in GESTURE_WEAK) and any(t in compact for t in COMMAND_TAILS):
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
        self._seq = 0

    def log(self, kind, data, image_b64=None):
        """Record one event; never raises."""
        import base64
        import json

        try:
            entry = {'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'kind': kind}
            entry.update(data)

            if image_b64:
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
    """Route a motion request through the opus session and execute it safely.

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
        """Full motion turn: ack -> opus -> validate -> speak + move."""
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
            # the opus session answered conversationally; just speak it.
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


def capture_view(reachy, side='left', width=640, quality=70, camera_index=0):
    """Grab one frame of the robot's view as base64 jpeg, or None on failure.

    Different reachy checkouts expose the camera differently, so this tries
    them in order: left/right_camera objects, head.get_image(), and finally
    opening the video device directly with OpenCV.
    """
    import base64

    try:
        import cv2 as cv

        head = reachy.head
        img = None

        # 1) left_camera / right_camera attributes (this repo's head.py)
        other = 'right' if side == 'left' else 'left'
        for name in (side, other):
            camera = getattr(head, name + '_camera', None)
            if camera is None:
                continue
            success, frame = camera.read()
            if success and frame is not None:
                img = frame
                break
            logger.warning('Camera read failed (%s)', name)

        # 2) head.get_image() (the variant installed on the robot's Pi)
        if img is None and hasattr(head, 'get_image'):
            try:
                frame = head.get_image()
                if frame is not None and len(frame) > 0:
                    img = frame
            except Exception:
                logger.warning('head.get_image() failed', exc_info=True)

        # 3) open the device directly - always available while video0 works
        if img is None:
            capture = cv.VideoCapture(camera_index)
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

    def __init__(self, language='ko-KR', mic_index=None, energy=300, phrase_limit=10,
                 pause=0.5):
        with quiet_stderr():
            import speech_recognition as sr

        self.sr = sr
        self.language = language
        self.phrase_limit = phrase_limit

        if mic_index is None:
            mic_index = self.pick_best_mic()
            if mic_index is not None:
                logger.info('Auto-selected microphone index %d', mic_index)

        self._check_input_device(mic_index)

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy
        self.recognizer.dynamic_energy_threshold = True
        # How much silence marks the end of an utterance. The 0.8s default
        # adds nearly half a second of dead air to every single turn.
        self.recognizer.pause_threshold = pause
        self.recognizer.non_speaking_duration = min(0.5, pause)

        with quiet_stderr():
            self.microphone = sr.Microphone(device_index=mic_index)

            # Calibrate for room noise once, so speech detection is reliable.
            with self.microphone as source:
                logger.info('Calibrating for ambient noise...')
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

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

    def listen(self):
        """Wait for one utterance and return its text.

        Returns:
            str: recognized text
            None: heard something but could not understand it
            '': network/service problem (treat as offline)
        """
        with quiet_stderr():
            with self.microphone as source:
                logger.info('Listening...')
                try:
                    audio = self.recognizer.listen(
                        source, timeout=20, phrase_time_limit=self.phrase_limit)
                except self.sr.WaitTimeoutError:
                    # Nothing said for a while; reopen the stream and keep waiting.
                    logger.info('(no speech for 20s)')
                    return None

        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
        except self.sr.UnknownValueError:
            logger.info('Heard something, could not understand it')
            return None
        except self.sr.RequestError as e:
            logger.warning('STT service unavailable: %s', e)
            return ''

        text = _normalize_name(text)
        logger.info('Heard: %s', text)
        return text


# STT often mangles the wake-name '리치'. Only fix clear name-mishears; leave
# real words like '위치'(position) alone unless they stand alone as address.
_NAME_MISHEARS = ('다비치', '리치야', '리치아', '루치아', '유치하', '유치아',
                  '니치', '이치', '릿지', '리찌', '리취', '리치가', '리치는')


def _normalize_name(text):
    for w in _NAME_MISHEARS:
        text = text.replace(w, '리치')
    # Address-position mishears at sentence start -> '리치' (leave mid-sentence
    # '위치'/'유치' alone, which are real words).
    stripped = text.strip()
    for w in ('위치', '유치', '지하'):
        if stripped == w or stripped.startswith(w + ' '):
            text = text.replace(w, '리치', 1)
            break
    return text


def is_stop_word(text):
    """Check if the user asked to end the conversation."""
    compact = text.replace(' ', '')
    return any(w.replace(' ', '') in compact for w in STOP_WORDS)


def prepare_fillers(speech, cache_dir='/tmp/reachy_fillers'):
    """Pre-synthesize short acknowledgements ("um...") for instant playback.

    Played when the LLM+TTS wait drags on, so it feels like pondering instead
    of lag. The cache key includes the engine and voice: switching TTS engines
    regenerates the fillers instead of replaying stale ones.
    """
    # Neutral hums only: "잠깐만요" sounds wrong after a greeting,
    # while "음..." fits any kind of question.
    phrases = ['음...', '음, 어디 보자...']

    os.makedirs(cache_dir, exist_ok=True)

    voice = getattr(speech, 'edge_voice', '') if speech.engine == 'edge' else speech.voice
    slug = '{}-{}'.format(speech.engine, ''.join(c for c in voice if c.isalnum()))

    paths = []
    for n, phrase in enumerate(phrases):
        path = os.path.join(cache_dir, 'filler_{}_{}.mp3'.format(slug, n))
        if os.path.exists(path) or speech.synthesize_to_file(phrase, path):
            paths.append(path)

    logger.info('%d filler sounds ready (%s)', len(paths), slug)
    return paths


def run_loop(listener, client, reachy=None, speech=None, head=None, fillers=(),
             ack_delay=1.2, idle=None, vision=False, camera_side='left',
             camera_index=0, motion_handler=None, turn_logger=None):
    """Main conversation loop. Blocks until a stop word or Ctrl-C."""
    import random

    from say_and_move import say_and_move

    def speak(line):
        if reachy is not None:
            say_and_move(reachy, text=line, speech=speech, head=head)
        elif speech is not None:
            speech.start(text=line)
            speech.wait()

    try:
        _run(listener, client, reachy, speech, head, fillers, ack_delay, idle,
             say_and_move, random, speak, vision, camera_side, camera_index,
             motion_handler, turn_logger)
    finally:
        if idle is not None:
            idle.stop()


def _run(listener, client, reachy, speech, head, fillers, ack_delay, idle,
         say_and_move, random, speak, vision=False, camera_side='left',
         camera_index=0, motion_handler=None, turn_logger=None):
    while True:
        # Re-assert neck stiffness each loop: the Orbita disks can thermally
        # cut torque while holding the head, and go limp until re-gripped.
        if reachy is not None and getattr(reachy, 'head', None) is not None:
            try:
                reachy.head.compliant = False
            except Exception:
                pass

        # Small random motions while waiting keep the robot looking alive.
        if idle is not None:
            idle.start()

        text = listener.listen()

        if text is None:
            # Silence timeout or not understood - just keep listening quietly.
            continue

        if text == '':
            speak('지금 인터넷이 불안정한 것 같아요.')
            time.sleep(2)
            continue

        if idle is not None:
            idle.stop()

        print('나:', text)

        if motion_handler is not None and wants_motion(text):
            image = None
            if vision and reachy is not None and wants_vision(text):
                image = capture_view(reachy, side=camera_side, camera_index=camera_index)
            try:
                motion_handler.handle(text, image, head)
            except Exception:
                logger.exception('Motion handling crashed')
                speak('동작을 처리하다가 문제가 생겼어요.')
            continue

        if is_stop_word(text):
            speak('네, 다음에 또 얘기해요!')
            break

        ack = None
        ack_timer = None
        if fillers and speech is not None:
            # "um..." only when the wait drags on - a reply that arrives fast
            # needs no filler, and an instant one right after the user stops
            # talking feels too eager.
            from threading import Timer

            ack = type(speech)(voice=speech.voice)
            ack_timer = Timer(ack_delay, lambda: ack.start(wav=random.choice(fillers)))
            ack_timer.start()

        image = None
        if vision and reachy is not None and wants_vision(text):
            logger.info('Vision request - capturing a frame')
            image = capture_view(reachy, side=camera_side, camera_index=camera_index)

        if head is not None:
            head.setup()
            head.start_thinking()

        reply = client.ask_or_fallback(text, image=image)

        if head is not None:
            head.stop_thinking()

        if ack_timer is not None:
            ack_timer.cancel()
        if ack is not None:
            ack.stop()

        if turn_logger is not None:
            turn_logger.log('vision_chat' if image else 'chat',
                            {'text': text, 'reply': reply}, image_b64=image)

        print('리치:', reply)
        speak(reply)


def startup_greeting(reachy, speech, head):
    """Wake up visibly: settle into the home pose, then say hello.

    The arms stay compliant (hanging); only the head and antennas move.
    """
    from say_and_move import say_and_move

    try:
        head.setup()
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
    parser.add_argument('--energy', type=int, default=300,
                        help='mic sensitivity threshold; raise if it self-triggers')
    parser.add_argument('--phrase-limit', type=float, default=10,
                        help='max seconds per utterance')
    parser.add_argument('--pause', type=float, default=0.5,
                        help='silence (s) that ends an utterance; lower = snappier')
    parser.add_argument('--no-ack', action='store_true',
                        help="don't play the short acknowledgement while thinking")
    parser.add_argument('--ack-delay', type=float, default=1.2,
                        help='seconds to wait before the acknowledgement plays; '
                             'a reply that arrives sooner skips it entirely')
    parser.add_argument('--voice', default='ko', help='TTS language')
    parser.add_argument('--io', default='/dev/ttyUSB*', help="serial port template, or 'ws'")
    parser.add_argument('--no-robot', action='store_true',
                        help='no head motion, voice only')
    parser.add_argument('--no-vision', action='store_true',
                        help="don't attach camera frames to see-related questions")
    parser.add_argument('--camera-side', default='left', choices=['right', 'left'],
                        help='which head camera to try first (left = /dev/video0)')
    parser.add_argument('--camera-index', type=int, default=0,
                        help='V4L2 device index for the direct-OpenCV fallback')
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
                        pause=args.pause)

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
        client = BrokerClient(url=args.url, token=args.token, session=args.session,
                              timeout=30)
        if client.health() is None:
            logger.warning('Broker is not answering; replies will be canned lines.')

    speech = Speech(voice=args.voice)
    logger.info('TTS engine: %s', speech.engine)

    fillers = () if args.no_ack else prepare_fillers(speech)

    reachy = None
    head = None
    idle = None

    motion_handler = None

    turn_logger = None
    if args.log_dir and args.log_dir != 'off':
        turn_logger = TurnLogger(args.log_dir)
        logger.info('Turn log: %s', turn_logger.path)

    if not args.no_robot:
        from reachy import Reachy, parts

        import say_and_move
        from say_and_move import IdleMotion

        if args.gaze_tilt is not None:
            say_and_move.GAZE_TILT = args.gaze_tilt

        if args.motions:
            # Arms + head, with this robot's custom hands. The arms stay
            # compliant; the executor powers them per-gesture.
            from base_pose import connect
            reachy = connect(io=args.io, with_head=True)
        else:
            reachy = Reachy(head=parts.Head(io=args.io))

        head = TalkingHead(reachy)
        head.setup()
        idle = IdleMotion(reachy)

        if args.motions:
            if not hasattr(client, 'ask_motion'):
                parser.error('--motions requires the broker (not --direct)')
            motion_handler = MotionHandler(reachy, client, speech,
                                           turn_logger=turn_logger)
            logger.info('Motion handling enabled')

        if args.io != 'ws' and not args.no_mirror:
            # Real hardware: broadcast commanded pose so sim_viewer can show
            # a live twin. (With io='ws' the sim server owns port 6171.)
            try:
                from state_mirror import StateMirror
                StateMirror(reachy).start()
            except Exception:
                logger.exception('State mirror failed to start (viewer only)')

        startup_greeting(reachy, speech, head)

        if motion_handler is not None:
            # Arms rise into the ready stance (slightly forward, soft elbow
            # bend) and hold it at low torque; they settle down on idle.
            motion_handler.executor.hold_ready()

    try:
        run_loop(listener, client, reachy=reachy, speech=speech, head=head,
                 fillers=fillers, ack_delay=args.ack_delay, idle=idle,
                 vision=not args.no_vision, camera_side=args.camera_side,
                 camera_index=args.camera_index, motion_handler=motion_handler,
                 turn_logger=turn_logger)
    except KeyboardInterrupt:
        print()
    finally:
        if motion_handler is not None:
            motion_handler.executor.shutdown()
        if reachy is not None:
            reachy.close()


if __name__ == '__main__':
    main()
