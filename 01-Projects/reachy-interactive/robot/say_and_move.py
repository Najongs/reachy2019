"""Make Reachy talk while moving its head.

Audio playback runs in its own subprocess, so the head keeps moving for as long
as the sound lasts instead of freezing until it is done.

The head motion writes the Orbita disk targets directly instead of calling
look_at() in a loop: look_at() spawns a trajectory thread per call, which piles
up and stutters when called at high frequency (same trick as the flyers demo).

Usage:
    python say_and_move.py "안녕하세요"              # espeak-ng 로 말하기
    python say_and_move.py --wav hello.wav          # 미리 녹음한 파일 재생
    python say_and_move.py "안녕" --audio-only      # 로봇 없이 음성만 확인
    python say_and_move.py --duration 5             # 음성 없이 동작만 확인
"""

import argparse
import logging
import os
import shutil
import subprocess
import time

import numpy as np


logger = logging.getLogger(__name__)


# Baseline vertical gaze offset (in m at the standard 0.5 m gaze distance).
# Negative looks down. On this robot the mechanical neutral points well above
# the horizon, and it sits on an elevated stand - the user confirmed that a
# gaze of about z -0.3 keeps both the person and the robot's own hands in
# frame. Tune with voice_chat --gaze-tilt.
GAZE_TILT = -0.15


# Words the TTS engines mispronounce, fixed just before synthesis.
# The LLM is also told to write these in hangul, but replies slip sometimes.
PRONUNCIATIONS = {
    'KIRO': '키로',
    'Kiro': '키로',
    'kiro': '키로',
    'Reachy': '리치',
    'reachy': '리치',
    'AI': '에이아이',
}


def speakable(text):
    """Rewrite words the TTS would garble into phonetic hangul."""
    for word, reading in PRONUNCIATIONS.items():
        text = text.replace(word, reading)
    return text


# --- Speech -----------------------------------------------------------------

class Speech(object):
    """Play a sound in the background so the robot can move while talking.

    Engines (auto picks the best installed one, in this order):
        'edge'   - Microsoft neural voices via `pip3 install edge-tts`.
                   Best Korean quality, voice/rate selectable, needs internet.
        'gtts'   - Google TTS. Natural Korean, needs internet and
                   `pip3 install gTTS` + `sudo apt install mpg123`.
        'espeak' - offline, always works, but Korean sounds very robotic.

    Args:
        voice (str): language/voice code (e.g. 'ko', 'en')
        speed (int): espeak-ng speed in words per minute (espeak only)
        pitch (int): espeak-ng pitch, 0-99 (espeak only)
        engine (str): 'auto', 'edge', 'gtts' or 'espeak'
        edge_voice (str): edge-tts voice name (e.g. 'ko-KR-SunHiNeural' warm
            female, 'ko-KR-InJoonNeural' male)
        edge_rate (str): edge-tts speaking rate (e.g. '-8%' a touch slower)
    """

    def __init__(self, voice='ko', speed=150, pitch=50, engine='auto',
                 edge_voice='ko-KR-SunHiNeural', edge_rate='-8%'):
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self.edge_voice = edge_voice
        self.edge_rate = edge_rate

        self._proc = None
        self._worker = None
        self._tmp_path = None

        self.engine = self._resolve_engine(engine)

    @staticmethod
    def _gtts_usable():
        try:
            import gtts  # noqa: F401
        except ImportError:
            return False
        return shutil.which('mpg123') is not None

    @staticmethod
    def _edge_usable():
        return (shutil.which('edge-tts') is not None
                and shutil.which('mpg123') is not None)

    def _resolve_engine(self, engine):
        if engine == 'auto':
            if self._edge_usable():
                return 'edge'
            if self._gtts_usable():
                return 'gtts'
            return 'espeak'

        if engine == 'edge' and not self._edge_usable():
            raise RuntimeError(
                'edge engine unavailable. Run "pip3 install edge-tts" and '
                '"sudo apt install mpg123".'
            )
        if engine == 'gtts' and not self._gtts_usable():
            raise RuntimeError(
                'gtts engine unavailable. Run "pip3 install gTTS" and '
                '"sudo apt install mpg123".'
            )
        return engine

    @staticmethod
    def available_backends():
        """List the audio tools actually installed on this machine."""
        found = {
            name: shutil.which(name)
            for name in ('edge-tts', 'espeak-ng', 'espeak', 'aplay', 'mpg123')
            if shutil.which(name) is not None
        }
        try:
            import gtts
            found['gtts'] = getattr(gtts, '__version__', 'installed')
        except ImportError:
            pass
        return found

    def _tts_command(self, text):
        for name in ('espeak-ng', 'espeak'):
            path = shutil.which(name)
            if path is not None:
                return [path, '-v', self.voice, '-s', str(self.speed), '-p', str(self.pitch), text]

        raise RuntimeError(
            'No TTS engine found. Install one with "sudo apt install espeak-ng", '
            'or use a pre-recorded file with --wav.'
        )

    def _wav_command(self, path):
        if path.endswith('.mp3'):
            player = shutil.which('mpg123')
            if player is None:
                raise RuntimeError('"mpg123" not found. Install it with "sudo apt install mpg123".')
            return [player, '-q', path]

        player = shutil.which('aplay')
        if player is None:
            raise RuntimeError('"aplay" not found. Install it with "sudo apt install alsa-utils".')

        return [player, '-q', path]

    def synthesize_to_file(self, text, path):
        """Pre-render a phrase to an mp3 for instant playback later.

        Only works on the gtts engine (espeak is instant anyway). Returns the
        path, or None when synthesis is unavailable or fails.
        """
        text = speakable(text)

        if self.engine == 'edge':
            try:
                result = subprocess.run(
                    [shutil.which('edge-tts'),
                     '--voice', self.edge_voice, '--rate={}'.format(self.edge_rate),
                     '--text', text, '--write-media', path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                    timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning('edge-tts pre-synthesis timed out')
                result = subprocess.CompletedProcess([], 1, stderr='timeout')
            if result.returncode == 0 and os.path.getsize(path) > 0:
                return path
            logger.warning('edge-tts pre-synthesis failed: %s',
                           (result.stderr or '').strip()[-300:] or '(no stderr)')
            # fall through to gtts below

        if self.engine not in ('edge', 'gtts') or not self._gtts_usable():
            return None

        try:
            from gtts import gTTS
            gTTS(text=text, lang=self.voice).save(path)
            return path
        except Exception:
            logger.warning('Could not pre-synthesize %r', text, exc_info=True)
            return None

    def _start_gtts(self, text):
        """Synthesize with Google TTS, then play the mp3.

        Synthesis needs a network round-trip (~0.5-1s), so it runs in a thread:
        is_playing turns True immediately and the head can start moving.
        """
        import tempfile
        from threading import Thread

        from gtts import gTTS

        fd, self._tmp_path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)

        def synth_and_play():
            try:
                gTTS(text=text, lang=self.voice).save(self._tmp_path)
            except Exception:
                logger.exception('gTTS synthesis failed, falling back to espeak')
                try:
                    cmd = self._tts_command(text)
                except RuntimeError:
                    return
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return

            self._proc = subprocess.Popen(
                [shutil.which('mpg123'), '-q', self._tmp_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self._worker = Thread(target=synth_and_play)
        self._worker.daemon = True
        self._worker.start()

    def _start_edge(self, text):
        """Synthesize with edge-tts (CLI), then play the mp3.

        Runs in a thread like the gtts path: is_playing turns True immediately
        so the head can start moving during the network round-trip.
        """
        import tempfile
        from threading import Thread

        fd, self._tmp_path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)

        def synth_and_play():
            try:
                result = subprocess.run(
                    [shutil.which('edge-tts'),
                     '--voice', self.edge_voice, '--rate={}'.format(self.edge_rate),
                     '--text', text, '--write-media', self._tmp_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                    timeout=15)
            except subprocess.TimeoutExpired:
                # A half-open connection must never freeze the robot.
                logger.warning('edge-tts timed out, falling back')
                result = subprocess.CompletedProcess([], 1, stderr='timeout')

            if result.returncode != 0 or os.path.getsize(self._tmp_path) == 0:
                logger.warning('edge-tts failed: %s',
                               (result.stderr or '').strip()[-300:] or '(no stderr)')

                # Second choice: gtts, which also plays an mp3.
                if self._gtts_usable():
                    try:
                        from gtts import gTTS
                        gTTS(text=text, lang=self.voice).save(self._tmp_path)
                        self._proc = subprocess.Popen(
                            [shutil.which('mpg123'), '-q', self._tmp_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return
                    except Exception:
                        logger.warning('gtts fallback failed too', exc_info=True)

                # Last resort: espeak.
                try:
                    cmd = self._tts_command(text)
                except RuntimeError:
                    return
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return

            self._proc = subprocess.Popen(
                [shutil.which('mpg123'), '-q', self._tmp_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self._worker = Thread(target=synth_and_play)
        self._worker.daemon = True
        self._worker.start()

    def start(self, text=None, wav=None):
        """Start playing, without blocking.

        Args:
            text (str): sentence to synthesize
            wav (str): path to a sound file to play instead
        """
        if (text is None) == (wav is None):
            raise ValueError('Give either "text" or "wav", not both.')

        if text is not None:
            text = speakable(text)

        if wav is not None:
            self._proc = subprocess.Popen(
                self._wav_command(wav),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif self.engine == 'edge':
            self._start_edge(text)
        elif self.engine == 'gtts':
            self._start_gtts(text)
        else:
            self._proc = subprocess.Popen(
                self._tts_command(text),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @property
    def is_playing(self):
        """Check whether the sound is still playing (or being synthesized)."""
        if self._worker is not None and self._worker.is_alive():
            return True
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        """Cut the sound short. Never blocks more than a few seconds."""
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=5)
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._cleanup()

    def wait(self):
        """Block until the sound is over (bounded - a hung synth can't
        freeze the robot for the day)."""
        if self._worker is not None:
            self._worker.join(timeout=25)
        if self._proc is not None:
            try:
                self._proc.wait(timeout=60)
            except Exception:
                self._proc.kill()
        self._cleanup()

    def _cleanup(self):
        if self._tmp_path is not None:
            try:
                os.remove(self._tmp_path)
            except OSError:
                pass
            self._tmp_path = None


# --- Head motion ------------------------------------------------------------

def gaze_thetas(reachy, x, y, z, tilt=True):
    """시선 방향 (x, y, z) 를 목 디스크 세 개의 각도로 바꾼다.

    도달할 수 없는 방향이면 None. point_head 와 목 되잡기가 같은 계산을 쓰도록
    떼어 두었다.
    """
    neck = reachy.head.neck
    offset = GAZE_TILT * (x / 0.5) if tilt else 0.0
    q = neck.model.find_quaternion_transform([1, 0, 0], [x, y, z + offset])
    try:
        return neck.model.get_angles_from_quaternion(q.w, q.x, q.y, q.z)
    except ValueError:
        return None


def point_head(reachy, x, y, z, tilt=True):
    """Point the neck at (x, y, z) by writing disk targets directly.

    look_at() spawns a trajectory thread per call, which piles up when called
    at high frequency; this just overwrites the targets.

    tilt=True applies the GAZE_TILT baseline - right for expressive motions
    (talking, idling). Pass tilt=False when aiming at a real 3D point, e.g.
    the hand-follow gaze, which would otherwise be pushed below its target.
    """
    neck = reachy.head.neck

    thetas = gaze_thetas(reachy, x, y, z, tilt=tilt)
    if thetas is None:
        # Target outside of Orbita's reachable orientations, skip this step.
        return

    for disk, theta in zip(neck.disks, thetas):
        disk.target_rot_position = theta

    # 마지막으로 '실제로 명령한' 디스크 각도. 시선 (y, z) 만 기억해 두면 tilt
    # 적용 여부를 알 수 없어, 되돌릴 때 엉뚱한 곳을 향한다(실측 10.4도 차이).
    reachy.head._soft_thetas = tuple(thetas)

    # Remember where we are looking, so the next motion can glide from here
    # instead of snapping.
    reachy.head._soft_gaze = (y, z)


# -- Neck hold ---------------------------------------------------------------
NECK_HOLD_INTERVAL = 1.0     # 목 강성을 다시 확인하는 주기(초)
NECK_REGRIP_GLIDE = 1.2      # 되잡은 뒤 원래 보던 곳으로 돌아가는 시간(초)
NECK_REGRIP_DEADBAND = 1.5   # 이보다 덜 어긋났으면 되돌리지 않는다(도)


class NeckHold(object):
    """대기 중에도 목을 계속 붙잡아 둔다.

    Orbita 목은 connect() 직후 풀린 상태로 온다(실측: compliant 셋 다 True).
    풀린 목은 중력에 처지고, 다음에 누가 다시 굳히는 순간 마지막 목표값으로
    확 돌아간다 - 대기 중에 이따금 '팍' 하고 움직이는 정체가 이것이다.
    대화 루프도 강성을 다시 잡긴 하지만 한 바퀴에 한 번뿐이라, 아무도 말을
    걸지 않는 복도에서는 그 간격이 십수 초까지 벌어진다.

    그래서 이 스레드가 (1) 1초마다 강성을 확인하고, (2) 풀려 있었다면 목표값을
    '지금 있는 자리'로 먼저 덮어써서 튀지 않게 잡은 뒤, (3) 원래 보던 방향으로
    천천히 되돌린다.

    되잡은 횟수를 로그로 남긴다. 정말 풀리는 일이 있는지, 있다면 얼마나 잦은지
    추측하지 않고 알기 위해서다.
    """

    def __init__(self, reachy, interval=NECK_HOLD_INTERVAL):
        self.reachy = reachy
        self.interval = interval
        self.regrips = 0
        self._stop = None
        self._thread = None

    def _disks(self):
        return self.reachy.head.neck.disks

    def is_limp(self):
        """목 디스크 중 하나라도 힘이 풀려 있나."""
        try:
            return any(d.compliant for d in self._disks())
        except Exception:
            return False

    def grip(self, quiet=False):
        """튀지 않게 목을 다시 잡는다."""
        disks = self._disks()
        try:
            here = [d.rot_position for d in disks]
        except Exception:
            here = None

        # 굳히기 전에 '지금 자리'를 목표로 써 둔다. 풀린 상태의 쓰기는 무시될
        # 수 있으므로 굳힌 뒤에 한 번 더 쓴다. 이 두 번 사이 간격은 순간이라
        # 눈에 보이는 움직임이 생기지 않는다.
        if here is not None:
            for d, p in zip(disks, here):
                d.target_rot_position = p
        try:
            self.reachy.head.compliant = False
        except Exception:
            logger.debug('목 강성 설정 실패', exc_info=True)
            return
        if here is not None:
            for d, p in zip(disks, here):
                d.target_rot_position = p

        self.regrips += 1
        if not quiet:
            logger.warning('목이 풀려 있어 다시 잡았습니다 (%d번째)', self.regrips)
        self._glide_back()

    def _glide_back(self):
        """처져 있었다면, 마지막으로 명령했던 목 자세로 천천히 되돌린다."""
        thetas = getattr(self.reachy.head, '_soft_thetas', None)
        if thetas is None:
            return                      # 아직 목을 움직인 적이 없다
        try:
            here = [d.rot_position for d in self._disks()]
        except Exception:
            return
        off = max(abs(a - b) for a, b in zip(thetas, here))
        if off < NECK_REGRIP_DEADBAND:
            return                      # 제자리다 - 건드리지 않는다
        try:
            # goto 가 보간을 해 준다. 되잡기는 드문 일이라 스레드 비용은 괜찮다.
            self.reachy.head.neck.goto(list(thetas), duration=NECK_REGRIP_GLIDE,
                                       wait=False, interpolation_mode='minjerk')
            logger.info('처져 있던 목을 %.1f도 되돌립니다', off)
        except Exception:
            logger.debug('되돌리기 실패', exc_info=True)

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                if self.is_limp():
                    self.grip()
            except Exception:
                logger.debug('목 붙잡기 실패', exc_info=True)

    def start(self):
        from threading import Event, Thread

        self._stop = Event()
        # 시작할 때 한 번은 조용히 잡아 둔다 (connect 직후는 늘 풀려 있다).
        self.grip(quiet=True)
        self._thread = Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()
        logger.info('목 붙잡기 시작 (%.1f초마다 확인)', self.interval)
        return self

    def stop(self):
        if self._stop is not None:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2)


# -- Neck (Orbita) gestures --------------------------------------------------
# The neck IS a motorised 3-disk Orbita parallel mechanism - the same one the
# gaze/idle/attend motions already drive. It just isn't a set of angle joints,
# so it can't go through the arm keyframe pipeline. These gestures move it
# directly via point_head (gaze direction), fully offline (no broker). Each is
# a list of (y, z) gaze waypoints; y = left(+)/right(-), z = up(+)/down(-).
HEAD_GESTURES = {
    'up':        [(0.0, 0.35)],
    'down':      [(0.0, -0.30)],
    'left':      [(0.40, 0.0)],
    'right':     [(-0.40, 0.0)],
    'center':    [(0.0, 0.0)],
    'tilt_back': [(0.0, 0.42)],
    'nod':       [(0.0, -0.22), (0.0, 0.10), (0.0, -0.22), (0.0, 0.0)],
    'shake':     [(0.35, 0.0), (-0.35, 0.0), (0.35, 0.0), (0.0, 0.0)],
    'roll':      [(0.30, 0.10), (0.0, 0.30), (-0.30, 0.10),
                  (0.0, -0.20), (0.0, 0.0)],
}


def run_head_gesture(reachy, name, distance=0.5, freq=50, seg=0.9, hold=0.35):
    """Move the neck through a named gesture, then return to neutral.

    Glides between gaze waypoints at `freq` Hz so the Orbita moves smoothly
    (it has no speed cap; the glide is the only rate control). Ends looking
    forward. Returns True if the gesture ran.
    """
    waypoints = HEAD_GESTURES.get(name)
    if waypoints is None:
        return False

    head = reachy.head
    try:
        head.compliant = False
        for m in head.motors:
            m.compliant = False
    except Exception:
        pass
    time.sleep(0.05)

    start = list(getattr(head, '_soft_gaze', (0.0, 0.0)))
    path = list(waypoints)
    if path[-1] != (0.0, 0.0):
        path = path + [(0.0, 0.0)]        # always settle looking forward

    for (ty, tz) in path:
        y0, z0 = start
        t0 = time.time()
        while True:
            t = (time.time() - t0) / seg
            if t >= 1:
                break
            k = 0.5 - 0.5 * np.cos(np.pi * t)      # cosine ease
            point_head(reachy, distance, y0 + (ty - y0) * k,
                       z0 + (tz - z0) * k, tilt=False)
            time.sleep(1.0 / freq)
        point_head(reachy, distance, ty, tz, tilt=False)
        start = [ty, tz]
        time.sleep(hold)
    return True


class TalkingHead(object):
    """Small idle-like head motion meant to run while Reachy talks.

    Args:
        reachy (reachy.Reachy): robot to move (must have a head)
        distance (float): how far in front of the robot it looks (in m)
        nod (float, float): (amplitude in m, frequency in Hz) of the up/down motion
        sway (float, float): (amplitude in m, frequency in Hz) of the left/right motion
        antenna (float, float): (amplitude in deg, frequency in Hz) of the antennas
        freq (float): update rate (in Hz), kept low enough not to flood the Luos bus
    """

    def __init__(self, reachy,
                 distance=0.5,
                 nod=(0.04, 0.3),
                 sway=(0.05, 0.15),
                 antenna=(10, 0.6),
                 freq=50):
        self.reachy = reachy
        self.distance = distance
        self.nod = nod
        self.sway = sway
        self.antenna = antenna
        self.freq = freq

        self._ramp_t0 = None
        self._ramp_gaze = (0.0, 0.0)
        self._ramp_ant = (0.0, 0.0)
        # Longer glide so the head eases into every motion instead of snapping.
        # The Orbita neck has no working speed cap (Head.moving_speed is a
        # no-op), so this ramp IS the only thing that keeps head moves gentle.
        self._ramp_d = 1.3

    def setup(self):
        """Turn the neck and the antennas stiff."""
        self.reachy.head.compliant = False
        for m in self.reachy.head.motors:
            m.compliant = False
        time.sleep(0.1)

    def begin_ramp(self, duration=1.3):
        """Start the next motion by gliding from the current pose.

        Without this, step()/think_step() jump straight to their own targets,
        which reads as a head snap whenever a motion takes over.
        """
        self._ramp_gaze = getattr(self.reachy.head, '_soft_gaze', (0.0, 0.0))
        self._ramp_ant = (self.reachy.head.left_antenna.goal_position,
                          self.reachy.head.right_antenna.goal_position)
        self._ramp_d = duration
        self._ramp_t0 = time.time()

    def _ramp_k(self):
        """Blend factor: 0 at ramp start, easing to 1, then stays 1."""
        if self._ramp_t0 is None:
            return 1.0

        t = (time.time() - self._ramp_t0) / self._ramp_d
        if t >= 1:
            self._ramp_t0 = None
            return 1.0

        return float(0.5 - 0.5 * np.cos(np.pi * t))

    def _apply(self, y, z, left, right):
        """Write gaze + antenna targets, blended through any active ramp."""
        k = self._ramp_k()

        if k < 1.0:
            gy, gz = self._ramp_gaze
            la, ra = self._ramp_ant
            y = gy + (y - gy) * k
            z = gz + (z - gz) * k
            left = la + (left - la) * k
            right = ra + (right - ra) * k

        point_head(self.reachy, self.distance, y, z)
        self.reachy.head.left_antenna.goal_position = left
        self.reachy.head.right_antenna.goal_position = right

    def step(self, t):
        """Apply the pose for time t (in seconds since the motion started)."""
        nod_amp, nod_freq = self.nod
        sway_amp, sway_freq = self.sway
        ant_amp, ant_freq = self.antenna

        y = sway_amp * np.sin(2 * np.pi * sway_freq * t)
        z = nod_amp * np.sin(2 * np.pi * nod_freq * t)
        p = ant_amp * np.sin(2 * np.pi * ant_freq * t)

        self._apply(y, z, p, -p)

    def think_step(self, t):
        """Pose for time t of the "thinking" motion.

        Gaze drifts up and to the side and the antennas move slowly and
        asymmetrically - reads as pondering, clearly different from the
        symmetric talking motion.
        """
        y = 0.14 + 0.03 * np.sin(2 * np.pi * 0.25 * t)
        z = 0.10 + 0.03 * np.sin(2 * np.pi * 0.40 * t)
        # Antennas carry the "I am working on it" signal: they are light and
        # fast (unlike the Orbita neck), so a clear asymmetric sweep is what a
        # person actually notices from a few metres away.
        left = 48 + 26 * np.sin(2 * np.pi * 0.55 * t)
        right = -14 + 18 * np.sin(2 * np.pi * 0.38 * t + 1.1)

        self._apply(y, z, left, right)

    def acknowledge(self, duration=0.7):
        """A quick antenna perk the instant speech is recognised.

        Runs BEFORE the model is even called, so the person gets immediate
        confirmation that their words landed - the dead zone between "I stopped
        talking" and "the robot started thinking" was the confusing part.
        Non-blocking; antennas only, so it never fights the neck or the arms.
        """
        from threading import Thread

        def run():
            head = self.reachy.head
            try:
                head.compliant = False
                for m in head.motors:
                    m.compliant = False
                t0 = time.time()
                while True:
                    t = time.time() - t0
                    if t >= duration:
                        break
                    # up fast, settle back - like ears pricking up
                    k = np.sin(np.pi * min(1.0, t / duration))
                    head.left_antenna.goal_position = 38 * k
                    head.right_antenna.goal_position = -38 * k
                    time.sleep(1 / self.freq)
                head.left_antenna.goal_position = 0
                head.right_antenna.goal_position = 0
            except Exception:
                logger.debug('acknowledge failed', exc_info=True)

        th = Thread(target=run)
        th.daemon = True
        th.start()
        return th

    def start_thinking(self):
        """Run the thinking motion in the background until stop_thinking().

        Use it to cover a wait (e.g. an LLM request): the robot ponders instead
        of freezing. Follow with stop_thinking() before talking or homing.
        """
        from threading import Event, Thread

        self.stop_thinking()

        self._think_running = Event()
        self._think_running.set()

        self.begin_ramp()

        def loop():
            t0 = time.time()
            while self._think_running.is_set():
                self.think_step(time.time() - t0)
                time.sleep(1 / self.freq)

        self._think_t = Thread(target=loop)
        self._think_t.daemon = True
        self._think_t.start()

    def stop_thinking(self):
        """Stop the background thinking motion, without homing the head."""
        running = getattr(self, '_think_running', None)
        if running is not None and running.is_set():
            running.clear()
            self._think_t.join()

    def run_while(self, keep_going, timeout=60):
        """Move the head until `keep_going()` returns False.

        Args:
            keep_going (callable): called every step, motion stops when it is False
            timeout (float): hard limit (in seconds) so a stuck player cannot loop forever
        """
        self.begin_ramp()

        t0 = time.time()

        while keep_going():
            t = time.time() - t0
            if t > timeout:
                logger.warning('Head motion timed out', extra={'timeout': timeout})
                break

            self.step(t)
            time.sleep(1 / self.freq)

    # 중립 자세의 시선. point_head 와 같은 좌표계(거리 0.5)로 적어 둔다.
    HOME_GAZE = (0.0, -0.05)

    def home(self, duration=1.8):
        """Bring the head back to its neutral pose, slowly and gently.

        예전에는 look_at 으로 돌아갔는데, look_at 은 head._soft_gaze 를 갱신하지
        않는다. 그래서 말이 끝날 때마다 '로봇이 기억하는 시선'과 실제 자세가
        어긋났고, 다음에 idle 이 그 어긋난 지점에서 출발하면서 첫 프레임에
        확 튀었다 - 대기 중에 이따금 '팍' 하고 움직이는 원인 중 하나다.
        point_head 로 부드럽게 미끄러지면 기억과 실제가 항상 같이 간다.
        """
        self.reachy.head.left_antenna.goto(0, duration, interpolation_mode='minjerk')
        self.reachy.head.right_antenna.goto(0, duration, interpolation_mode='minjerk')

        y0, z0 = getattr(self.reachy.head, '_soft_gaze', (0.0, 0.0))
        ty, tz = self.HOME_GAZE
        t0 = time.time()
        while True:
            t = (time.time() - t0) / duration
            if t >= 1:
                break
            k = 0.5 - 0.5 * np.cos(np.pi * t)      # 부드러운 가감속
            point_head(self.reachy, 0.5,
                       y0 + (ty - y0) * k, z0 + (tz - z0) * k)
            time.sleep(1 / self.freq)
        point_head(self.reachy, 0.5, ty, tz)


class IdleMotion(object):
    """Background repertoire of small random motions while the robot waits.

    Keeps the robot looking alive between turns: slow glances around, little
    antenna twitches, a breathing bob. Runs in its own thread; start() before
    waiting for input, stop() before thinking or talking motions take over.

    Args:
        reachy (reachy.Reachy): robot to move (must have a head)
        distance (float): how far in front the gaze wanders (in m)
        freq (float): update rate (in Hz)
    """

    # Attend (person-follow) tuning. The servo is relative and iterated, so
    # the gain just sets how fast the head centers a face; ATTEND_SIGN must
    # match the camera (flip to -1 if the head turns away from people).
    ATTEND_GAIN_Y = 0.18
    ATTEND_GAIN_Z = 0.10
    ATTEND_MAX_Y = 0.35
    ATTEND_MAX_Z = 0.22
    ATTEND_SIGN = 1.0
    # 얼굴이 이미 화면 가운데면 더 움직이지 않는다. 계속 미세하게 쫓아가면
    # 시선이 떨려 보이고, 머리에 달린 카메라가 흔들려 사진도 흐려진다.
    # 학습용 정면 사진은 '머리가 멈춰 있을 때' 나온다.
    ATTEND_DEADBAND = 0.15

    def __init__(self, reachy, distance=0.5, freq=50, watcher=None):
        self.reachy = reachy
        self.distance = distance
        self.freq = freq
        # Optional PresenceWatcher; when set and a person is visible, the head
        # gently turns to face them instead of glancing around at random.
        self.watcher = watcher

        self._running = None
        self._thread = None
        from threading import Lock
        self._lifecycle = Lock()   # start/stop may race (main loop vs greeter)

        # Gaze state, interpolated between behaviors so motion stays smooth.
        self._gaze = [0.0, 0.0]

        self._behaviors = [
            self._glance_around,
            self._breathe,
            self._antenna_twitch,
            self._antenna_perk,
            self._recenter,
        ]

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        """Start idling in the background (idempotent, thread-safe)."""
        import random
        from threading import Event, Thread

        with self._lifecycle:
            if self._running is not None and self._running.is_set():
                return

            # Continue from wherever the head is currently pointing.
            self._gaze = list(getattr(self.reachy.head, '_soft_gaze', (0.0, 0.0)))

            self._running = Event()
            self._running.set()

            def loop():
                while self._running.is_set():
                    # When someone is in view, keep looking at them; otherwise fall
                    # back to the random idle repertoire.
                    if self.watcher is not None and self.watcher.person:
                        try:
                            self._attend()
                        except Exception:
                            logger.exception('Attend behavior failed')
                        self._pause(random.uniform(0.3, 0.7))
                        continue

                    # No face, but the scene just moved (someone passing by):
                    # glance toward the motion with an antenna perk.
                    if self._motion_nearby():
                        try:
                            self._glance_motion()
                        except Exception:
                            logger.exception('Motion glance failed')
                        self._pause(random.uniform(0.5, 1.0))
                        continue

                    behavior = random.choice(self._behaviors)
                    try:
                        behavior()
                    except Exception:
                        logger.exception('Idle behavior failed')
                    self._pause(random.uniform(1.0, 3.0))

            self._thread = Thread(target=loop)
            self._thread.daemon = True
            self._thread.start()

    def stop(self):
        """Stop idling and wait for the thread to finish (thread-safe)."""
        with self._lifecycle:
            if self._running is not None and self._running.is_set():
                self._running.clear()
                self._thread.join(timeout=3)

    # -- helpers ------------------------------------------------------------

    def _pause(self, duration):
        t0 = time.time()
        while self._running.is_set() and time.time() - t0 < duration:
            time.sleep(1 / self.freq)

    def _ease(self, t):
        """Cosine ease-in-out on t in [0, 1]."""
        return 0.5 - 0.5 * np.cos(np.pi * t)

    def _move_gaze(self, y, z, duration):
        """Glide the gaze from where it is to (y, z) over `duration` seconds."""
        y0, z0 = self._gaze
        t0 = time.time()

        while self._running.is_set():
            t = (time.time() - t0) / duration
            if t >= 1:
                break
            k = self._ease(t)
            self._gaze = [y0 + (y - y0) * k, z0 + (z - z0) * k]
            point_head(self.reachy, self.distance, self._gaze[0], self._gaze[1])
            time.sleep(1 / self.freq)

        self._gaze = [y, z]
        point_head(self.reachy, self.distance, y, z)

    def _set_antennas(self, left, right):
        self.reachy.head.left_antenna.goal_position = left
        self.reachy.head.right_antenna.goal_position = right

    # -- behaviors ----------------------------------------------------------

    def _glance_around(self):
        """Look somewhere nearby, slowly, and linger there."""
        import random
        y = random.uniform(-0.22, 0.22)
        z = random.uniform(-0.12, 0.15)
        self._move_gaze(y, z, duration=random.uniform(1.8, 3.0))
        self._pause(random.uniform(0.8, 2.0))

    def _breathe(self):
        """A few slow, tiny vertical bobs."""
        y0, z0 = self._gaze
        t0 = time.time()
        d = 4.0
        while self._running.is_set() and time.time() - t0 < d:
            z = z0 + 0.015 * np.sin(2 * np.pi * 0.25 * (time.time() - t0))
            point_head(self.reachy, self.distance, y0, z)
            time.sleep(1 / self.freq)

    def _antenna_twitch(self):
        """One antenna flicks briefly - like an ear twitch."""
        import random
        side = random.choice(('left', 'right'))
        motor = getattr(self.reachy.head, side + '_antenna')

        t0 = time.time()
        d = 1.2
        while self._running.is_set() and time.time() - t0 < d:
            t = time.time() - t0
            motor.goal_position = 18 * np.sin(2 * np.pi * 1.2 * t) * (1 - t / d)
            time.sleep(1 / self.freq)
        motor.goal_position = 0

    def _antenna_perk(self):
        """Both antennas rise gently, hold, and settle back."""
        t0 = time.time()
        d = 2.5
        while self._running.is_set() and time.time() - t0 < d:
            t = time.time() - t0
            k = self._ease(min(t / 0.8, 1)) * self._ease(min((d - t) / 0.8, 1))
            self._set_antennas(20 * k, -20 * k)
            time.sleep(1 / self.freq)
        self._set_antennas(0, 0)

    def _recenter(self):
        """Drift back to looking straight ahead."""
        self._move_gaze(0.0, 0.0, duration=2.0)

    def _motion_nearby(self):
        """Scene movement worth glancing at (fresh, above noise, not too often)."""
        w = self.watcher
        if w is None:
            return False
        now = time.time()
        if now - getattr(self, '_last_motion_glance', 0.0) < 6.0:
            return False
        return (now - getattr(w, 'motion_at', 0.0) < 1.5
                and getattr(w, 'motion_level', 0.0) > 0.04)

    def _glance_motion(self):
        """Glance toward where the scene just moved, antennas perking."""
        self._last_motion_glance = time.time()
        w = self.watcher

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        ty = clamp(-self.ATTEND_SIGN * w.motion_x * 0.30,
                   -self.ATTEND_MAX_Y, self.ATTEND_MAX_Y)
        self._set_antennas(18, -18)
        self._move_gaze(ty, 0.05, duration=0.9)
        self._pause(1.2)
        self._set_antennas(0, 0)

    def _attend(self):
        """Turn the head to face the person the watcher is tracking.

        A relative, iterated servo: each call nudges the gaze to shrink the
        face's offset from the image center, so over a couple of cycles the
        head settles on the person and then follows them if they move.
        """
        w = self.watcher

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        if (abs(w.face_error) < self.ATTEND_DEADBAND
                and abs(w.face_yerr) < self.ATTEND_DEADBAND):
            return          # 이미 정면으로 보고 있다 - 가만히 둔다

        # face_error > 0 means the face is on the right of the image; move the
        # gaze that way to re-center it. Sign is flippable via ATTEND_SIGN.
        ty = clamp(self._gaze[0] - self.ATTEND_SIGN * w.face_error * self.ATTEND_GAIN_Y,
                   -self.ATTEND_MAX_Y, self.ATTEND_MAX_Y)
        tz = clamp(self._gaze[1] - w.face_yerr * self.ATTEND_GAIN_Z,
                   -self.ATTEND_MAX_Z, self.ATTEND_MAX_Z)
        self._move_gaze(ty, tz, duration=0.8)


def say_and_move(reachy, text=None, wav=None, speech=None, head=None, timeout=60):
    """Say something while the head moves along, then go back to neutral.

    Args:
        reachy (reachy.Reachy): robot to drive
        text (str): sentence to synthesize
        wav (str): sound file to play instead of synthesizing
        speech (Speech): custom speech settings, a default one is built if omitted
        head (TalkingHead): custom motion settings, a default one is built if omitted
        timeout (float): hard limit (in seconds) on the head motion
    """
    speech = speech if speech is not None else Speech()
    head = head if head is not None else TalkingHead(reachy)

    head.setup()
    speech.start(text=text, wav=wav)

    try:
        head.run_while(lambda: speech.is_playing, timeout=timeout)
    finally:
        speech.stop()
        head.home()


def move_only(reachy, duration, head=None):
    """Run the talking head motion for a fixed duration, without any sound."""
    head = head if head is not None else TalkingHead(reachy)

    head.setup()

    t0 = time.time()
    head.run_while(lambda: time.time() - t0 < duration, timeout=duration + 1)
    head.home()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', nargs='?', help='sentence to say')
    parser.add_argument('--wav', help='play this sound file instead of synthesizing text')
    parser.add_argument('--io', default='/dev/ttyUSB*', help="serial port template, or 'ws'")
    parser.add_argument('--voice', default='ko', help='espeak-ng voice')
    parser.add_argument('--speed', type=int, default=150, help='espeak-ng speed (words per minute)')
    parser.add_argument('--duration', type=float,
                        help='move for this long without any sound (ignores text/--wav)')
    parser.add_argument('--audio-only', action='store_true',
                        help='play the sound without connecting to the robot')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.audio_only:
        speech = Speech(voice=args.voice, speed=args.speed)
        logger.info('Audio backends found: %s', Speech.available_backends() or 'none')
        speech.start(text=args.text, wav=args.wav)
        speech.wait()
        return

    if args.duration is None and args.text is None and args.wav is None:
        parser.error('give a sentence, --wav, or --duration')

    from reachy import Reachy, parts

    reachy = Reachy(head=parts.Head(io=args.io))

    try:
        if args.duration is not None:
            move_only(reachy, duration=args.duration)
        else:
            say_and_move(
                reachy,
                text=args.text, wav=args.wav,
                speech=Speech(voice=args.voice, speed=args.speed),
            )
    finally:
        reachy.close()


if __name__ == '__main__':
    main()
