"""Play a short music clip and dance a little while it plays.

The robot keeps a handful of ~20s royalty-free clips (Kevin MacLeod,
incompetech.com, CC-BY 3.0 - see music/CREDITS.txt) in ~/Documents/music.
"노래 틀어줘" plays one, bobbing the head and antennas in time and adding
light arm accents when the arms are powered. Fully offline.

Clips are deliberately short: this is a hallway, not a concert.
"""

import logging
import os
import random
import subprocess
import time

logger = logging.getLogger('reachy.music')

MUSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'music')

# "노래 틀어줘" and friends. Kept narrow so ordinary talk about music
# ("노래 좋아해?") does not start playback.
PLAY_WORDS = ('노래 틀', '노래틀', '음악 틀', '음악틀', '노래 들려', '노래들려',
              '음악 들려', '노래 해줘', '노래해줘', '노래 불러', '노래불러',
              '한 곡', '한곡', '음악 좀', '노래 좀', '춤춰', '춤 춰', '디제이')
STOP_WORDS = ('노래 그만', '음악 그만', '노래 꺼', '음악 꺼', '노래 멈춰', '음악 멈춰',
              '그만 틀어', '조용히 해')


def wants_music(text):
    """True if the person asked for a song."""
    compact = (text or '').replace(' ', '')
    return any(w.replace(' ', '') in compact for w in PLAY_WORDS)


def wants_music_stop(text):
    compact = (text or '').replace(' ', '')
    return any(w.replace(' ', '') in compact for w in STOP_WORDS)


def available(music_dir=MUSIC_DIR):
    """List the clips on disk."""
    try:
        return sorted(f for f in os.listdir(music_dir) if f.endswith('.mp3'))
    except OSError:
        return []


def pick(music_dir=MUSIC_DIR, avoid=None):
    """Choose a clip, avoiding the one just played when possible."""
    clips = available(music_dir)
    if not clips:
        return None
    choices = [c for c in clips if c != avoid] or clips
    return os.path.join(music_dir, random.choice(choices))


class MusicPlayer(object):
    """Plays a clip in the background and dances while it runs."""

    def __init__(self, reachy=None, executor=None, music_dir=MUSIC_DIR):
        self.reachy = reachy
        self.executor = executor
        self.music_dir = music_dir
        self._proc = None
        self._thread = None
        self._stop = None
        self._last = None

    @property
    def playing(self):
        return self._proc is not None and self._proc.poll() is None

    def play(self, path=None):
        """Start a clip + dance. Returns the clip name, or None."""
        import shutil
        import threading

        if self.playing:
            return None

        path = path or pick(self.music_dir, avoid=self._last)
        if path is None:
            logger.warning('No music clips in %s', self.music_dir)
            return None

        player = shutil.which('mpg123')
        if player is None:
            logger.warning('mpg123 not installed - cannot play music')
            return None

        self._last = os.path.basename(path)
        self._proc = subprocess.Popen(
            [player, '-q', path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if self.reachy is not None:
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._dance)
            self._thread.daemon = True
            self._thread.start()

        logger.info('Playing %s', self._last)
        return self._last

    def stop(self):
        """Cut the music and the dance short."""
        if self._stop is not None:
            self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None
        self._proc = None

    def wait(self, timeout=40):
        """Block until the clip ends (or timeout)."""
        t0 = time.time()
        while self.playing and time.time() - t0 < timeout:
            time.sleep(0.1)
        self.stop()

    # -- dancing --------------------------------------------------------------

    def _dance(self):
        """Bob the head and antennas in time, with occasional arm accents."""
        import math

        from say_and_move import point_head

        head = self.reachy.head
        try:
            head.compliant = False
            for m in head.motors:
                m.compliant = False
        except Exception:
            pass

        # The Orbita neck cannot track a 2 Hz beat - driving it that fast just
        # makes it jitter. So the ANTENNAS carry the rhythm (they are light
        # dynamixels and can) while the head only sways slowly underneath.
        beat = 1.9          # antennas
        head_sway = 0.22    # Hz - one slow sweep every ~4.5s
        head_bob = 0.30     # Hz
        freq = 25.0
        t0 = time.time()
        next_accent = 3.0

        while not self._stop.is_set() and self.playing:
            t = time.time() - t0
            # Head: slow, small - a relaxed sway, never a nod on the beat.
            z = 0.045 * math.sin(2 * math.pi * head_bob * t)
            y = 0.10 * math.sin(2 * math.pi * head_sway * t)
            try:
                point_head(self.reachy, 0.5, y, z, tilt=False)
                # Antennas do the dancing: full beat plus a flick on the
                # off-beat, and they counter-rotate so it reads as rhythm.
                a = (30 * math.sin(2 * math.pi * beat * t)
                     + 12 * math.sin(2 * math.pi * beat * 2 * t + 0.6))
                head.left_antenna.goal_position = a
                head.right_antenna.goal_position = -a
            except Exception:
                pass

            # Arms join in now and then, when they are powered and free.
            if self.executor is not None and t >= next_accent:
                try:
                    self.executor.talk_accent(duration=2.2)
                except Exception:
                    pass
                next_accent = t + random.uniform(4.0, 6.5)

            if self._stop.wait(1.0 / freq):
                break

        # Settle: head forward, antennas down.
        try:
            point_head(self.reachy, 0.5, 0.0, 0.0, tilt=False)
            head.left_antenna.goal_position = 0
            head.right_antenna.goal_position = 0
        except Exception:
            pass
