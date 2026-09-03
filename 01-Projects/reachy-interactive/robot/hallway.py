"""Hallway demo mode: greet people who appear and invite them to talk.

A background thread watches the PresenceWatcher. When a person newly appears
(after a real absence) while the robot is idle, it greets them out loud -
optionally waving - and invites them to speak. If they linger without talking,
it offers a gentle prompt. Every event goes to the turn log, so a day in the
hallway accumulates data about who stopped by and what worked.

The tricky part: the main loop is blocked inside the microphone listen() when
we speak, so the robot HEARS ITS OWN greeting through the far-field mic. The
greeter therefore keeps an echo guard - the main loop asks is_echo() on every
recognized utterance and drops the ones that are just the robot's own recent
words coming back.
"""

import logging
import random
import time

logger = logging.getLogger('reachy.hallway')


def _bigrams(text):
    s = ''.join(text.split())
    return {s[i:i + 2] for i in range(len(s) - 1)}


class HallwayGreeter(object):
    """Proactive greeting/prompting for an unattended hallway demo.

    Args:
        watcher (PresenceWatcher): face/motion detector (must be started)
        speech (Speech): TTS
        head (TalkingHead): head motion while speaking
        idle (IdleMotion): idle behavior to pause/resume around speaking
        executor (MotionExecutor): optional, for a greeting wave
        wave_segments (list): pre-validated wave keyframes for the executor
        turn_logger (TurnLogger): optional event log
    """

    GREETINGS = [
        '안녕하세요! 저는 로봇 리치예요. 지나가다 심심하시면 말 걸어주세요!',
        '어서 오세요! 저는 리치예요. 안녕 하고 인사해보세요!',
        '안녕하세요, 반가워요! 궁금한 게 있으면 뭐든 물어보세요!',
        '안녕하세요! 인사해봐 하고 말하면 제가 인사해드려요!',
    ]
    PROMPTS = [
        '혹시 궁금한 거 있으세요? 편하게 말 걸어주세요!',
        '만세 해봐, 박수 쳐봐 처럼 말하면 제가 움직여요!',
        '여기는 키로 수도권센터예요. 저에 대해 물어봐도 좋아요!',
    ]

    def __init__(self, watcher, speech, head, idle,
                 executor=None, wave_segments=None, turn_logger=None,
                 greet_cooldown=3600.0, absence_reset=20.0,
                 linger_after=120.0, prompt_cooldown=3600.0,
                 gesture_cooldown=300.0, max_prompts=1):
        self.watcher = watcher
        self.speech = speech
        self.head = head
        self.idle = idle
        self.executor = executor
        self.wave_segments = wave_segments
        self.turn_logger = turn_logger

        self.greet_cooldown = greet_cooldown
        self.absence_reset = absence_reset
        self.linger_after = linger_after
        self.prompt_cooldown = prompt_cooldown
        self.gesture_cooldown = gesture_cooldown
        self.max_prompts = max_prompts

        # True while the greeter is speaking/gesturing; the main loop skips
        # restarting idle during that window.
        self.busy = False
        # True while the main loop is processing a user turn (set via
        # begin_turn/end_turn) - hard-mutes all proactive behavior.
        self.in_turn = False

        self._last_activity = 0.0
        self._last_greet = 0.0
        self._last_prompt = 0.0
        self._last_gesture = 0.0
        self._greeted_this_visit = False
        self._prompts_this_visit = 0
        self._echo = []          # [(bigrams, until_ts)]
        from threading import Lock
        self._echo_lock = Lock()
        # The greeter gets its OWN Speech instance so a visitor's reply
        # landing mid-greeting can never corrupt the main loop's player state.
        if speech is not None:
            try:
                self.speech = type(speech)(voice=speech.voice)
            except Exception:
                pass   # fall back to the shared one

        self._stop = None
        self._thread = None

    # -- coordination with the main loop -------------------------------------

    def note_activity(self):
        """Main loop: a real user turn is happening (heard text / speaking)."""
        self._last_activity = time.time()

    def begin_turn(self):
        """Main loop: a user turn started - hard-mute the greeter until end_turn.

        The activity timestamp alone is not enough: a motion turn (opus wait +
        gesture) can run well past the recency window, and the greeter must
        never speak over an ongoing conversation.
        """
        self.in_turn = True
        self.note_activity()

    def end_turn(self):
        """Main loop: the turn finished; stay quiet for a grace period after."""
        if self.in_turn:
            self.in_turn = False
            self.note_activity()

    def _conversation_active(self):
        return self.in_turn or time.time() - self._last_activity < 20.0

    def is_echo(self, text):
        """True if `text` is likely the robot's own recent words re-heard.

        Two-sided test: the utterance must be mostly inside the spoken phrase
        AND cover a fair share of it. One-sided matching used to swallow the
        very replies the robot invites - a visitor obeying "만세 해봐 하고
        말하면..." with "만세 해봐" was dropped as self-echo, because their
        short reply sits fully inside the long prompt. A real echo is most of
        the sentence, not three syllables of it.
        """
        now = time.time()
        with self._echo_lock:
            self._echo = [(bg, until) for bg, until in self._echo if until > now]
            candidates = list(self._echo)
        tb = _bigrams(text)
        if len(tb) < 3:
            return False        # too short to judge - let it through
        for bg, _ in candidates:
            overlap = len(tb & bg) / float(len(tb))
            coverage = len(tb & bg) / float(len(bg) or 1)
            if overlap >= 0.55 and coverage >= 0.35:
                return True
        return False

    def _register_echo(self, phrase, linger=7.0):
        with self._echo_lock:
            self._echo.append((_bigrams(phrase), time.time() + linger))

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        from threading import Event, Thread

        self._stop = Event()
        self._thread = Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()
        logger.info('Hallway greeter started')
        return self

    def stop(self):
        if getattr(self, '_stop', None) is not None and not self._stop.is_set():
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=3)

    # -- internals ------------------------------------------------------------

    def _log(self, event, extra=None):
        if self.turn_logger is not None:
            data = {'event': event}
            if extra:
                data.update(extra)
            self.turn_logger.log('hallway', data)

    def _loop(self):
        w = self.watcher
        prev_person = w.person

        while not self._stop.wait(0.3):
            try:
                person = w.person
                now = time.time()

                if person and not prev_person:
                    # New appearance only counts after a real absence, so a
                    # briefly-lost face does not retrigger the greeting.
                    absence = now - w.left_at if w.left_at else 1e9
                    if absence >= self.absence_reset:
                        self._greeted_this_visit = False
                        self._prompts_this_visit = 0
                        self._log('appeared', {'absence_s': round(min(absence, 9999), 1)})

                if not person and prev_person:
                    self._log('left', {
                        'stayed_s': round(now - w.appeared_at, 1),
                        'greeted': self._greeted_this_visit,
                    })

                prev_person = person

                if person and not self._conversation_active():
                    if (not self._greeted_this_visit
                            and now - w.appeared_at >= 1.0):
                        # Speak rarely (hour-scale cooldown, hallways are
                        # noisy); between spoken greetings, greet each new
                        # visitor with a silent wave instead.
                        if now - self._last_greet >= self.greet_cooldown:
                            self._greet()
                        elif (self.executor is not None and self.wave_segments
                              and now - self._last_gesture >= self.gesture_cooldown):
                            self._greet_silent()
                    elif (self._greeted_this_visit
                          and self._prompts_this_visit < self.max_prompts
                          and now - w.appeared_at >= self.linger_after
                          and now - self._last_prompt >= self.prompt_cooldown
                          and now - self._last_activity >= self.linger_after):
                        self._prompt()
            except Exception:
                logger.exception('Hallway greeter loop error')

    def _greet(self):
        line = random.choice(self.GREETINGS)
        self._last_greet = time.time()
        self._greeted_this_visit = True
        self._last_prompt = time.time()   # linger timer starts after greeting

        wave = (self.executor is not None and self.wave_segments
                and time.time() - self._last_gesture >= self.gesture_cooldown)
        logger.info('Greeting a visitor%s', ' (with wave)' if wave else '')
        self._say(line, gesture=self.wave_segments if wave else None)
        if wave:
            self._last_gesture = time.time()
        self._log('greeted', {'text': line, 'wave': bool(wave)})

    def _greet_silent(self):
        """Wave hello without speaking - the between-greetings default."""
        logger.info('Silent wave for a visitor')

        ok = False
        self.busy = True
        try:
            if self.idle is not None:
                self.idle.stop()
            ok, reason = self.executor.execute(self.wave_segments)
            if not ok:
                logger.info('Silent wave skipped: %s', reason)
        except Exception:
            logger.exception('Silent wave failed')
        finally:
            self.busy = False
            self._restart_idle()

        if ok:
            # Burn the cooldowns only when the visitor actually got a wave -
            # a refused execute (executor busy) must not silence this visit.
            self._greeted_this_visit = True
            self._last_gesture = time.time()
            self._log('greeted_silent', {'wave': True})

    def _restart_idle(self):
        """Hand the head back to idle - unless the main loop took a turn.

        in_turn is set by the main loop BEFORE it stops idle, so checking it
        here closes the race where the greeter re-started idle right as a
        user turn began (two threads then drove the head all turn).
        """
        if self.idle is None or self.in_turn or self._conversation_active():
            return
        try:
            self.idle.start()
        except Exception:
            logger.exception('Idle restart failed')

    def _prompt(self):
        line = random.choice(self.PROMPTS)
        self._last_prompt = time.time()
        self._prompts_this_visit += 1
        logger.info('Prompting a lingering visitor')
        self._say(line)
        self._log('prompted', {'text': line})

    def _say(self, line, gesture=None):
        """Speak (and optionally wave) without fighting the idle motions."""
        self.busy = True
        try:
            self._register_echo(line)
            if self.idle is not None:
                self.idle.stop()

            if self.speech is not None:
                self.speech.start(text=line)

            if gesture is not None:
                # Wave while the greeting plays; execute() owns arms + head.
                ok, reason = self.executor.execute(gesture)
                if not ok:
                    logger.info('Greeting wave skipped: %s', reason)
            elif self.head is not None:
                self.head.setup()
                self.head.run_while(
                    lambda: self.speech is not None and self.speech.is_playing,
                    timeout=15)

            if self.speech is not None:
                self.speech.wait()
            # Echo lingers in the open mic buffer + STT pipeline for a bit.
            self._register_echo(line)
        except Exception:
            logger.exception('Hallway say failed')
        finally:
            self.busy = False
            self._restart_idle()
