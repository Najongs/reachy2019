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


# 사내 인사 캠페인 멘트. 코드가 아니라 config/campaign_lines.json 에 둔다 -
# 문구는 홍보 사정으로 바뀌고, 그때마다 로봇 코드를 고쳐 다시 배포하는 건
# 과하다. 파일이 없거나 깨져 있으면 아래 기본값으로 조용히 돌아간다.
CAMPAIGN_LINES = [
    '먼저 건넨 인사, 한 걸음 가까워진 우리!',
    '우리 먼저 인사해요!',
]
# 포스터 QR 안내. 표어와 나눠 두는 이유: QR 은 걸음을 멈춘 사람만 찍을 수
# 있으므로, 스쳐 지나가는 사람에게 읊으면 그냥 흘러간다.
CAMPAIGN_INVITE = [
    '제 앞에 포스터 보이시죠? QR 코드 찍으시면 행사에 참여하실 수 있어요!',
]
CAMPAIGN_EVERY = 3          # 인사 세 번 중 한 번만 덧붙인다


def load_campaign(path=None):
    """캠페인 멘트를 읽어 (표어들, QR 안내들, 주기) 로 돌려준다.

    Pi 는 모든 파일을 ~/Documents/ 에 평면으로 두고, DGX/git 은 PARA 구조라
    config/ 아래에 둔다. 양쪽을 다 본다.
    """
    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [path] if path else [
        os.path.join(here, 'campaign_lines.json'),
        os.path.join(here, '..', 'config', 'campaign_lines.json'),
    ]
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            with open(candidate, encoding='utf-8') as fh:
                data = json.load(fh)
            lines = [ln.strip() for ln in (data.get('lines') or []) if ln.strip()]
            invite = [ln.strip() for ln in (data.get('invite') or []) if ln.strip()]
            every = int(data.get('every') or CAMPAIGN_EVERY)
            if lines or invite:
                logger.info('캠페인 멘트 %d개 + 포스터 QR 안내 %d개 '
                            '(%d번에 한 번) - %s',
                            len(lines), len(invite), every, candidate)
                return lines, invite, max(1, every)
        except Exception:
            logger.warning('캠페인 멘트를 읽지 못했습니다: %s', candidate,
                           exc_info=True)
        break
    return list(CAMPAIGN_LINES), list(CAMPAIGN_INVITE), CAMPAIGN_EVERY


def all_spoken_lines():
    """복도 인사에서 나올 수 있는 모든 문장. TTS 미리 합성용.

    캠페인 멘트는 인사말 뒤에 붙어 하나의 문장으로 합성되므로, 조합까지
    전부 돌려줘야 실제로 캐시가 맞는다.
    """
    lines, invite, _ = load_campaign()
    tags = _interleave(lines, invite)
    out = (list(HallwayGreeter.GREETINGS) + list(HallwayGreeter.PROMPTS)
           + list(lines) + list(invite))
    for greeting in HallwayGreeter.SHORT_GREETINGS:
        for tag in tags:
            out.append(greeting + ' ' + tag)
    return list(dict.fromkeys(out))


def _interleave(lines, invite):
    """표어와 QR 안내를 번갈아 놓은 한 줄짜리 순번표.

    덧붙일 차례마다 이 목록을 순서대로 돈다. 그래서 표어만 나가는 날도,
    QR 안내만 되풀이되는 날도 없다.
    """
    out = []
    for i in range(max(len(lines), len(invite))):
        if i < len(lines):
            out.append(lines[i])
        if i < len(invite):
            out.append(invite[i])
    return out


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
    # 캠페인 멘트를 덧붙일 때 쓰는 짧은 인사. 평소 인사말(44자)에 멘트까지
    # 붙이면 15초가 넘어가는데, 복도를 지나가는 사람은 그 말을 끝까지 듣지
    # 않는다. 붙일 때는 인사를 줄여서 전체가 한 호흡에 끝나게 한다.
    SHORT_GREETINGS = [
        '안녕하세요!',
        '어서 오세요!',
        '안녕하세요, 반가워요!',
    ]
    PROMPTS = [
        '혹시 궁금한 거 있으세요? 편하게 말 걸어주세요!',
        '만세 해봐, 박수 쳐봐 처럼 말하면 제가 움직여요!',
        '여기는 키로 수도권센터예요. 저에 대해 물어봐도 좋아요!',
    ]

    def __init__(self, watcher, speech, head, idle,
                 executor=None, wave_segments=None, turn_logger=None,
                 greet_cooldown=3600.0, absence_reset=20.0,
                 linger_after=40.0, prompt_cooldown=600.0,
                 gesture_cooldown=300.0, max_prompts=1,
                 snap=None, snap_interval=90.0, snap_budget=200,
                 person_db=None, campaign=None, campaign_every=None,
                 invite=None, sleeper=None):
        self.watcher = watcher
        self.speech = speech
        self.head = head
        self.idle = idle
        self.executor = executor
        self.wave_segments = wave_segments
        self.turn_logger = turn_logger
        # 밤에 불이 꺼지고 사람도 없으면 아예 말을 걸지 않는다. 인사는 이
        # 로봇이 '스스로 시작하는' 유일한 소리라, 빈 사무실에서 가장 무서운
        # 부분이다. (sleep_mode.SleepWatcher, 없으면 늘 깨어 있는 셈)
        self.sleeper = sleeper

        self.greet_cooldown = greet_cooldown
        self.absence_reset = absence_reset
        self.linger_after = linger_after
        self.prompt_cooldown = prompt_cooldown
        self.gesture_cooldown = gesture_cooldown
        self.max_prompts = max_prompts
        # Photograph visitors as they appear, so the logs accumulate real
        # scenes to learn from. Rate-limited and capped so a busy day cannot
        # fill the SD card. snap() -> base64 jpeg or None.
        self.snap = snap
        self.snap_interval = snap_interval
        self.snap_budget = snap_budget
        self._last_snap = 0.0
        self._snaps = 0
        # 사람 데이터셋. 방문 단위로 묶어 두면 나중에 "한 사람의 여러 장"으로
        # 다룰 수 있고, 대화까지 이어진 방문인지도 표시된다.
        self.person_db = person_db

        # True while the greeter is speaking/gesturing; the main loop skips
        # restarting idle during that window.
        self.busy = False
        # True while the main loop is processing a user turn (set via
        # begin_turn/end_turn) - hard-mutes all proactive behavior.
        self.in_turn = False

        # 사내 인사 캠페인. 인사 끝에 가끔만 붙인다 - 매번 붙이면 광고가
        # 되고, 지나가는 사람이 로봇 말을 끝까지 듣지 않게 된다.
        loaded_lines, loaded_invite, loaded_every = load_campaign()
        self.campaign = list(campaign) if campaign is not None else loaded_lines
        self.campaign_invite = (list(invite) if invite is not None
                                else loaded_invite)
        self.campaign_every = (campaign_every if campaign_every is not None
                               else loaded_every)
        # 표어와 QR 안내를 번갈아 붙인다.
        self._campaign_rotation = _interleave(self.campaign, self.campaign_invite)
        self._greets = 0            # 몇 번째 인사인지 (주기 계산용)
        self._campaign_at = 0       # 다음에 쓸 멘트 (순서대로 돌린다)

        self._last_activity = 0.0
        self._last_greet = 0.0
        self._last_prompt = 0.0
        self._greeted_at = 0.0
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

    def _log(self, event, extra=None, image_b64=None):
        if self.turn_logger is not None:
            data = {'event': event}
            if extra:
                data.update(extra)
            self.turn_logger.log('hallway', data, image_b64=image_b64)

    def _maybe_snap(self):
        """A photo of the visitor, if we are due one. Returns base64 or None.

        Every appearance is a real scene worth keeping for later analysis, but
        a busy corridor would otherwise write a frame a minute all day - hence
        the interval and the per-run budget.
        """
        if self.snap is None or self._snaps >= self.snap_budget:
            return None
        now = time.time()
        if now - self._last_snap < self.snap_interval:
            return None
        try:
            img = self.snap()
        except Exception:
            logger.debug('visitor snapshot failed', exc_info=True)
            return None
        if img:
            self._last_snap = now
            self._snaps += 1
        return img

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
                        if self.person_db is not None:
                            self.person_db.start_visit(
                                time.strftime('%Y%m%d-%H%M%S'))
                        self._log('appeared',
                                  {'absence_s': round(min(absence, 9999), 1)},
                                  image_b64=self._maybe_snap())

                if not person and prev_person:
                    if self.person_db is not None:
                        # 이 방문에 대화가 있었는지 표시해 둔다(데이터 선별용).
                        self.person_db.end_visit(
                            interacted=(now - self._last_activity) < 60)
                    self._log('left', {
                        'stayed_s': round(now - w.appeared_at, 1),
                        'greeted': self._greeted_this_visit,
                    })

                prev_person = person

                # 자는 중에는 인사도 손짓도 하지 않는다. 사람 검출과 방문
                # 기록(appeared/left)은 위에서 그대로 남는다 - 밤에 누가
                # 지나갔는지는 기록해 두되, 소리와 동작만 멈춘다.
                if self.sleeper is not None and self.sleeper.asleep:
                    continue

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
                          and now - self._greeted_at >= self.linger_after
                          and now - self._last_prompt >= self.prompt_cooldown
                          and now - self._last_activity >= self.linger_after):
                        self._prompt()
            except Exception:
                logger.exception('Hallway greeter loop error')

    def _next_campaign_line(self):
        """이번 인사에 붙일 캠페인 멘트. 붙일 차례가 아니면 None.

        무작위가 아니라 순서대로 돌린다: 하루에 인사가 몇 번 안 나가므로,
        무작위로 뽑으면 같은 멘트만 계속 나가는 날이 생긴다.
        """
        if not self._campaign_rotation:
            return None
        self._greets += 1
        if self._greets % self.campaign_every:
            return None
        line = self._campaign_rotation[
            self._campaign_at % len(self._campaign_rotation)]
        self._campaign_at += 1
        return line

    def _greet(self):
        tag = self._next_campaign_line()
        if tag:
            line = random.choice(self.SHORT_GREETINGS) + ' ' + tag
        else:
            line = random.choice(self.GREETINGS)
        self._last_greet = time.time()
        self._greeted_this_visit = True
        # 인사 시각은 _last_prompt 가 아니라 여기에 남긴다. 예전에는 인사할 때
        # _last_prompt 를 찍었는데, 루프가 그 값에 prompt_cooldown(1시간)을
        # 걸어 검사하므로 "인사 후 1시간이 지난 방문" 이라는 불가능한 조건이
        # 됐다. 로그 159건에 prompted 가 0건이었던 이유다.
        self._greeted_at = time.time()

        wave = (self.executor is not None and self.wave_segments
                and time.time() - self._last_gesture >= self.gesture_cooldown)
        logger.info('Greeting a visitor%s%s', ' (with wave)' if wave else '',
                    ' (+campaign)' if tag else '')
        self._say(line, gesture=self.wave_segments if wave else None)
        if wave:
            self._last_gesture = time.time()
        self._log('greeted', {'text': line, 'wave': bool(wave),
                              'campaign': tag})

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
        # 걸음을 멈추고 머무는 사람 - 포스터를 읽고 QR 을 찍을 수 있는
        # 유일한 상대다. 그래서 여기서는 QR 안내를 먼저 건넨다.
        kind = 'prompt'
        if self.campaign_invite:
            line = random.choice(self.campaign_invite)
            kind = 'invite'
        elif self.campaign and random.random() < 0.3:
            line = random.choice(self.campaign)
            kind = 'campaign'
        else:
            line = random.choice(self.PROMPTS)
        self._last_prompt = time.time()
        self._prompts_this_visit += 1
        logger.info('Prompting a lingering visitor (%s)', kind)
        self._say(line)
        self._log('prompted', {'text': line, 'campaign': kind})

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
