"""LLM broker — runs on the workstation, not on the robot.

The Pi is stuck on Python 3.7 (Buster, armv7l) and has no CPU headroom to spare
from the motor loop. Model calls therefore live here and the robot only sends a
short HTTP request over the LAN.

Usage:
    python llm_broker.py --backend ollama         # listens on 0.0.0.0:8080
    python llm_broker.py --backend echo            # no API needed, for wiring tests
    python llm_broker.py --token secret123         # require X-Auth-Token from clients

API:
    GET  /health          -> {"ok": true, "backend": "OllamaBackend", ...}
    POST /reply           {"text": "...", "session": "default"} -> {"reply": "..."}
    POST /ground          {"kind": "ground", ...} -> {"result": {...}}
    POST /evaluate        {"kind": "critique", ...} -> {"result": {...}}
    POST /reset           {"session": "default"}                -> {"ok": true}
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    'You are the voice of Reachy, a friendly desk robot with two arms and a moving head. '
    'Your replies are read aloud by a speech synthesizer, so: answer in one or two short '
    'sentences, use plain spoken language, and never use markdown, emoji, lists, or '
    'parentheses. Answer in the language the person used. If you do not know something, '
    'say so briefly instead of guessing.'
)

# How many past turns to keep per session (user + assistant each count as one).
HISTORY_LIMIT = 12

FALLBACK_REPLY = '미안해요, 지금은 대답하기 어려워요.'


GROUND_SCHEMA = {
    'type': 'object',
    'properties': {
        'box': {'type': ['integer', 'null']},
        'tray_box': {'type': ['integer', 'null']},
        'memory': {'type': ['string', 'null']},
        'action': {'type': 'string',
                   'enum': ['look', 'point', 'reach', 'pick', 'lift']},
        'approach': {'type': 'string', 'enum': ['top', 'side']},
        'reason': {'type': 'string'},
    },
    'required': ['box', 'tray_box', 'memory', 'action', 'approach', 'reason'],
    'additionalProperties': False,
}

PILOT_SCHEMA = {
    'type': 'object',
    'properties': {
        'cmd': {'type': 'string', 'enum': [
            'gaze', 'ground', 'approach', 'descend', 'adjust', 'close',
            'lift', 'carry', 'release', 'done', 'abort']},
        'why': {'type': 'string'},
        'box': {'type': ['integer', 'null']},
        'dx': {'type': ['number', 'null']},
        'dy': {'type': ['number', 'null']},
    },
    'required': ['cmd', 'why', 'box', 'dx', 'dy'],
    'additionalProperties': False,
}

EVAL_SCHEMAS = {
    'health': {
        'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
        'required': ['ok'], 'additionalProperties': False,
    },
    'duel': {
        'type': 'object',
        'properties': {'winner': {'type': 'string', 'enum': ['A', 'B']},
                       'why': {'type': 'string'}},
        'required': ['winner', 'why'], 'additionalProperties': False,
    },
    'review': {
        'type': 'object',
        'properties': {'review': {'type': 'string'}},
        'required': ['review'], 'additionalProperties': False,
    },
    'critique': {
        'type': 'object',
        'properties': {
            'naturalness': {'type': 'number'},
            'rubric': {
                'type': 'object',
                'properties': {k: {'type': 'number'} for k in
                               ('smooth', 'direct', 'posture', 'tempo')},
                'required': ['smooth', 'direct', 'posture', 'tempo'],
                'additionalProperties': False,
            },
            'worst_frame': {'type': 'integer'},
            'failure_cause': {'type': ['string', 'null']},
            'fix_hint': {'type': 'string'},
            'issues': {'type': 'array', 'items': {'type': 'string'}},
            'advice': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'parameter': {'type': 'string'},
                        'direction': {'type': 'string', 'enum': ['up', 'down']},
                    },
                    'required': ['parameter', 'direction'],
                    'additionalProperties': False,
                },
            },
        },
        'required': ['naturalness', 'rubric', 'worst_frame', 'failure_cause',
                     'fix_hint', 'issues', 'advice'],
        'additionalProperties': False,
    },
    'orchestra': {
        'type': 'object',
        'properties': {
            'experiment': {
                'type': ['object', 'null'],
                'properties': {
                    'table': {'type': 'array', 'items': {'type': 'number'}},
                    'objects': {'type': 'array', 'items': {'type': 'integer'}},
                    'min_gap': {'type': 'number'},
                },
                'required': ['table', 'objects', 'min_gap'],
                'additionalProperties': False,
            },
            'feedback_add': {'type': 'array', 'items': {'type': 'string'}},
            'feedback_done': {'type': 'array', 'items': {'type': 'integer'}},
            'why': {'type': 'string'},
        },
        'required': ['experiment', 'feedback_add', 'feedback_done', 'why'],
        'additionalProperties': False,
    },
    'director': {
        'type': 'object',
        'properties': {
            'env': {
                'type': ['object', 'null'],
                'properties': {
                    'table': {'type': 'array', 'items': {'type': 'number'}},
                    'objects': {'type': 'array', 'items': {'type': 'integer'}},
                    'min_gap': {'type': 'number'},
                },
                'required': ['table', 'objects', 'min_gap'],
                'additionalProperties': False,
            },
            'kinds': {'type': ['array', 'null'], 'items': {
                'type': 'string',
                'enum': ['pick', 'lift', 'reach', 'look']}},
            'natural_min': {'type': ['integer', 'null']},
            'reset_params': {'type': ['string', 'null'],
                             'enum': ['defaults', None]},
            'feedback_add': {'type': 'array', 'items': {'type': 'string'}},
            'feedback_done': {'type': 'array', 'items': {'type': 'integer'}},
            'code_suspect': {'type': ['string', 'null']},
            'why': {'type': 'string'},
        },
        'required': ['env', 'kinds', 'natural_min', 'reset_params',
                     'feedback_add', 'feedback_done', 'code_suspect', 'why'],
        'additionalProperties': False,
    },
}


class Backend(object):
    """Turn a line of text into a reply."""

    def reply(self, text, history, session='default', image=None):
        raise NotImplementedError

    def reset(self, session):
        """Forget a conversation. Most backends keep no state of their own."""

    def health(self):
        """이 백엔드가 정말 답할 수 있는 상태인가. {'ok': bool, ...}

        기본값은 '살아 있다'. 밖으로 나가는 백엔드는 실제로 확인한다 -
        브로커 프로세스가 떠 있다는 것과 모델이 답한다는 것은 다른 얘기다.
        """
        return {'ok': True}

    def reply_stream(self, text, history, session='default', image=None):
        """Yield the reply one spoken sentence at a time.

        The default just runs the blocking path and hands back a single
        chunk, so a backend that cannot stream still works over the streaming
        endpoint - it simply gains nothing.
        """
        out = self.reply(text, history, session=session, image=image)
        if out:
            yield out


class EchoBackend(Backend):
    """Repeat what was said. For testing the plumbing without an API key."""

    def reply(self, text, history, session='default', image=None):
        suffix = ' (사진도 봤어요)' if image else ''
        return '네, "{}" 라고 하셨군요.{}'.format(text, suffix)


class ClaudeBackend(Backend):
    """Claude via the Anthropic SDK."""

    def __init__(self, model='claude-opus-5', max_tokens=512, effort='low',
                 workspace_id=None):
        import anthropic
        import os

        # Identity-linked keys must name the workspace they act in.
        workspace_id = workspace_id or os.environ.get('ANTHROPIC_WORKSPACE_ID')
        headers = {'anthropic-workspace-id': workspace_id} if workspace_id else None

        self.client = anthropic.Anthropic(default_headers=headers)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def reply(self, text, history, session='default', image=None):
        if image:
            content = [
                {'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/jpeg', 'data': image}},
                {'type': 'text', 'text': text},
            ]
        else:
            content = text

        messages = list(history) + [{'role': 'user', 'content': content}]

        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            # Short spoken answers do not need deep reasoning; low effort keeps the
            # robot's response time down. Thinking stays on (adaptive) by default.
            output_config={'effort': self.effort},
            # A policy decline is re-run on Anthropic's recommended fallback model
            # instead of coming back as an empty refusal.
            betas=['server-side-fallback-2026-07-01'],
            fallbacks='default',
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        if response.stop_reason == 'refusal':
            category = getattr(response.stop_details, 'category', None)
            logger.warning('Request refused', extra={'category': category})
            return '그건 대답하기 어려운 질문이에요.'

        return ''.join(b.text for b in response.content if b.type == 'text').strip()


class OllamaJsonBackend(Backend):
    """Ollama 로 구조화 JSON 을 만드는 백엔드 (동작 생성용).

    외부 CLI 연동이 자꾸 끊긴다는 사용자 판단으로 동작 생성을 로컬
    모델로 옮겼다. 채팅용 OllamaBackend 와 다른 점:
    - 시스템 프롬프트를 직접 받는다 (전역 페르소나가 아니라 동작 설계서)
    - format=json 으로 JSON 만 나오게 강제한다
    - 발화용 후처리(문장 자르기·이모지 제거)를 하지 않는다 - JSON 이 깨진다
    - 이미지는 무시한다 (qwen2.5 는 텍스트 전용; 시각 접지는 /ground 몫)
    """

    def __init__(self, model='qwen2.5:7b', system='', host='127.0.0.1:11434',
                 timeout=60, num_predict=600, num_ctx=8192):
        self.model = model
        self.system = system
        self.url = 'http://{}/api/chat'.format(host)
        self.timeout = timeout
        self.num_predict = num_predict
        self.num_ctx = num_ctx

    def reply(self, text, history, session='default', image=None):
        import urllib.request
        body = json.dumps({
            'model': self.model,
            'messages': [{'role': 'system', 'content': self.system},
                         {'role': 'user', 'content': text}],
            'stream': False,
            'format': 'json',
            'keep_alive': -1,
            'options': {'num_predict': self.num_predict,
                        'num_ctx': self.num_ctx,
                        'temperature': 0.2},
        }).encode('utf-8')
        req = urllib.request.Request(
            self.url, data=body,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return (data.get('message', {}).get('content') or '').strip()


class ClaudeCliBackend(Backend):
    """Claude via a persistent Claude Code CLI process.

    Uses the subscription of the account logged into this machine, so it needs
    no API credits and no API key. The CLI process stays alive between turns:
    startup (~5s) is paid once, after which a turn takes ~1.5-2.5s, and the
    process keeps the conversation context itself.

    One CLI process per session, so different sessions stay separate
    conversations. A dead process is respawned on the next request.
    """

    MAX_SESSIONS = 4
    # A CLI child holds its whole conversation, so an all-day session grows
    # turn after turn (latency drift, eventual auto-compact stall). Hallway
    # chats are bursty and stateless-ish: after this much quiet, start fresh.
    IDLE_RESET = 900.0

    def __init__(self, model='haiku', timeout=60, system_prompt=None, effort='low'):
        import shutil

        self.claude = shutil.which('claude')
        if self.claude is None:
            raise RuntimeError('"claude" CLI not found on this machine.')

        self.model = model
        self.timeout = timeout
        self.effort = effort
        self._system_prompt = system_prompt

        import threading
        self._lock = threading.Lock()
        self._procs = {}   # session -> Popen
        self._last_use = {}   # session -> time.time() of last turn

        # 놀고 있는 CLI 프로세스를 스스로 거둔다. 예전에는 '다음 요청이 올 때'
        # 만 나이를 봤기 때문에, 동작 요청이 하루에 몇 번뿐인 복도에서는
        # 300~500MB 짜리 프로세스가 밤새 그대로 떠 있었다. 다음 요청이 영영
        # 안 오면 영영 안 죽는다.
        self._reaper = threading.Thread(target=self._reap_loop)
        self._reaper.daemon = True
        self._reaper.start()

    def _reap_loop(self):
        import time as _time

        while True:
            _time.sleep(60)
            try:
                with self._lock:
                    stale = [name for name, last in list(self._last_use.items())
                             if _time.time() - last > self.IDLE_RESET
                             and name in self._procs]
                    for name in stale:
                        logger.info('%r 세션이 %.0f분째 놀고 있어 정리합니다',
                                    name, self.IDLE_RESET / 60.0)
                        self._kill(name)
                        self._last_use.pop(name, None)
            except Exception:
                logger.debug('CLI 세션 정리 실패', exc_info=True)

    def _spawn(self):
        import subprocess

        cmd = [
            self.claude, '-p',
            '--input-format', 'stream-json',
            '--output-format', 'stream-json',
            '--verbose',
            '--model', self.model,
            # Speed: short spoken answers / motion JSON need no deep reasoning
            # and no tools. Low effort + tools off cuts per-turn latency.
            '--effort', self.effort,
            '--disallowedTools', 'Bash Read Write Edit Glob Grep WebFetch '
            'WebSearch Task NotebookEdit TodoWrite',
            '--append-system-prompt', (self._system_prompt or SYSTEM_PROMPT +
            ' Reply with the sentence only - no preamble.') +
            ' Do not use any tools.',
        ]
        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def _get_proc(self, session):
        import time as _time

        proc = self._procs.get(session)
        if proc is not None and proc.poll() is None:
            # Fresh context after a long quiet spell (see IDLE_RESET).
            idle = _time.time() - self._last_use.get(session, 0)
            if idle > self.IDLE_RESET:
                logger.info('Session %r idle %.0fs - starting fresh', session, idle)
                self._kill(session)
            else:
                self._last_use[session] = _time.time()
                return proc
        self._last_use[session] = _time.time()

        if len(self._procs) >= self.MAX_SESSIONS and session not in self._procs:
            # Drop the oldest conversation to stay bounded.
            oldest = next(iter(self._procs))
            self._kill(oldest)

        logger.info('Starting claude CLI process', extra={'session': session})
        proc = self._spawn()
        self._procs[session] = proc
        return proc

    def _kill(self, session):
        proc = self._procs.pop(session, None)
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _turn(self, proc, text, image=None):
        import select

        content = []
        if image:
            content.append({'type': 'image', 'source': {
                'type': 'base64', 'media_type': 'image/jpeg', 'data': image}})
        content.append({'type': 'text', 'text': text})

        message = {
            'type': 'user',
            'message': {'role': 'user', 'content': content},
        }
        proc.stdin.write(json.dumps(message) + '\n')
        proc.stdin.flush()

        import time
        deadline = time.time() + self.timeout

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError('claude CLI timed out after {}s'.format(self.timeout))

            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                raise RuntimeError('claude CLI timed out after {}s'.format(self.timeout))

            line = proc.stdout.readline()
            if not line:
                raise RuntimeError('claude CLI process died (rc={})'.format(proc.poll()))

            try:
                event = json.loads(line)
            except ValueError:
                continue

            if event.get('type') == 'result':
                # Error results (usage limit, API failure) carry an English
                # error string - never hand that to the TTS. Raising here
                # triggers the retry-on-fresh-process path, then a 502 and
                # the Pi's canned Korean line.
                if event.get('is_error') or event.get('subtype') not in (None, 'success'):
                    raise RuntimeError('claude CLI error result: {}'.format(
                        (event.get('result') or event.get('subtype') or '')[:200]))
                return (event.get('result') or '').strip()

    def reply(self, text, history, session='default', image=None):
        # The CLI process holds the conversation itself; `history` is unused.
        with self._lock:
            proc = self._get_proc(session)
            try:
                return self._turn(proc, text, image=image)
            except RuntimeError:
                # One retry on a fresh process, so a died/hung CLI heals itself.
                logger.warning('claude CLI failed, respawning', exc_info=True)
                self._kill(session)
                proc = self._get_proc(session)
                return self._turn(proc, text, image=image)

    def reset(self, session):
        """Forget a conversation by dropping its process."""
        with self._lock:
            self._kill(session)


class CodexCliBackend(Backend):
    """Codex CLI 비대화 실행기. 평가/접지마다 독립된 구조화 응답을 만든다."""

    def __init__(self, model=None, timeout=180, system_prompt=None,
                 workspace=None):
        import shutil

        self.codex = shutil.which('codex')
        if self.codex is None:
            raise RuntimeError('"codex" CLI not found on this machine.')
        self.model = model or None
        self.timeout = timeout
        self.system_prompt = system_prompt or ''
        self.workspace = workspace or os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))

    def health(self):
        try:
            p = subprocess.run([self.codex, 'login', 'status'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, timeout=10)
            return {'ok': p.returncode == 0,
                    'detail': (p.stdout or p.stderr).strip()[-160:]}
        except Exception as e:
            return {'ok': False, 'error': '{}: {}'.format(type(e).__name__, e)}

    def reply(self, text, history, session='default', image=None):
        return self.reply_structured(text, session=session, image=image,
                                     schema=None)

    def reply_structured(self, text, session='default', image=None, schema=None):
        prompt = text
        if self.system_prompt:
            prompt = self.system_prompt + '\n\n' + text
        with tempfile.TemporaryDirectory(prefix='reachy-codex-') as td:
            out_path = os.path.join(td, 'answer.txt')
            cmd = [self.codex, '--ask-for-approval', 'never', 'exec',
                   '--ephemeral', '--sandbox', 'read-only', '--skip-git-repo-check',
                   '--color', 'never', '-C', self.workspace,
                   '--output-last-message', out_path]
            if self.model:
                cmd += ['--model', self.model]
            if schema is not None:
                schema_path = os.path.join(td, 'schema.json')
                with open(schema_path, 'w', encoding='utf-8') as fh:
                    json.dump(schema, fh, ensure_ascii=False)
                cmd += ['--output-schema', schema_path]
            if image:
                image_path = os.path.join(td, 'frame.jpg')
                try:
                    raw_image = base64.b64decode(image, validate=True)
                except Exception as e:
                    raise RuntimeError('invalid base64 image: {}'.format(e))
                with open(image_path, 'wb') as fh:
                    fh.write(raw_image)
                cmd += ['--image', image_path]
            cmd.append('-')
            p = subprocess.run(cmd, input=prompt, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True,
                               timeout=self.timeout)
            if p.returncode != 0:
                raise RuntimeError('codex CLI failed (rc={}): {}'.format(
                    p.returncode, (p.stderr or p.stdout)[-300:]))
            try:
                with open(out_path, encoding='utf-8') as fh:
                    return fh.read().strip()
            except OSError as e:
                raise RuntimeError('codex CLI produced no final response: {}'.format(e))


# Strip emojis / pictographs - the local models add them despite the persona,
# and the TTS reads them as garbage.
_EMOJI = re.compile(
    '[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF'
    '\U00002190-\U000021FF\U00002B00-\U00002BFF️‍]+')


# Small models keep describing the robot in the third person ("리치는 ~해요")
# even when told to speak as itself, so rewrite it deterministically. Also fix
# spellings the TTS mispronounces.
#
# Only SUBJECT/OBJECT particles are rewritten. Predicate forms ("리치예요",
# "리치입니다") must be left alone: they are how the robot says its own NAME,
# and rewriting them turned "저는 리치예요" into "저는 저예요" - the robot
# could not introduce itself at all.
_THIRD_PERSON = [
    ('리치는 ', '저는 '), ('리치가 ', '제가 '), ('리치도 ', '저도 '),
    ('리치를 ', '저를 '), ('리치의 ', '제 '), ('리치에게 ', '저에게 '),
    ('리치와 ', '저와 '), ('리치한테 ', '저한테 '),
]
_SPELLING = [('KIRO', '키로'), ('Kiro', '키로'), ('kiro', '키로'),
             ('Reachy', '리치'), ('reachy', '리치'), ('QR', '큐알')]


def speakable_reply(text):
    """Fix third-person slips and TTS-hostile spellings in a model reply."""
    if not text:
        return text
    for a, b in _THIRD_PERSON:
        text = text.replace(a, b)
    for a, b in _SPELLING:
        text = text.replace(a, b)
    # The persona forbids parentheses (the TTS reads them awkwardly), and the
    # spelling fixes can produce duplicates like "리치(리치)". Drop the whole
    # parenthetical - the sentence reads fine without it.
    text = re.sub(r'\s*[（(][^)）]*[)）]', '', text)
    # Markdown the model sprinkles in despite the persona - the TTS would read
    # "dash dash dash" / "star star".
    text = re.sub(r'(?m)^\s*[-*#>]{1,4}\s*', ' ', text)
    # 번호 매기기도 마찬가지 - TTS 가 "일 점" 하고 읽는다.
    text = re.sub(r'(?m)^\s*\d{1,2}[.)]\s+', ' ', text)
    text = text.replace('**', '').replace('---', ' ').replace('__', '')
    return ' '.join(text.split())


# Asking for a story / joke / explanation is a request for a LONGER answer -
# cutting those to three sentences left the listener hanging mid-story.
_WANTS_LONG = ('이야기', '얘기해', '얘기 해', '스토리', '동화', '농담', '개그',
               '유머', '재밌는 거', '재미있는 거', '설명해', '알려줘', '가르쳐',
               '소개해', '노래', '시 한', '자세히', '더 말해', '계속해', '더 해',
               '왜 그래', '어떻게 되', '무슨 일')


def wants_long_answer(user_text):
    """True when the person asked for something that needs room to tell."""
    if not user_text:
        return False
    compact = user_text.replace(' ', '')
    return any(w.replace(' ', '') in compact for w in _WANTS_LONG)


def spoken_trim(text, max_sentences=3, max_chars=170):
    """Cut a reply down to something a robot can say out loud.

    Small local models ignore "answer in one or two sentences" and happily
    write multi-paragraph stories, which the TTS then reads for a minute. For
    a normal turn keep it short; when the person actually asked for a story or
    an explanation the caller passes a bigger budget so it can finish.
    """
    if not text:
        return text

    # Keep paragraphs for long-form answers, first paragraph only for short.
    if max_sentences <= 4:
        text = text.split('\n\n')[0]
    else:
        text = text.replace('\n\n', ' ').replace('\n', ' ')
    text = ' '.join(text.split())

    out, count = [], 0
    buf = ''
    for ch in text:
        buf += ch
        if ch in '.!?。':
            out.append(buf)
            buf = ''
            count += 1
            if count >= max_sentences:
                break
    if buf and count < max_sentences:
        out.append(buf)

    result = ''.join(out).strip() or text
    truncated = len(result) < len(text)
    if len(result) > max_chars:
        cut = result[:max_chars]
        for mark in ('. ', '! ', '? '):
            i = cut.rfind(mark)
            if i > max_chars * 0.5:
                cut = cut[:i + 1]
                break
        result = cut.strip()
        truncated = True

    # A story cut off mid-telling just dangles; offer to go on instead. Only
    # for long-form answers (short chat replies are complete as they are).
    if truncated and max_sentences > 4 and not result.rstrip().endswith('?'):
        result = result.rstrip() + ' 더 들려드릴까요?'
    return result


_SENTENCE_END = '.!?。'


def clean_sentence(text):
    """말할 수 있는 한 문장으로 다듬는다. 남길 게 없으면 빈 문자열."""
    return speakable_reply(_EMOJI.sub('', text or '')).strip()


def stream_sentences(pieces, max_sentences=3, max_chars=170):
    """토큰 조각들을 받아 완성된 문장이 나올 때마다 하나씩 내보낸다.

    로봇이 답을 다 만들 때까지 기다렸다가 말하면, 생성(1.2초)과 음성 합성
    (1.5초)이 통째로 직렬이 된다. 첫 문장은 0.4초면 나오므로, 나오는 대로
    말하기 시작하면 첫 소리까지가 그만큼 빨라지고 나머지 문장은 앞 문장이
    재생되는 동안 합성된다.

    spoken_trim 과 같은 예산(문장 수·글자 수)을 스트리밍에서도 지킨다.
    """
    buffer = ''
    said = 0
    chars = 0

    def budget_left():
        return said < max_sentences and chars < max_chars

    for piece in pieces:
        if not piece:
            continue
        buffer += piece
        while budget_left():
            # 따옴표 안에서는 자르지 않는다. 인용문을 그냥 마침표로 쪼개면
            # 로봇이 '"라고 물어보셨어요.' 같은 조각을 따로 말하게 된다.
            quoted = False
            cut = -1
            for i, ch in enumerate(buffer):
                if ch in '"\u201c\u201d\'':
                    quoted = not quoted
                elif ch in _SENTENCE_END and not quoted:
                    # "1." 같은 목록 번호나 "3.5" 의 소수점은 문장 끝이
                    # 아니다. 여기서 자르면 로봇이 "일." 하고 끊어 말한다.
                    if ch == '.' and i and buffer[i - 1].isdigit():
                        continue
                    cut = i + 1
                    break
            if cut < 0:
                break
            sentence, buffer = buffer[:cut], buffer[cut:]
            out = clean_sentence(sentence)
            if not out:
                continue
            yield out
            said += 1
            chars += len(out)
        if not budget_left():
            return

    tail = clean_sentence(buffer)
    if tail and budget_left():
        yield tail


class OllamaBackend(Backend):
    """Local LLM via an Ollama server (free, on the DGX GPUs/CPU, no API cost).

    Uses the broker's per-session history (Conversations), so it is stateless
    itself. Replies are cleaned for speech: emojis stripped, length capped.
    """

    # Ollama's default context is 2048 tokens and it truncates from the FRONT,
    # which is exactly where the system prompt sits. config/persona.txt is
    # ~2.4k tokens, so with the default the persona was silently cut away and
    # the robot introduced itself as "EXAONE 3.5, LG AI 연구원" instead of as
    # 리치. Costs nothing to fix: the prefix is KV-cached, so prompt_eval stays
    # at ~0.03s and generation time is unchanged.
    #
    # Keep this value STABLE. Ollama reloads the model whenever num_ctx changes
    # (~40s stall), so a broker that sends a different value than the loaded
    # one thrashes the GPU on every turn.
    NUM_CTX = 8192

    def __init__(self, model='exaone3.5:7.8b', host='127.0.0.1:11434',
                 timeout=90, num_predict=90, temperature=0.7, num_ctx=None):
        self.model = model
        self.url = 'http://{}/api/chat'.format(host)
        self.timeout = timeout
        self.num_predict = num_predict
        self.temperature = temperature
        self.num_ctx = num_ctx or self.NUM_CTX

    def reply(self, text, history, session='default', image=None):
        # A story/explanation request gets room to finish; a normal turn stays
        # short so the robot does not monologue at someone passing by.
        num_predict, max_sentences, max_chars = self._budget(text)

        with self._post(text, history, num_predict, False) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        out = (data.get('message', {}).get('content') or '').strip()
        out = _EMOJI.sub('', out).strip()
        return spoken_trim(speakable_reply(out),
                           max_sentences=max_sentences, max_chars=max_chars)

    def _budget(self, text):
        """(num_predict, 최대 문장 수, 최대 글자 수) - 일반 턴과 긴 답을 가른다."""
        if wants_long_answer(text):
            return 260, 6, 420
        return self.num_predict, 3, 170

    def _post(self, text, history, num_predict, stream):
        import urllib.request

        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        messages += list(history)
        messages.append({'role': 'user', 'content': text})

        body = json.dumps({
            'model': self.model,
            'messages': messages,
            'stream': stream,
            'options': {'num_predict': num_predict,
                        'num_ctx': self.num_ctx,
                        'temperature': self.temperature},
        }).encode('utf-8')

        req = urllib.request.Request(
            self.url, data=body, headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=self.timeout)

    def reply_stream(self, text, history, session='default', image=None):
        """Yield spoken sentences as Ollama produces them."""
        num_predict, max_sentences, max_chars = self._budget(text)

        def pieces(response):
            try:
                for line in response:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line.decode('utf-8'))
                    except ValueError:
                        continue
                    yield event.get('message', {}).get('content') or ''
                    if event.get('done'):
                        return
            finally:
                response.close()

        response = self._post(text, history, num_predict, True)
        for sentence in stream_sentences(pieces(response),
                                         max_sentences=max_sentences,
                                         max_chars=max_chars):
            yield sentence

    def health(self):
        """Ollama 가 살아 있고 모델이 올라와 있는지 실제로 물어본다.

        예전 /health 는 브로커가 떠 있다는 것만 알려 줬다. 그래서 Ollama 가
        죽어도 워치독은 '정상'을 보고 아무것도 하지 않았고, 로봇은 하루 종일
        "음, 잘 모르겠어요" 만 했다. 프로세스가 살아 있는 것과 답할 수 있는
        것은 다른 얘기다.
        """
        import urllib.request

        base = self.url.rsplit('/api/', 1)[0]
        out = {'ok': False, 'model': self.model}
        try:
            with urllib.request.urlopen(base + '/api/tags', timeout=5) as r:
                tags = json.loads(r.read().decode('utf-8'))
            names = [m.get('name') for m in (tags.get('models') or [])]
            out['ok'] = self.model in names
            if not out['ok']:
                out['error'] = '모델이 설치돼 있지 않습니다'
                return out
        except Exception as e:
            out['error'] = 'ollama 에 닿지 못했습니다: {}'.format(e)
            return out

        # 올라와 있는지(=첫 요청이 40초짜리 적재가 되지 않는지)까지 본다.
        try:
            with urllib.request.urlopen(base + '/api/ps', timeout=5) as r:
                ps = json.loads(r.read().decode('utf-8'))
            loaded = [m.get('name') for m in (ps.get('models') or [])]
            out['loaded'] = self.model in loaded
        except Exception:
            out['loaded'] = None    # 예전 버전에는 /api/ps 가 없다
        return out

    def reset(self, session):
        pass   # history lives in the broker's Conversations


class Stats(object):
    """브로커가 실제로 무슨 일을 했는지 센다.

    로그를 뒤지지 않고도 "오늘 대화 몇 건, 실패 몇 건, 얼마나 걸렸나" 를 볼 수
    있어야 문제가 생겼을 때 알아차린다. 예전에 대화 8.7%가 통째로 실패하고
    있었는데, 그건 나중에 로그를 파고 나서야 드러났다.
    """

    KEEP = 200      # 최근 몇 턴의 소요 시간을 들고 있을지

    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self.started_at = time.time()
        self.counts = defaultdict(int)
        self._latencies = defaultdict(lambda: deque(maxlen=self.KEEP))
        self.last_error = None
        self.last_error_at = None

    def record(self, kind, seconds=None, error=None):
        with self._lock:
            self.counts[kind] += 1
            if seconds is not None:
                self._latencies[kind].append(seconds)
            if error is not None:
                self.counts[kind + '_error'] += 1
                self.last_error = str(error)[:200]
                self.last_error_at = time.time()

    def snapshot(self):
        with self._lock:
            out = {
                'uptime_s': round(time.time() - self.started_at),
                'counts': dict(self.counts),
            }
            for kind, values in self._latencies.items():
                if not values:
                    continue
                ordered = sorted(values)
                out.setdefault('latency_s', {})[kind] = {
                    'n': len(ordered),
                    'p50': round(ordered[len(ordered) // 2], 2),
                    'p90': round(ordered[int(len(ordered) * 0.9)], 2),
                    'max': round(ordered[-1], 2),
                }
            if self.last_error:
                out['last_error'] = self.last_error
                out['last_error_ago_s'] = round(time.time() - self.last_error_at)
            return out


class Conversations(object):
    """Per-session rolling history."""

    def __init__(self, limit=HISTORY_LIMIT):
        self.limit = limit
        self._sessions = defaultdict(lambda: deque(maxlen=self.limit))

    def history(self, session):
        return list(self._sessions[session])

    def record(self, session, user_text, reply):
        self._sessions[session].append({'role': 'user', 'content': user_text})
        self._sessions[session].append({'role': 'assistant', 'content': reply})

    def reset(self, session):
        self._sessions.pop(session, None)


def extract_motion_json(text):
    """Pull the motion JSON object out of a model reply, tolerantly.

    Strips code fences, grabs the first balanced {...}, checks the shape.
    Returns the dict, or None when nothing parseable/valid is there.
    """
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`')
        if cleaned.startswith('json'):
            cleaned = cleaned[4:]

    start = cleaned.find('{')
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape:
            escape = False
        elif c == '\\':
            escape = True
        elif c == '"':
            in_string = not in_string
        elif not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        payload = json.loads(cleaned[start:i + 1])
                    except ValueError:
                        return None
                    break
    else:
        return None

    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get('say'), str):
        return None
    preset = payload.get('preset')
    moves = payload.get('moves')
    if preset is not None and not isinstance(preset, str):
        return None
    if moves is not None and not isinstance(moves, list):
        return None

    return {'say': payload['say'], 'preset': preset, 'moves': moves}


def extract_json_object(text):
    """모델 원문에서 JSON 객체 하나를 꺼낸다."""
    if not text:
        return None
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        out = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def _validate_schema(value, schema, path='$'):
    """이 브로커가 쓰는 JSON Schema 부분집합을 백엔드와 무관하게 검사한다."""
    expected = schema.get('type')
    expected = expected if isinstance(expected, list) else [expected]

    def type_ok(kind):
        if kind == 'null':
            return value is None
        if kind == 'object':
            return isinstance(value, dict)
        if kind == 'array':
            return isinstance(value, list)
        if kind == 'string':
            return isinstance(value, str)
        if kind == 'boolean':
            return isinstance(value, bool)
        if kind == 'integer':
            return isinstance(value, int) and not isinstance(value, bool)
        if kind == 'number':
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return True

    if expected != [None] and not any(type_ok(kind) for kind in expected):
        raise RuntimeError('{} has wrong type (expected {})'.format(
            path, '/'.join(expected)))
    if 'enum' in schema and value not in schema['enum']:
        raise RuntimeError('{} is outside enum'.format(path))
    if value is None:
        return
    if isinstance(value, dict) and 'object' in expected:
        properties = schema.get('properties', {})
        missing = [key for key in schema.get('required', []) if key not in value]
        if missing:
            raise RuntimeError('{} missing required keys: {}'.format(
                path, ', '.join(missing)))
        if schema.get('additionalProperties') is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise RuntimeError('{} has unexpected keys: {}'.format(
                    path, ', '.join(extra)))
        for key, child in value.items():
            if key in properties:
                _validate_schema(child, properties[key], path + '.' + key)
    elif isinstance(value, list) and 'array' in expected:
        item_schema = schema.get('items')
        if item_schema:
            for idx, child in enumerate(value):
                _validate_schema(child, item_schema,
                                 '{}[{}]'.format(path, idx))


def structured_reply(provider, text, session, image, schema):
    if hasattr(provider, 'reply_structured'):
        raw = provider.reply_structured(text, session=session, image=image,
                                        schema=schema)
    else:
        raw = provider.reply(text, [], session=session, image=image)
    out = extract_json_object(raw)
    if out is None:
        raise RuntimeError('model did not return a JSON object')
    _validate_schema(out, schema)
    return out


def make_handler(backend, conversations, token, motion_backend=None,
                 grounding_backend=None, evaluation_backend=None,
                 vision_backend=None, stats=None):
    # vision_backend 은 옛 호출자 호환용 이름이다.
    if grounding_backend is None:
        grounding_backend = vision_backend
    stats = stats if stats is not None else Stats()
    health_cache = {'at': 0.0, 'value': None}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        # 백엔드를 실제로 찔러 보는 건 비싸지 않지만 공짜도 아니다. 워치독이
        # 2분마다, 로봇 서비스가 기동마다 두드리므로 짧게 캐시한다.
        HEALTH_CACHE_S = 10.0

        def log_message(self, fmt, *args):
            logger.info('%s %s', self.address_string(), fmt % args)

        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # -- 문장 단위 스트리밍 응답 (chunked NDJSON) ---------------------
        #
        # 길이를 미리 알 수 없으니 Content-Length 를 못 쓴다. HTTP/1.1 의
        # chunked 로 문장 하나를 한 줄씩 흘려보내면, 로봇은 첫 줄이 닿는
        # 순간 바로 말하기 시작할 수 있다.

        def _stream_start(self):
            self.send_response(200)
            self.send_header('Content-Type',
                             'application/x-ndjson; charset=utf-8')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()

        def _stream_write(self, payload):
            body = (json.dumps(payload, ensure_ascii=False) + '\n').encode('utf-8')
            self.wfile.write(('%x\r\n' % len(body)).encode('ascii'))
            self.wfile.write(body)
            self.wfile.write(b'\r\n')
            self.wfile.flush()

        def _stream_end(self):
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()

        def _reply_streaming(self, text, session, image):
            """Send the reply sentence by sentence as the model makes it."""
            started = False
            parts = []
            t0 = time.time()
            first_at = None
            try:
                for sentence in backend.reply_stream(
                        text, conversations.history(session), session,
                        image=image):
                    if not started:
                        self._stream_start()
                        started = True
                        first_at = time.time() - t0
                    parts.append(sentence)
                    self._stream_write({'sentence': sentence})
            except Exception as e:
                logger.exception('Streaming backend failed')
                if not started:
                    # 아직 한 글자도 안 보냈으면 평소대로 502 를 준다 -
                    # 로봇은 캔 답변으로 내려간다.
                    stats.record('stream', time.time() - t0, error=e)
                    self._send(502, {'error': '{}: {}'.format(
                        type(e).__name__, e)})
                    return
                # 이미 말하기 시작한 뒤라면 되돌릴 수 없다. 받은 데까지만
                # 마무리하고 끊는다.

            if not started:
                self._stream_start()

            reply = ' '.join(parts).strip()
            if reply:
                conversations.record(session, text, reply)
                # 사람이 체감하는 지연은 '첫 소리까지' 다. 전체 생성 시간이
                # 아니라 그걸 센다.
                stats.record('stream', first_at)
            else:
                stats.record('stream', time.time() - t0, error='empty reply')
            self._stream_write({'done': True, 'reply': reply})
            self._stream_end()

        def _authorized(self):
            if token is None:
                return True
            return self.headers.get('X-Auth-Token') == token

        def _read_json(self):
            length = int(self.headers.get('Content-Length') or 0)
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode('utf-8'))

        def do_GET(self):
            if self.path.startswith('/stats'):
                self._send(200, stats.snapshot())
                return

            if not self.path.startswith('/health'):
                self._send(404, {'error': 'not found'})
                return

            now = time.time()
            if health_cache['value'] is None or (
                    now - health_cache['at'] > self.HEALTH_CACHE_S):
                try:
                    detail = backend.health()
                except Exception as e:
                    detail = {'ok': False, 'error': '{}: {}'.format(
                        type(e).__name__, e)}
                health_cache['at'] = now
                health_cache['value'] = detail
            detail = dict(health_cache['value'])

            payload = {'ok': bool(detail.pop('ok', False)),
                       'backend': type(backend).__name__}
            payload.update(detail)
            payload['uptime_s'] = round(time.time() - stats.started_at)
            components = {}
            for name, provider in (('grounding', grounding_backend),
                                   ('evaluation', evaluation_backend)):
                if provider is None:
                    components[name] = {'enabled': False}
                    continue
                try:
                    h = provider.health()
                except Exception as e:
                    h = {'ok': False, 'error': '{}: {}'.format(
                        type(e).__name__, e)}
                components[name] = dict({'enabled': True}, **h)
            # 대화 백엔드의 정상 여부만 HTTP 상태에 반영한다. 평가기 장애로
            # 로봇 대화 워치독이 브로커를 재시작하는 폭주를 막는다.
            payload['components'] = components

            # 실패는 실패로 알린다. 200 으로 돌려주면 워치독이 못 알아챈다 -
            # 예전에는 브로커가 떠 있기만 하면 늘 200 이라, Ollama 가 죽어도
            # 아무도 손대지 않았다.
            self._send(200 if payload['ok'] else 503, payload)

        def do_POST(self):
            if not self._authorized():
                self._send(401, {'error': 'bad token'})
                return

            try:
                payload = self._read_json()
            except ValueError as e:
                self._send(400, {'error': 'invalid json: {}'.format(e)})
                return

            session = payload.get('session', 'default')

            if self.path == '/reset':
                conversations.reset(session)
                backend.reset(session)
                for provider in (motion_backend, grounding_backend,
                                 evaluation_backend):
                    if provider is not None:
                        provider.reset(session)
                self._send(200, {'ok': True})
                return

            if self.path == '/motion':
                if motion_backend is None:
                    self._send(404, {'error': 'motion backend not enabled'})
                    return

                text = (payload.get('text') or '').strip()
                if not text:
                    self._send(400, {'error': 'missing "text"'})
                    return

                image = payload.get('image') or None
                t0 = time.time()
                try:
                    raw = motion_backend.reply(text, [], payload.get('session', 'motion'),
                                               image=image)
                except Exception as e:
                    logger.exception('Motion backend failed')
                    stats.record('motion', time.time() - t0, error=e)
                    self._send(502, {'error': '{}: {}'.format(type(e).__name__, e)})
                    return

                motion = extract_motion_json(raw)
                if motion is None:
                    logger.warning('Unparseable motion reply: %r', (raw or '')[:300])
                    stats.record('motion', time.time() - t0,
                                 error='unparseable motion json')
                    self._send(502, {'error': 'model did not return valid motion json'})
                    return

                stats.record('motion', time.time() - t0)
                self._send(200, motion)
                return

            if self.path in ('/ground', '/vision'):
                if grounding_backend is None:
                    self._send(404, {'error': 'grounding backend not enabled'})
                    return
                text = (payload.get('text') or '').strip()
                if not text:
                    self._send(400, {'error': 'missing "text"'})
                    return
                t0 = time.time()
                try:
                    ground_kind = payload.get('kind', 'ground')
                    if ground_kind not in ('ground', 'pilot'):
                        self._send(400, {'error': 'unknown grounding kind'})
                        return
                    schema = (PILOT_SCHEMA if ground_kind == 'pilot'
                              else GROUND_SCHEMA)
                    result = structured_reply(
                        grounding_backend, text,
                        payload.get('session', 'ground'),
                        payload.get('image') or None, schema)
                except Exception as e:
                    logger.exception('Grounding backend failed')
                    stats.record('ground', time.time() - t0, error=e)
                    self._send(502, {'error': '{}: {}'.format(
                        type(e).__name__, e)})
                    return
                stats.record('ground', time.time() - t0)
                if self.path == '/vision':
                    self._send(200, {'text': json.dumps(
                        result, ensure_ascii=False)})
                else:
                    self._send(200, {'result': result})
                return

            if self.path == '/evaluate':
                if evaluation_backend is None:
                    self._send(404, {'error': 'evaluation backend not enabled'})
                    return
                text = (payload.get('text') or '').strip()
                kind = payload.get('kind')
                if not text:
                    self._send(400, {'error': 'missing "text"'})
                    return
                if kind not in EVAL_SCHEMAS:
                    self._send(400, {'error': 'unknown evaluation kind'})
                    return
                t0 = time.time()
                try:
                    result = structured_reply(
                        evaluation_backend, text,
                        payload.get('session', 'evaluate'),
                        payload.get('image') or None, EVAL_SCHEMAS[kind])
                except Exception as e:
                    logger.exception('Evaluation backend failed')
                    stats.record('evaluate', time.time() - t0, error=e)
                    self._send(502, {'error': '{}: {}'.format(
                        type(e).__name__, e)})
                    return
                stats.record('evaluate', time.time() - t0)
                self._send(200, {'result': result})
                return

            if self.path != '/reply':
                self._send(404, {'error': 'not found'})
                return

            text = (payload.get('text') or '').strip()
            if not text:
                self._send(400, {'error': 'missing "text"'})
                return

            image = payload.get('image') or None

            if payload.get('stream'):
                self._reply_streaming(text, session, image)
                return

            t0 = time.time()
            try:
                reply = backend.reply(text, conversations.history(session), session,
                                      image=image)
            except Exception as e:
                logger.exception('Backend failed')
                stats.record('reply', time.time() - t0, error=e)
                self._send(502, {'error': '{}: {}'.format(type(e).__name__, e)})
                return

            if not reply:
                reply = FALLBACK_REPLY
                stats.record('reply', time.time() - t0, error='empty reply')
            else:
                conversations.record(session, text, reply)
                stats.record('reply', time.time() - t0)

            self._send(200, {'reply': reply})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0', help='bind address')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--backend', default='ollama',
                        choices=['claude', 'claude-cli', 'ollama', 'echo'])
    parser.add_argument('--model', default='claude-opus-5')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--effort', default='low', choices=['low', 'medium', 'high'])
    parser.add_argument('--workspace-id',
                        help='workspace the key acts in, required for identity-linked keys '
                             '(default: $ANTHROPIC_WORKSPACE_ID)')
    parser.add_argument('--system-prompt',
                        help='replace the built-in robot persona prompt')
    parser.add_argument('--system-prompt-file',
                        help='file whose contents replace the persona prompt (UTF-8)')
    parser.add_argument('--cli-model', default='haiku',
                        help="model for the claude-cli backend (e.g. 'haiku', 'sonnet')")
    parser.add_argument('--ollama-model', default='exaone3.5:7.8b',
                        help='model for the ollama chat backend (local, free)')
    parser.add_argument('--ollama-host', default='127.0.0.1:11434',
                        help='host:port of the Ollama server')
    parser.add_argument('--ollama-ctx', type=int, default=OllamaBackend.NUM_CTX,
                        help='context window; must fit the persona or Ollama '
                             'truncates it away (default: %(default)s)')
    parser.add_argument('--motion-backend', default='ollama',
                        choices=('cli', 'ollama'),
                        help='동작 생성을 레거시 CLI 로 할지 로컬 Ollama 로 할지')
    parser.add_argument('--motion-model', default='qwen2.5:7b',
                        help='model for the motion-generation session')
    parser.add_argument('--motion-prompt-file',
                        help='enable POST /motion using this system prompt file')
    parser.add_argument('--vision-model', default='opus',
                        help='model for the visual-grounding session')
    parser.add_argument('--vision-prompt-file',
                        help='deprecated alias for --ground-prompt-file')
    parser.add_argument('--ground-backend', default='codex',
                        choices=('codex', 'claude-cli'),
                        help='backend for POST /ground (default: codex)')
    parser.add_argument('--ground-model',
                        help='optional Codex model for grounding; default uses Codex config')
    parser.add_argument('--ground-prompt-file',
                        help='enable POST /ground using this system prompt file')
    parser.add_argument('--eval-backend', default='codex',
                        choices=('codex', 'claude-cli', 'disabled'),
                        help='backend for POST /evaluate (default: codex)')
    parser.add_argument('--eval-model',
                        help='optional Codex model for evaluation; default uses Codex config')
    parser.add_argument('--token', help='shared secret clients must send as X-Auth-Token')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Both backends read the module-level SYSTEM_PROMPT, so overriding it here
    # covers the API and the CLI paths alike.
    global SYSTEM_PROMPT
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding='utf-8') as f:
            SYSTEM_PROMPT = f.read().strip()
        logger.info('System prompt loaded from %s (%d chars)',
                    args.system_prompt_file, len(SYSTEM_PROMPT))
    elif args.system_prompt:
        SYSTEM_PROMPT = args.system_prompt

    if args.backend == 'echo':
        backend = EchoBackend()
    elif args.backend == 'claude-cli':
        backend = ClaudeCliBackend(model=args.cli_model)
    elif args.backend == 'ollama':
        backend = OllamaBackend(model=args.ollama_model, host=args.ollama_host,
                                num_ctx=args.ollama_ctx)
        logger.info('Chat backend: Ollama %s @ %s (num_ctx=%d)',
                    args.ollama_model, args.ollama_host, args.ollama_ctx)
    else:
        backend = ClaudeBackend(model=args.model, max_tokens=args.max_tokens,
                                effort=args.effort, workspace_id=args.workspace_id)

    motion_backend = None
    if args.motion_prompt_file:
        with open(args.motion_prompt_file, encoding='utf-8') as f:
            motion_prompt = f.read().strip()
        if args.motion_backend == 'ollama':
            motion_backend = OllamaJsonBackend(model=args.motion_model,
                                               system=motion_prompt,
                                               host=args.ollama_host)
        else:
            motion_backend = ClaudeCliBackend(model=args.motion_model,
                                              timeout=120,
                                              system_prompt=motion_prompt)
        logger.info('Motion backend enabled (model=%s, prompt %d chars)',
                    args.motion_model, len(motion_prompt))

    ground_prompt_file = args.ground_prompt_file or args.vision_prompt_file
    grounding_backend = None
    if ground_prompt_file:
        with open(ground_prompt_file, encoding='utf-8') as f:
            ground_prompt = f.read().strip()
        if args.ground_backend == 'codex':
            grounding_backend = CodexCliBackend(
                model=args.ground_model, timeout=180,
                system_prompt=ground_prompt)
        else:
            grounding_backend = ClaudeCliBackend(
                model=args.vision_model, timeout=90,
                system_prompt=ground_prompt)
        logger.info('Grounding backend enabled (%s, prompt %d chars)',
                    type(grounding_backend).__name__, len(ground_prompt))

    evaluation_backend = None
    if args.eval_backend == 'codex':
        evaluation_backend = CodexCliBackend(
            model=args.eval_model, timeout=240,
            system_prompt=('로봇 시뮬레이션 평가 전용이다. 주어진 증거만 보고 '
                           '요청된 JSON 객체만 반환하라. 도구를 사용하지 마라.'))
    elif args.eval_backend == 'claude-cli':
        evaluation_backend = ClaudeCliBackend(
            model=args.vision_model, timeout=120,
            system_prompt=('로봇 시뮬레이션 평가 전용이다. 요청한 JSON 형식만 '
                           '반환하라.'))
    if evaluation_backend is not None:
        logger.info('Evaluation backend enabled (%s)',
                    type(evaluation_backend).__name__)

    handler = make_handler(backend, Conversations(), args.token,
                           motion_backend=motion_backend,
                           grounding_backend=grounding_backend,
                           evaluation_backend=evaluation_backend, stats=Stats())
    server = ThreadingHTTPServer((args.host, args.port), handler)

    logger.info('Broker listening on %s:%d (backend=%s)', args.host, args.port, args.backend)
    if args.token is None:
        logger.warning('No --token set: anyone on this network can use this broker.')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info('Shutting down')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
