"""LLM broker — runs on the workstation, not on the robot.

The Pi is stuck on Python 3.7 (Buster, armv7l), where the current Anthropic SDK
will not install, and it has no CPU headroom to spare from the motor loop. So the
model call lives here and the robot only sends a short HTTP request over the LAN.
Keeping it here also keeps the API key off the robot.

Usage:
    export ANTHROPIC_API_KEY=...
    python llm_broker.py                          # Claude, listens on 0.0.0.0:8080
    python llm_broker.py --backend echo            # no API needed, for wiring tests
    python llm_broker.py --token secret123         # require X-Auth-Token from clients

API:
    GET  /health          -> {"ok": true, "backend": "claude"}
    POST /reply           {"text": "...", "session": "default"} -> {"reply": "..."}
    POST /reset           {"session": "default"}                -> {"ok": true}
"""

import argparse
import json
import logging
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


class Backend(object):
    """Turn a line of text into a reply."""

    def reply(self, text, history, session='default', image=None):
        raise NotImplementedError

    def reset(self, session):
        """Forget a conversation. Most backends keep no state of their own."""


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

    def __init__(self, model='haiku', timeout=60, system_prompt=None):
        import shutil

        self.claude = shutil.which('claude')
        if self.claude is None:
            raise RuntimeError('"claude" CLI not found on this machine.')

        self.model = model
        self.timeout = timeout
        self._system_prompt = system_prompt

        import threading
        self._lock = threading.Lock()
        self._procs = {}   # session -> Popen

    def _spawn(self):
        import subprocess

        cmd = [
            self.claude, '-p',
            '--input-format', 'stream-json',
            '--output-format', 'stream-json',
            '--verbose',
            '--model', self.model,
            '--append-system-prompt', (self._system_prompt or SYSTEM_PROMPT +
            ' Reply with the sentence only - no preamble.') +
            ' Do not use any tools.',
        ]
        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

    def _get_proc(self, session):
        proc = self._procs.get(session)
        if proc is not None and proc.poll() is None:
            return proc

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


def make_handler(backend, conversations, token, motion_backend=None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            logger.info('%s %s', self.address_string(), fmt % args)

        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
            if self.path != '/health':
                self._send(404, {'error': 'not found'})
                return
            self._send(200, {'ok': True, 'backend': type(backend).__name__})

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
                try:
                    raw = motion_backend.reply(text, [], payload.get('session', 'motion'),
                                               image=image)
                except Exception as e:
                    logger.exception('Motion backend failed')
                    self._send(502, {'error': '{}: {}'.format(type(e).__name__, e)})
                    return

                motion = extract_motion_json(raw)
                if motion is None:
                    logger.warning('Unparseable motion reply: %r', (raw or '')[:300])
                    self._send(502, {'error': 'model did not return valid motion json'})
                    return

                self._send(200, motion)
                return

            if self.path != '/reply':
                self._send(404, {'error': 'not found'})
                return

            text = (payload.get('text') or '').strip()
            if not text:
                self._send(400, {'error': 'missing "text"'})
                return

            try:
                image = payload.get('image') or None
                reply = backend.reply(text, conversations.history(session), session,
                                      image=image)
            except Exception as e:
                logger.exception('Backend failed')
                self._send(502, {'error': '{}: {}'.format(type(e).__name__, e)})
                return

            if not reply:
                reply = FALLBACK_REPLY
            else:
                conversations.record(session, text, reply)

            self._send(200, {'reply': reply})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0', help='bind address')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--backend', default='claude',
                        choices=['claude', 'claude-cli', 'echo'])
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
    parser.add_argument('--motion-model', default='opus',
                        help='model for the motion-generation session')
    parser.add_argument('--motion-prompt-file',
                        help='enable POST /motion using this system prompt file')
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
    else:
        backend = ClaudeBackend(model=args.model, max_tokens=args.max_tokens,
                                effort=args.effort, workspace_id=args.workspace_id)

    motion_backend = None
    if args.motion_prompt_file:
        with open(args.motion_prompt_file, encoding='utf-8') as f:
            motion_prompt = f.read().strip()
        motion_backend = ClaudeCliBackend(model=args.motion_model,
                                          timeout=120,
                                          system_prompt=motion_prompt)
        logger.info('Motion backend enabled (model=%s, prompt %d chars)',
                    args.motion_model, len(motion_prompt))

    handler = make_handler(backend, Conversations(), args.token,
                           motion_backend=motion_backend)
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
