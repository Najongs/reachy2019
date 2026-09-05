"""Get a reply for the robot to say.

Runs on the Pi, so it sticks to Python 3.7 and the standard library only.

Two ways to get a reply, both with the same interface:

* BrokerClient   -> asks llm_broker.py over the LAN (keeps the API key off the robot)
* DirectClient   -> calls the Claude API itself over the internet (no LAN needed,
                    but the API key lives on the robot)

Use DirectClient when the robot cannot route to the machine running the broker.
The current Anthropic SDK needs Python 3.10+, so this talks raw HTTP instead.

Every failure path returns a canned line instead of raising. A network hiccup
should make the robot say something bland, not freeze mid-conversation.

Usage:
    python llm_client.py "안녕" --url http://127.0.0.1:8080 --token secret
    python llm_client.py "안녕" --direct                 # needs ANTHROPIC_API_KEY
    python llm_client.py --chat --direct
    python llm_client.py "안녕" --direct --speak         # 로봇이 말하며 움직임
"""

import argparse
import json
import logging
import os
import random
import urllib.error
import urllib.request


logger = logging.getLogger(__name__)


DEFAULT_URL = 'http://127.0.0.1:8080'

ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages'
ANTHROPIC_VERSION = '2023-06-01'

SYSTEM_PROMPT = (
    'You are the voice of Reachy, a friendly desk robot with two arms and a moving head. '
    'Your replies are read aloud by a speech synthesizer, so: answer in one or two short '
    'sentences, use plain spoken language, and never use markdown, emoji, lists, or '
    'parentheses. Answer in the language the person used. If you do not know something, '
    'say so briefly instead of guessing.'
)

HISTORY_LIMIT = 12

# Said when the model cannot be reached at all.
OFFLINE_LINES = [
    '지금은 생각이 잘 안 나네요.',
    '잠깐만요, 머리가 멍하네요.',
    '음, 잘 모르겠어요.',
]

REFUSAL_LINE = '그건 대답하기 어려운 질문이에요.'


# 터널이 끊긴 순간(브로커 포트가 안 열려 있음)에는 urlopen 이 즉시 실패한다.
# 로그를 보면 실패한 대화 18건 중 10건이 지연 0.0초 - 모델이 느린 게 아니라
# 연결 자체가 없었다. pi_tunnel 은 Restart=always 로 곧 돌아오므로, 그 몇 초를
# 기다려 주기만 하면 대부분 살아난다. 어차피 그동안 로봇은 "음..." 하며
# 뜸을 들이고 있으니, 사람이 듣기에는 캔 답변보다 훨씬 낫다.
CONNECT_RETRIES = 2
CONNECT_BACKOFF = 2.5


def _retry_connect(attempt, what):
    """연결 자체가 실패했을 때 다시 시도할지. 기다렸다가 True."""
    import time

    if attempt >= CONNECT_RETRIES:
        return False
    wait = CONNECT_BACKOFF * (attempt + 1)
    logger.warning('%s - %.1f초 뒤 다시 시도합니다 (%d/%d)',
                   what, wait, attempt + 1, CONNECT_RETRIES)
    time.sleep(wait)
    return True


def _post_json(url, payload, headers, timeout):
    """POST json and return the parsed response."""
    data = json.dumps(payload).encode('utf-8')

    request = urllib.request.Request(url, data=data, method='POST')
    request.add_header('Content-Type', 'application/json')
    for name, value in headers.items():
        request.add_header(name, value)

    response = urllib.request.urlopen(request, timeout=timeout)
    try:
        return json.loads(response.read().decode('utf-8'))
    finally:
        response.close()


class BaseClient(object):
    """Shared conversation handling and the never-raise wrapper."""

    def __init__(self):
        self._history = []

    def _remember(self, text, reply):
        self._history.append({'role': 'user', 'content': text})
        self._history.append({'role': 'assistant', 'content': reply})
        # Keep the tail; each turn is two entries.
        if len(self._history) > HISTORY_LIMIT:
            self._history = self._history[-HISTORY_LIMIT:]

    def ask(self, text, image=None):
        """Send a line (and optionally a base64 jpeg the robot sees).

        Returns the reply, or None if anything went wrong.
        """
        raise NotImplementedError

    def ask_or_fallback(self, text, image=None):
        """Same as ask(), but always returns something the robot can say."""
        reply = self.ask(text, image=image)
        if reply:
            return reply

        return random.choice(OFFLINE_LINES)

    def reset(self):
        """Forget the conversation so far."""
        self._history = []
        return True


class BrokerClient(BaseClient):
    """Ask llm_broker.py running on another machine.

    Args:
        url (str): broker base url (e.g. 'http://127.0.0.1:8080')
        token (str): shared secret, if the broker was started with --token
        timeout (float): request timeout in seconds
        session (str): conversation id, so several scripts keep separate histories
    """

    def __init__(self, url=DEFAULT_URL, token=None, timeout=10.0, session='default'):
        BaseClient.__init__(self)
        self.url = url.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.session = session

    def _headers(self):
        return {'X-Auth-Token': self.token} if self.token is not None else {}

    def ask(self, text, image=None):
        body = {'text': text, 'session': self.session}
        if image:
            body['image'] = image

        for attempt in range(CONNECT_RETRIES + 1):
            try:
                payload = _post_json(self.url + '/reply', body,
                                     self._headers(), self.timeout)
            except urllib.error.HTTPError as e:
                # 브로커는 살아 있고 모델이 실패한 것 - 다시 물어도 같다.
                logger.warning('Broker returned %s: %s', e.code, e.read()[:200])
                return None
            except (urllib.error.URLError, OSError) as e:
                if _retry_connect(attempt, '브로커에 닿지 못했습니다: {}'.format(e)):
                    continue
                logger.warning('Cannot reach broker at %s: %s', self.url, e)
                return None
            except ValueError as e:
                logger.warning('Broker sent invalid json: %s', e)
                return None

            return payload.get('reply')

        return None

    def ask_stream(self, text, on_sentence, image=None):
        """문장이 완성되는 대로 하나씩 받아 on_sentence 로 넘긴다.

        답을 다 기다렸다 말하면 생성(1.2초)과 음성 합성(1.5초)이 통째로
        직렬이 된다. 첫 문장은 0.4초면 나오므로, 받는 즉시 말하기 시작하면
        첫 소리까지가 1초쯤 빨라진다.

        전체 답을 돌려준다. 한 문장도 못 받았으면 None (호출한 쪽은 평소의
        캔 답변으로 내려가면 된다). 절대 예외를 올리지 않는다.
        """
        body = {'text': text, 'session': self.session, 'stream': True}
        if image:
            body['image'] = image
        data = json.dumps(body).encode('utf-8')

        for attempt in range(CONNECT_RETRIES + 1):
            request = urllib.request.Request(self.url + '/reply', data=data,
                                             method='POST')
            request.add_header('Content-Type', 'application/json')
            for name, value in self._headers().items():
                request.add_header(name, value)

            said_anything = False
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                logger.warning('Broker returned %s: %s', e.code, e.read()[:200])
                return None
            except (urllib.error.URLError, OSError) as e:
                if _retry_connect(attempt, '브로커에 닿지 못했습니다: {}'.format(e)):
                    continue
                logger.warning('Cannot reach broker at %s: %s', self.url, e)
                return None

            parts = []
            try:
                for line in response:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line.decode('utf-8'))
                    except ValueError:
                        continue
                    sentence = event.get('sentence')
                    if sentence:
                        parts.append(sentence)
                        said_anything = True
                        try:
                            on_sentence(sentence)
                        except Exception:
                            logger.exception('문장 처리 중 오류')
                    if event.get('done'):
                        break
            except (urllib.error.URLError, OSError) as e:
                # 말하는 도중 끊겼다. 이미 소리를 냈다면 되돌릴 수 없으니
                # 받은 데까지 쓴다.
                logger.warning('스트림이 중간에 끊겼습니다: %s', e)
                if not said_anything and _retry_connect(
                        attempt, '스트림이 시작도 못 했습니다'):
                    continue
            finally:
                try:
                    response.close()
                except Exception:
                    pass

            reply = ' '.join(parts).strip()
            return reply or None

        return None

    def ask_motion(self, text, image=None, timeout=150):
        """Ask the motion session to design a gesture.

        Returns {'say': str, 'preset': str|None, 'moves': list|None},
        or None when the broker/model failed. Never raises.
        """
        body = {'text': text, 'session': self.session + '-motion'}
        if image:
            body['image'] = image

        try:
            return _post_json(self.url + '/motion', body, self._headers(), timeout)
        except urllib.error.HTTPError as e:
            logger.warning('Motion endpoint returned %s: %s', e.code, e.read()[:200])
            return None
        except (urllib.error.URLError, OSError) as e:
            logger.warning('Cannot reach motion endpoint: %s', e)
            return None
        except ValueError as e:
            logger.warning('Motion endpoint sent invalid json: %s', e)
            return None

    def ask_vision(self, text, image=None, timeout=90):
        """시각 접지 세션에 묻는다. 원문 텍스트 또는 None. 절대 안 던진다."""
        body = {'text': text, 'session': self.session + '-vision'}
        if image:
            body['image'] = image
        try:
            out = _post_json(self.url + '/vision', body, self._headers(), timeout)
            return out.get('text')
        except urllib.error.HTTPError as e:
            logger.warning('Vision endpoint returned %s: %s', e.code,
                           e.read()[:200])
            return None
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.warning('Cannot reach vision endpoint: %s', e)
            return None

    def reset(self):
        BaseClient.reset(self)
        try:
            _post_json(self.url + '/reset', {'session': self.session},
                       self._headers(), self.timeout)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as e:
            logger.warning('Reset failed: %s', e)
            return False

    def health(self):
        """Return the broker's health payload, or None if it is not reachable."""
        try:
            response = urllib.request.urlopen(self.url + '/health', timeout=self.timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            logger.warning('Health check failed: %s', e)
            return None

        try:
            return json.loads(response.read().decode('utf-8'))
        except ValueError:
            return None
        finally:
            response.close()


class DirectClient(BaseClient):
    """Call the Claude API straight from the robot.

    Uses raw HTTP because the Anthropic SDK requires a newer Python than the Pi has.

    Args:
        api_key (str): Anthropic API key, defaults to $ANTHROPIC_API_KEY
        model (str): model id
        max_tokens (int): reply cap; spoken answers are short by design
        effort (str): 'low' keeps the robot's response time down
        timeout (float): request timeout in seconds
    """

    def __init__(self, api_key=None, workspace_id=None, model='claude-opus-5',
                 max_tokens=512, effort='low', timeout=30.0):
        BaseClient.__init__(self)

        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError('No API key. Set ANTHROPIC_API_KEY or pass --api-key.')

        # Identity-linked keys must name the workspace they act in, otherwise every
        # request comes back 400. Plain keys ignore the header.
        self.workspace_id = workspace_id or os.environ.get('ANTHROPIC_WORKSPACE_ID')

        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.timeout = timeout

    def _headers(self):
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': ANTHROPIC_VERSION,
            # Re-runs a policy-declined request on Anthropic's recommended
            # fallback model instead of returning an empty refusal.
            'anthropic-beta': 'server-side-fallback-2026-07-01',
        }
        if self.workspace_id:
            headers['anthropic-workspace-id'] = self.workspace_id

        return headers

    def ask(self, text, image=None):
        if image:
            content = [
                {'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/jpeg', 'data': image}},
                {'type': 'text', 'text': text},
            ]
        else:
            content = text

        payload = {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'system': SYSTEM_PROMPT,
            'messages': self._history + [{'role': 'user', 'content': content}],
            'output_config': {'effort': self.effort},
            'fallbacks': 'default',
        }

        try:
            response = _post_json(ANTHROPIC_URL, payload, self._headers(), self.timeout)
        except urllib.error.HTTPError as e:
            logger.warning('Claude API returned %s: %s', e.code, e.read()[:300])
            return None
        except (urllib.error.URLError, OSError) as e:
            logger.warning('Cannot reach the Claude API: %s', e)
            return None
        except ValueError as e:
            logger.warning('Claude API sent invalid json: %s', e)
            return None

        # A policy decline comes back as a successful response, not an error.
        if response.get('stop_reason') == 'refusal':
            details = response.get('stop_details') or {}
            logger.warning('Request refused (category=%s)', details.get('category'))
            return REFUSAL_LINE

        reply = ''.join(
            block.get('text', '')
            for block in response.get('content', [])
            if block.get('type') == 'text'
        ).strip()

        if reply:
            self._remember(text, reply)

        return reply


def _force_utf8_io():
    """Make stdin/stdout UTF-8 regardless of the system locale.

    Old Pi images sometimes run with a non-UTF-8 locale, which turns typed
    Korean into mojibake or UnicodeDecodeError. errors='replace' keeps the
    program alive even when genuinely broken bytes arrive.
    """
    import sys

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, OSError):
            pass


def main():
    _force_utf8_io()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text', nargs='?', help='what to say to the model')
    parser.add_argument('--direct', action='store_true',
                        help='call the Claude API from here instead of using a broker')
    parser.add_argument('--api-key', help='Anthropic API key (default: $ANTHROPIC_API_KEY)')
    parser.add_argument('--workspace-id',
                        help='workspace the key acts in, required for identity-linked keys '
                             '(default: $ANTHROPIC_WORKSPACE_ID)')
    parser.add_argument('--model', default='claude-opus-5')
    parser.add_argument('--url', default=DEFAULT_URL, help='broker base url')
    parser.add_argument('--token', help='broker shared secret, if it requires one')
    parser.add_argument('--timeout', type=float, help='request timeout in seconds')
    parser.add_argument('--session', default='default', help='broker conversation id')
    parser.add_argument('--chat', action='store_true', help='keep asking until Ctrl-D')
    parser.add_argument('--speak', action='store_true',
                        help='say the reply on the robot while moving its head')
    parser.add_argument('--io', default='/dev/ttyUSB*', help="serial port template, or 'ws'")
    parser.add_argument('--voice', default='ko', help='espeak-ng voice')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    if args.text is None and not args.chat:
        parser.error('give some text, or use --chat')

    if args.direct:
        try:
            client = DirectClient(api_key=args.api_key, workspace_id=args.workspace_id,
                                  model=args.model,
                                  timeout=args.timeout if args.timeout else 30.0)
        except ValueError as e:
            parser.error(str(e))
        logger.info('Calling the Claude API directly (model=%s)', args.model)
    else:
        client = BrokerClient(url=args.url, token=args.token,
                              timeout=args.timeout if args.timeout else 10.0,
                              session=args.session)
        health = client.health()
        if health is None:
            logger.warning('Broker is not answering; replies will be canned lines.')
        else:
            logger.info('Broker ok: %s', health)

    reachy = None
    head = None
    speech = None

    if args.speak:
        from reachy import Reachy, parts
        from say_and_move import Speech, TalkingHead, say_and_move

        reachy = Reachy(head=parts.Head(io=args.io))
        head = TalkingHead(reachy)
        speech = Speech(voice=args.voice)
        logger.info('TTS engine: %s', speech.engine)

    def handle(line):
        if head is not None:
            # Ponder while the model is working instead of freezing.
            head.setup()
            head.start_thinking()

        reply = client.ask_or_fallback(line)

        if head is not None:
            head.stop_thinking()

        print(reply)

        if head is not None:
            say_and_move(reachy, text=reply, speech=speech, head=head)

    try:
        if args.text is not None:
            handle(args.text)

        while args.chat:
            try:
                line = input('> ').strip()
            except EOFError:
                break
            if line:
                handle(line)
    finally:
        if reachy is not None:
            reachy.close()


if __name__ == '__main__':
    main()
