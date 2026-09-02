"""Broadcast the real robot's commanded pose for the web viewer.

When running on real hardware, motion commands go out over serial and the
sim_viewer has nothing to connect to. This serves the same WebSocket protocol
as reachy's io='ws' (port 6171) but as a read-only mirror: every tick it
sends the current motor goal positions (motor frame, like ws.py does), and
drains/ignores whatever the client replies.

Zero serial traffic: goal targets are read from pyluos' local shadows.
"""

import asyncio
import json
import logging
from threading import Thread


logger = logging.getLogger(__name__)


class StateMirror(object):
    """Read-only ws://0.0.0.0:6171 mirror of the robot's commanded state.

    Args:
        reachy (reachy.Reachy): connected robot (real io)
        host (str): bind address
        port (int): must match what sim_viewer connects to
        rate (float): broadcasts per second
    """

    def __init__(self, reachy, host='0.0.0.0', port=6171, rate=10, source='present'):
        # source: 'present' = actual encoder angle (shows sag/stall/hand-moves),
        #         'goal'    = commanded angle (always smooth, ignores reality).
        self.reachy = reachy
        self.host = host
        self.port = port
        self.rate = rate
        self.source = source

    def _snapshot(self):
        motors = []
        for m in self.reachy.motors:
            raw = getattr(m, '_motor', None)
            if raw is None:
                continue
            try:
                if self.source == 'present':
                    # pyluos keeps rot_position updated from the gate's stream,
                    # so this is a cached read, not a serial round-trip.
                    value = raw.rot_position
                else:
                    value = raw.target_rot_position
                motors.append({'name': m.name, 'goal_position': value})
            except Exception:
                continue

        # Head gaze: the neck is an Orbita (not a single dxl), but the last
        # commanded look direction is cached on the head as _soft_gaze (y, z at
        # x=0.5). Send it so the viewer can turn the head node.
        head = {}
        try:
            gy, gz = getattr(self.reachy.head, '_soft_gaze', (0.0, 0.0))
            head = {'gy': gy, 'gz': gz}
        except Exception:
            pass

        return json.dumps({'motors': motors, 'disks': [], 'head': head})

    async def _drain(self, websocket):
        # The viewer echoes a reply per frame (ws.py protocol); discard them
        # so buffers never grow.
        try:
            async for _message in websocket:
                pass
        except Exception:
            pass

    async def _handler(self, websocket, path=None):
        logger.info('Viewer connected to state mirror')
        drain = asyncio.ensure_future(self._drain(websocket))
        try:
            while True:
                await websocket.send(self._snapshot())
                await asyncio.sleep(1.0 / self.rate)
        except Exception:
            pass
        finally:
            drain.cancel()
            logger.info('Viewer disconnected from state mirror')

    def start(self):
        """Serve forever in a daemon thread."""
        import websockets

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            serve = websockets.serve(self._handler, self.host, self.port)
            loop.run_until_complete(serve)
            loop.run_forever()

        thread = Thread(target=run)
        thread.daemon = True
        thread.start()
        logger.info('State mirror on ws://%s:%d', self.host, self.port)
