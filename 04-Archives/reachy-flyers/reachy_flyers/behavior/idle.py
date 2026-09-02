import numpy as np
import time

from threading import Thread, Event

from . import flyer_actions as fa


class Idle(object):
    def __init__(self, reachy, hand_empty):
        self.reachy = reachy
        self.hand_empty = hand_empty
        self.behavior_played = None

        self.behaviors = {
            'look_around': (0.7, 0.6, fa.look_around), 
            'read_flyer': (0.0, 0.1, fa.read_flyer), 
            'stretch_head': (0.1, 0.1, fa.stretch_head),  
            'lonely': (0.07, 0.1, fa.lonely), 
            'look_hand': (0.07, 0.0, fa.look_hand), 
            'waiting': (0.04, 0.0, fa.waiting), 
            'pull_flyer': (0.02, 0.0, fa.pull_flyer_idle),
            'show_flyer': (0.0, 0.1, fa.show_flyer),
        }

        self._t = None
        
        self.y_look_at = 0
        self.z_look_at = 0 

    def play(self, behavior_name, wait):
        if self.is_playing():
            self.wait_for_end_of_play()
        
        _, _, behavior_func = self.behaviors[behavior_name]
        self.behavior_played = behavior_name

        self._t = Thread(target=behavior_func, args=[self.reachy])
        self._t.start()

        while not self._t.is_alive():
            time.sleep(0.01)

        if wait:
            self._t.join()

    def is_playing(self):
        if self._t is None:
            return False
        return self._t.is_alive()

    def wait_for_end_of_play(self):
        if self.is_playing():
            self._t.join()


class IdleForever(object):
    def __init__(self, idle_behavior):
        self.idle_behavior = idle_behavior

        self._t = None
        self.running = False

    def _play_random_behavior_forever(self):
        names = list(self.idle_behavior.behaviors.keys())
        
        while self.running:
            if self.idle_behavior.hand_empty:
                p = [v[0] for v in self.idle_behavior.behaviors.values()]
            else:
                p = [v[1] for v in self.idle_behavior.behaviors.values()]

            behavior_name = np.random.choice(names, p=p)
            self.idle_behavior.play(behavior_name, wait=True)
            if behavior_name == 'pull_flyer':
                self.idle_behavior.hand_empty = False

    def start(self):
        if self._t is not None:
            return

        self.running = True

        self._t = Thread(target=self._play_random_behavior_forever)
        self._t.start()

        while not self._t.is_alive():
            time.sleep(0.01)


    def stop(self):
        self.running = False
        
        if self._t is not None:
            self._t.join()
            self._t = None