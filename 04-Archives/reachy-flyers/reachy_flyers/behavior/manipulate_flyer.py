import numpy as np
import time

from threading import Thread, Event

from . import flyer_actions as fa
import logging

logger = logging.getLogger('reachy.flyers')


class Manipulate_flyer(object):
    def __init__(self, reachy):
        self.reachy = reachy
        
        self._target = [0.5, 0, 0]
        
        self.behaviors = {
            'give_flyer_adapted': (lambda: [fa.give_flyer_adapted, [self.reachy, self._target[0], self._target[1], self._target[2]]]), 
            'pull_flyer_adapted': (lambda: [fa.pull_flyer_adapted, [self.reachy, self._target[0], self._target[1], self._target[2]]]),
            'grab_flyer': (lambda: [fa.grab_flyer, [self.reachy]]), 
            'hold_flyer_adapted': (lambda: [fa.hold_flyer_adapted, [self.reachy,self._target[0], self._target[1], self._target[2]]]),
            'has_been_ignored': (lambda: [fa.has_been_ignored, [self.reachy]]), 
            'do_not_give_flyer': (lambda: [fa.do_not_give_flyer, [self.reachy, self._target[0], self._target[1], self._target[2]]]),
        }
        
        self._t = None


    def play(self, behavior_name):
        if self.is_playing():
            self.wait_for_end_of_play()
        
        get_behavior = self.behaviors[behavior_name]
        [behavior_func, arg_behavior] = get_behavior()
        
        self._t = Thread(target=behavior_func, args=arg_behavior)
        self._t.start()

        while not self._t.is_alive():
            time.sleep(0.01)
            
        logger.info('Playing a manipulation', extra={'manipulation-name':behavior_func})

        self._t.join()

    def is_playing(self):
        if self._t is None:
            return False
        return self._t.is_alive()

    def wait_for_end_of_play(self):
        if self.is_playing():
            self._t.join()