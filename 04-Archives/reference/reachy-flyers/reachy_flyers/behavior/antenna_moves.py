from threading import Thread
import time
import random
import numpy as np

class Antenna_moves(object):
    def __init__(self,reachy):
        self.reachy = reachy

        self.running = False
        self._t = None
        
    def start(self):
            
        if self._t is not None:
            return
        
        self.running = True
        
        self._t = Thread(target=self._random_behavior)
        self._t.start()
        

        while not self._t.is_alive():
            time.sleep(0.01)
    
    def stop(self):
        self.running = False
        self._t = None


    def is_playing(self):
        if self._t is None:
            return False
        return self._t.is_alive()

    
    def _random_behavior(self):
        
        while self.running:
            
            r_a = random.randint(-20,20)
            l_a = random.randint(-20,20)
            dur = random.randint(3,10)/10

            self.traj = self.reachy.goto({   
                'head.right_antenna' : r_a,
                'head.left_antenna' : l_a,
            },duration = dur, wait=True)
            
            time.sleep(0.3)
    
    def happy_moves(self):
        self.stop()
        
        self.running = True
        
        self._t = Thread(target=self._happy)
        self._t.start()
        while not self._t.is_alive():
            time.sleep(0.01)
        self._t.join()
        
        self.running = False
        self._t = None
            
        self.start()
        
    def _happy(self):
        dur = 1
        t = np.linspace(0, dur, dur * 100)
        pos = 10 * np.sin(2 * np.pi * 5 * t)

        for p in pos:
            self.reachy.head.left_antenna.goal_position = p
            self.reachy.head.right_antenna.goal_position = -p
            time.sleep(0.01)