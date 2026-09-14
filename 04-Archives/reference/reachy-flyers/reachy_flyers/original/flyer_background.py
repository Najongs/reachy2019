import logging
from .behavior import flyer_actions as fa
import numpy as np 
import time

from reachy import Reachy, parts
from .behavior.head_controller import Head_Controller 
from .behavior.detection import Detection
from .behavior.antenna_moves import Antenna_moves
from .behavior.manipulate_flyer import Manipulate_flyer
from .behavior.idle import Idle, IdleForever
from .behavior.embeddings import Embeddings


from collections import deque

from PIL import Image
from edgetpu.detection.engine import DetectionEngine

logger = logging.getLogger('reachy.flyers')

model_path = '/home/pi/dev/reachy-flyers-2/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'
embeddings_data_path = '/home/pi/dev/reachy-flyers-2/embeddings_data'
facenet_path = '/home/pi/dev/reachy-flyers-2/models/FaceNet_128.tflite'

class FlyerBackground:

    def __init__(self):
        self.reachy = Reachy(
        head=parts.Head(io='/dev/ttyUSB*'),
        right_arm=parts.RightArm(io='/dev/ttyUSB*', hand='force_gripper'),
        left_arm=parts.LeftArm(io='/dev/ttyUSB*', hand='flyer_hand')
        )

        self.prev_y, self.prev_z = 0, 0
        self.cmd_y, self.cmd_z = self.prev_y, self.prev_z
        self.xM = 0
        self.yM = 0
        self.target_size = 0
        self.center = np.array([0,0])

        self.hand_empty = True

        self.queue = deque([],50)

        self.ssd_engine = DetectionEngine(model_path)
        self.detection = Detection(self.reachy, engine=self.ssd_engine)
        self.controller = Head_Controller([0, 0], cb = self.servoing, pid_params=[0.0004, 0.0001, 0, 0, 0.017, 0.002])
        self.a_moves = Antenna_moves(self.reachy)

        self.idle = Idle(self.reachy,self.hand_empty)
        self.idleForever = IdleForever(self.idle)

        self.manip = Manipulate_flyer(self.reachy)

        self.emb = Embeddings(facenet_path=facenet_path, embeddings_data_path=embeddings_data_path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        logger.info(
            'Closing the playground',
            extra={
                'exc': exc,
            }
        )
        self.reachy.close()

    def set_compliant(self):
        for m in self.reachy.right_arm.motors:
            m.compliant = True
        for m in self.reachy.head.motors:
            m.compliant = True
        for m in self.reachy.left_arm.motors:
            m.compliant = True
        self.reachy.head.compliant = True

    def set_stiff(self):
        for m in self.reachy.right_arm.motors:
            m.compliant = False
        for m in self.reachy.head.motors:
            m.compliant = False
        for m in self.reachy.left_arm.motors:
            m.compliant = False
        self.reachy.head.compliant = False
    
    def set_stiff_right_arm(self):
        for m in self.reachy.right_arm.motors:
            m.compliant = False
        time.sleep(0.01)

    def setup(self): #MODIFIED
        logger.info('Setup Reachy before starting.')
        self.set_compliant()
        time.sleep(0.5)
        self.set_stiff()
        time.sleep(0.5)

        fa.head_home(self.reachy,False)
        fa.base_pos_right(self.reachy)
        fa.base_pos_left(self.reachy)

        img = self.reachy.head.get_image()
        self.center = np.array([int(np.shape(img)[0]/2), int(np.shape(img)[1]/2)])
        self.queue.append(self.prev_y)

        fa.initialize_gripper_threshold(self.reachy)


    # Functions related to tracking

    def servoing(self,res):
        x = 0.5    
        y, z = res

        quat = self.reachy.head.neck.model.find_quaternion_transform([1, 0, 0], [x, y, z])    
        
        try:
            thetas = self.reachy.head.neck.model.get_angles_from_quaternion(quat.w, quat.x, quat.y, quat.z)
            for d, p in zip(self.reachy.head.neck.disks, thetas):
                d.target_rot_position = p
                
        except ValueError:
            return
    
    def activate_tracking_mode(self):
        self.controller.start()
        self.a_moves.start()
    
    def deactivate_tracking_mode(self):
        self.controller.stop()
        self.a_moves.stop()
    
    def get_target_info(self):
        self.xM, self.yM, self.target_size = self.detection._face_target
        if len(self.queue) == 0:
            self.queue.append(self.prev_y)
    
    def track(self):
        self.cmd_y, self.cmd_z = self.controller.track([self.cmd_y, self.cmd_z], [self.prev_y, self.prev_z], goal=self.center, input_controller=[self.xM, self.yM])
        self.prev_y, self.prev_z = self.cmd_y, self.cmd_z
        self.queue.append(self.prev_y)

    def look_at_previous_target(self):
        self.controller.set_new_target([self.prev_y, self.prev_z])
        
    def reinitialize_target(self):
        self.prev_y, self.prev_z = self.reachy.head.previous_look_at[1], self.reachy.head.previous_look_at[2]
        self.cmd_y, self.cmd_z = self.prev_y, self.prev_z
        self.controller.origin = np.array([self.reachy.head.previous_look_at[1], self.reachy.head.previous_look_at[2]])
        self.controller.target = np.array([self.reachy.head.previous_look_at[1], self.reachy.head.previous_look_at[2]])
        self.controller.t0 = time.time()
        self.controller.last_update.clear()
        self.detection._prev_face_target = [0,0,0]


    # Functions related to actions   
    
    def take_flyer(self): #ADDED
        logger.info('Reachy is going to take a flyer')
        self.manip._target = [0.5,self.prev_y,self.prev_z]
        self.reachy.right_arm.hand.gripper.goal_position = 0
        self.manip.play('grab_flyer')
        self.manip.play('pull_flyer_adapted')
        logger.info('Reachy achieves to take a flyer')
        self.hand_empty = False
        self.idle.hand_empty = False
        
    
    def give_flyer(self): #MODIFIED
        self.set_stiff_right_arm()
        self.manip._target = [0.5,self.prev_y,self.prev_z]
        self.manip.play('hold_flyer_adapted')
        logger.info('Reachy is holding the flyer')
        while not self.person_ready_to_take_flyer():
            logger.info('Waiting for the person to grab the flyer')
            time.sleep(0.1)
        
        ind = np.random.randint(1000)
        logger.info('The person has grabbed the flyer, Reachy unhands the flyer')
        self.manip.play('give_flyer_adapted')
        self.hand_empty = True
        self.idle.hand_empty = True
    
    def person_ready_to_take_flyer(self): #TO IMPLEMENT CORRECTLY
        return True
    
    def detect_new_person(self):
        self.a_moves.happy_moves()

    def is_new_person(self):
        img = self.detection._image
        emb_val = self.emb.get_embedding(img[self.detection._face_emb[1]:self.detection._face_emb[3], self.detection._face_emb[0]:self.detection._face_emb[2]])
        name, total_persons = self.emb.get_id_from_embedding(np.array([data[1] for data in emb_val]).reshape(128, 1), threshold=1.7)
        if name == 'Unknown':
            logger.info('Unknown person', extra={'total known persons': str(total_persons)})
            return True
        logger.info('I know you', extra={'person-name': name, 'total known persons': str(total_persons)})
        return True

    def no_flyer(self):
        self.manip._target = [0.5,self.prev_y,self.prev_z]
        self.manip.play('do_not_give_flyer')

 
    def person_comes_for_flyer(self):
        y0 = self.queue[0]
        for y in self.queue:
            if abs(y-y0)>0.07:
                logger.info('Person detected as moving too fast')
                return False
        logger.info('Person detected as intending to take a flyer',
            extra={'ys': list(self.queue),}
        )
        return True
  
    def has_been_ignored(self):
        self.manip.play('has_been_ignored')
        