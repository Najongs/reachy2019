import numpy as np 
import time

from reachy import Reachy, parts

from PIL import Image

from edgetpu.detection.engine import DetectionEngine

import sys
sys.path.append('../')
from behavior.head_controller import Head_Controller 
from behavior.detection import Detection
from behavior.antenna_moves import Antenna_moves
from behavior.manipulate_flyer import Manipulate_flyer
from behavior.idle import Idle, IdleForever

import behavior.flyer_actions as fa

model_path = '/home/pi/dev/reachy-flyers/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'

reachy = Reachy(
    head=parts.Head(io='/dev/ttyUSB*'),
    right_arm=parts.RightArm(io='/dev/ttyUSB*', hand='force_gripper'),
    left_arm=parts.LeftArm(io='/dev/ttyUSB*', hand='flyer_hand')
)

def compliant(reachy):
    for m in reachy.right_arm.motors:
        m.compliant = True
    for m in reachy.head.motors:
        m.compliant = True
    for m in reachy.left_arm.motors:
        m.compliant = True
    reachy.head.compliant = True

def stiff(reachy):
    for m in reachy.right_arm.motors:
        m.compliant = False
    for m in reachy.head.motors:
        m.compliant = False
    for m in reachy.left_arm.motors:
        m.compliant = False
    reachy.head.compliant = False

def stiff_right(reachy):
    for m in reachy.right_arm.motors:
        m.compliant = False
    time.sleep(0.01)

def servoing(res):
    x = 0.5    
    y, z = res

    quat = reachy.head.neck.model.find_quaternion_transform([1, 0, 0], [x, y, z])    
    
    try:
        thetas = reachy.head.neck.model.get_angles_from_quaternion(quat.w, quat.x, quat.y, quat.z)
        for d, p in zip(reachy.head.neck.disks, thetas):
            d.target_rot_position = p
            
    except ValueError:
        return

print('a')
compliant(reachy)

time.sleep(0.2)

print('b')
stiff(reachy)

#######

tracking_threshold = 20*20 
action_threshold = 250*250

print('c')
fa.head_home(reachy,False)
fa.base_pos_right(reachy)
fa.base_pos_left(reachy)

print('d')
grip_threshold = fa.initialize_gripper_threshold(reachy)

print(reachy.right_arm.elbow_pitch.present_position)

#compliant(reachy)