import numpy as np 
import time

from reachy import Reachy, parts

from PIL import Image

from edgetpu.detection.engine import DetectionEngine

from reachy import Reachy, parts
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

compliant(reachy)

time.sleep(0.5)

stiff(reachy)

#######

tracking_threshold = 20*20 
action_threshold = 250*250

prev_y, prev_z = 0, 0
cmd_y, cmd_z = prev_y, prev_z
track_count = 0
give_count = 0

hand_empty = True

######
ssd_engine = DetectionEngine(model_path)
d = Detection(reachy, engine=ssd_engine)
controller = Head_Controller([0, 0], cb = servoing, pid_params=[0.0004, 0.0001, 0, 0, 0.017, 0.002])
a_moves = Antenna_moves(reachy)

idle = Idle(reachy,hand_empty)
idleForever = IdleForever(idle)

manip = Manipulate_flyer(reachy)

########
fa.head_home(reachy,False)
fa.base_pos_right(reachy)
fa.base_pos_left(reachy)

img = reachy.head.get_image()
center = np.array([int(np.shape(img)[0]/2), int(np.shape(img)[1]/2)])

grip_threshold = fa.initialize_gripper_threshold(reachy)

###
d.start()
controller.start()
a_moves.start()
time.sleep(0.02)
###

idle_t = False
nobody_here = True

while True:
    if d._somebody_detected :
        
        idleForever.stop()
        if idle_t == True:
            prev_y, prev_z = reachy.head.previous_look_at[1], reachy.head.previous_look_at[2]
            cmd_y, cmd_z = prev_y, prev_z
            controller.origin = np.array([reachy.head.previous_look_at[1],reachy.head.previous_look_at[2]])
            controller.target = np.array([reachy.head.previous_look_at[1],reachy.head.previous_look_at[2]])
            controller.t0 = time.time()
            controller.last_update.clear()
            idle_t = False
        xM, yM, target_size = d._face_target
        
        #if someone is close, but not too close 
        if target_size > tracking_threshold:
            track_count = 0
            
            #if someone is close enough to start giving the flyer 
            if target_size < action_threshold:
                controller.start()
                a_moves.start()
                cmd_y, cmd_z = controller.track([cmd_y, cmd_z], [prev_y, prev_z], goal=center, input_controller=[xM, yM])
                prev_y, prev_z = cmd_y, cmd_z 
                give_count = 0
            else:
                controller.stop()
                a_moves.stop()
                
                #give count used to prevent Reachy from giving flyers too frequently 
                if give_count == 0:
                    
                    if hand_empty:
                        manip._target = [0.5,prev_y,prev_z]
                        manip.play('grab_flyer')
                        time.sleep(0.1)
                        manip.play('pull_flyer_adapted')
                        time.sleep(1.0)
                        hand_empty=False
                    
                    track = 0 
                    controller.start()
                    a_moves.start() 
                    
                    # after pulling the flyer, track a bit in case the person has moved
                    while track < 5:
                        if d._somebody_detected :
                            xM, yM, target_size = d._face_target
                            cmd_y, cmd_z = controller.track([cmd_y, cmd_z], [prev_y, prev_z], goal=center, input_controller=[xM, yM])
                            prev_y, prev_z = cmd_y, cmd_z 
                            track += 1
                            time.sleep(0.02)
                            nobody_here = False
                        else:
                            controller.set_new_target([prev_y, prev_z])
                            track+=1
                            nobody_here = True
                        
                    controller.stop()
                    a_moves.stop()
                    if nobody_here:
                        manip.play('has_been_ignored')
                        give_count = 5
                    else:
                        stiff_right(reachy)
                        manip._target = [0.5,prev_y,prev_z]
                        manip.play('hold_flyer_adapted')
                        time.sleep(0.01)
                        manip.play('give_flyer_adapted')
                        give_count = 40
                        hand_empty=True
                    
                give_count -=1
                time.sleep(0.1)
                controller.start()
                a_moves.start()
                
                xM, yM, _ = d._face_target
                cmd_y, cmd_z = controller.track([cmd_y, cmd_z], [prev_y, prev_z], goal=center, input_controller=[xM, yM])
                prev_y, prev_z = cmd_y, cmd_z 
                
    else:
        if track_count< 40:
            track_count+=1
            print('lost')
        else :
            d._prev_face_target = [0,0,0]
            track_count = 0
            controller.stop()
            a_moves.stop()
            time.sleep(0.2)
            print('random')
            idleForever.start()
            #idle.play('look_around',wait=True)
            idle_t = True
        
    
    time.sleep(0.02)                                    

d.stop()
controller.stop()
a_moves.stop()
#manip.stop()
idleForever.stop()
#idle.stop()

compliant(reachy)