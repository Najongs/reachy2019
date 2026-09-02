from reachy import parts, Reachy
import time
from pyquaternion import Quaternion
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

from reachy import Reachy, parts
import logging

logger = logging.getLogger('reachy.flyers')

base_pos_right_array = [13.4, -17.78, 31.692, -67.121, -101.906, -1.275, 33.871, 18.328]
base_pos_left_array = [10.4, 17.1, -45.934, -51.648, 12.757, -58]

def gather(*trajs):
    for traj in trajs:
        for t in traj:
            t.wait()
    
    
def base_pos(reachy,wait):
    
    return reachy.goto({
        'right_arm.shoulder_pitch': -11.5,
        'right_arm.shoulder_roll': -7,
        'right_arm.arm_yaw': 25.5,    
        'right_arm.elbow_pitch': -69,
        'right_arm.hand.forearm_yaw': -99,
        'right_arm.hand.wrist_pitch': -5,
        'right_arm.hand.wrist_roll': 13,
        
        'left_arm.shoulder_pitch': -17,
        'left_arm.shoulder_roll': 10,
        'left_arm.arm_yaw': -88,    
        'left_arm.elbow_pitch': -83,
        'left_arm.hand.forearm_yaw': 66,
        'left_arm.hand.wrist_pitch': 14,
        'left_arm.hand.wrist_roll': -8.5,
    }, duration=1.5, wait=wait)


def base_pos_right(reachy):
    
    #base_pos_right = [13.4, -17.78, 31.692, -67.121, -101.906, -1.275, 33.871, 18.328]
    reachy.goto({
        m.name: j
        for j, m in zip(base_pos_right_array, reachy.right_arm.motors)
    }, duration=1.5, wait=True, interpolation_mode='minjerk')
    
    for m in reachy.right_arm.motors:
        m.compliant = True
        
def base_pos_right_ignored(reachy):
    
    #base_pos_right = [13.4, -17.78, 31.692, -67.121, -101.906, -1.275, 33.871]
    reachy.goto({
        m.name: j
        for j, m in zip(base_pos_right_array[:7], reachy.right_arm.motors[:7])
    }, duration=1.5, wait=True, interpolation_mode='minjerk')
    
    for m in reachy.right_arm.motors[:7]:
        m.compliant = True
    
def base_pos_left(reachy):
    
    #base_pos_left = [10.4, 17.1, -45.934, -51.648, 12.757, -58]
    reachy.goto({
        m.name: j
        for j, m in zip(base_pos_left_array, reachy.left_arm.motors)
    }, duration=2, wait=True, interpolation_mode='minjerk')
    
    for m in reachy.left_arm.motors:
        m.compliant = True
    
    
def head_home(reachy,wait):
    
    traj1 = reachy.head.look_at(0.5,0,0,duration=1.5,wait=False)
    traj2 = reachy.goto({
        'head.right_antenna' : 0,
        'head.left_antenna' : 0,
    }, duration = 1.5,wait=wait)
    
    if wait:
         gather(traj1)
    return traj1+traj2


def grab_flyer(reachy): #MODIFIED
    
    right_arm_step1 = [-0.5, -56.549, 52.44, -117.319, -106.012, 28.527, -13.343, -23.607]
    right_arm_step2 = [-5.6, -40.901, 62.022, -114.593, -108.944, 31.78, -24.487, -48.827]
    right_arm_step3 = [  -3.67,  -36.34,   62.6,  -101.25,  -99.04,   33.76,  -12.57,  -48.83]
    right_arm_step4 = [  -0.86,  -45.01,   59.6,  -103.67, -100.,     31.02,  -12.31,  -48.83]
    
    left_goal_position = [9.2, 34.747, -57.89, -89.451, 12.757, -41.187]
    
    for m in reachy.left_arm.motors:
        m.compliant = False
    
    for m in reachy.right_arm.motors:
        m.compliant = False
    
    traj_left = reachy.goto({
        m.name: j
        for j, m in zip(left_goal_position, reachy.left_arm.motors)
    }, duration=1, wait=False, interpolation_mode='minjerk')
    
    time.sleep(0.2)
    
    traj_look = reachy.head.look_at(0.5,0.1,-0.4, duration=0.9,wait=False)
    
    gather(
        traj_look,

        reachy.goto({
            m.name: j
            for j, m in zip(right_arm_step1, reachy.right_arm.motors)
        }, duration=1, wait=False, interpolation_mode='minjerk')
    )
    
    traj_look = reachy.head.look_at(0.5,-0.15,-0.4, duration=0.6,wait=False)
    
    reachy.goto({
        m.name: j
        for j, m in zip(right_arm_step2, reachy.right_arm.motors)
    }, duration=0.5, wait=True)
    
    r_a = random.randint(-40,40)
    l_a = random.randint(-40,40)
    
    trajs = (  
        reachy.goto({   
            'head.right_antenna' : r_a,
            'head.left_antenna' : l_a,
        },duration = 1, wait=False)
    )
    
    reachy.goto({
        m.name: j
        for j, m in zip(right_arm_step3, reachy.right_arm.motors)
    }, duration=0.5, wait=True)
    
    reachy.goto({
        m.name: j
        for j, m in zip(right_arm_step4, reachy.right_arm.motors)
    }, duration=0.5, wait=True)
    
    reachy.goto({
        'right_arm.hand.gripper': 90,
    }, duration = 0.5, wait=True)
    
    gather(trajs,traj_look,traj_left)
    

def pull_flyer_adapted(reachy,x,y,z):
    
    y = np.clip(y,-2,2)
    z = np.clip(z,-0.5,0.3)
   
    right_arm_step5 = [   2.78 , -56.09,   49.91, -125. ,  -100. ,    33.51 ,  -9.38 ,  90.  ]
    
    traj_right = reachy.goto({
        m.name: j
        for j, m in zip(right_arm_step5, reachy.right_arm.motors)
    }, duration=0.7, wait=False)
    
    time.sleep(0.3)
    
    traj_look = reachy.head.look_at(x, y, z, duration=1, wait=False)
    
    gather(traj_right,traj_look)
    
    return True

def hold_flyer_adapted(reachy,x,y,z):
    
    r_a = random.randint(-40,40)
    l_a = random.randint(-40,40)
        
    traj_ant = reachy.goto({   
    'head.right_antenna' : r_a,
    'head.left_antenna' : l_a,
    },duration = 1, wait=False)
    
    y = np.clip(y,-2,2)
    z = np.clip(z,-0.5,0.3)
    
    y_mod =  np.clip(y,-0.4, 0.3)
    z_mod =  np.clip(z,-0.15, 0.2) 
    
    y_mod = np.round(round(y_mod/0.05)*0.05,2)
    z_mod = np.round(round(z_mod/0.05)*0.05,2)
    
    
    Tab = np.load('/home/pi/dev/reachy-flyers/reachy_flyers/behavior/hold_inv_kin.npy')
    JA = np.ndarray.tolist([tab for (y1,z1,tab) in Tab if np.logical_and(y1==y_mod,z1==z_mod)==True][0])
    
    #base_pos_left = [10.4, 17.1, -45.934, -51.648, 12.757, -58] 
    
    traj_left = reachy.goto({
        m.name: j
        for j, m in zip(base_pos_left_array, reachy.left_arm.motors)
    }, duration=1, wait=False, interpolation_mode='minjerk')
    
    time.sleep(0.5)
    
    gather(
        traj_ant,
        
        reachy.head.look_at(x,y,z,duration=1.7, wait=False),
    
        # Movement
        reachy.goto({
            m.name: j
            for j, m in zip(JA, reachy.right_arm.motors)
        }, duration=1.5, wait=True, interpolation_mode='minjerk')
    )
    
    gather(traj_left)
    
    for m in reachy.left_arm.motors:
        m.compliant = True
        
    logger.info('End of hold flyer')


def give_flyer_adapted(reachy,x,y,z):
    
    y = np.clip(y,-2,2)
    z = np.clip(z,-0.5,0.3)
    
    logger.info('Beginning of give_flyer')
    reachy.goto({'right_arm.hand.gripper': 5}, duration = 0.4, wait= True)
    
    time.sleep(0.4)
    
    #base_pos_right = [13.4, -17.78, 31.692, -67.121, -101.906, -1.275, 33.871, 18.328]
    traj_arm = reachy.goto({
        m.name: j
        for j, m in zip(base_pos_right_array, reachy.right_arm.motors)
    }, duration=1.5, wait=False, interpolation_mode='minjerk')
   
    rand_action = random.randint(0,2)
    
    if(rand_action==1):
        reachy.head.look_at(x,y,-0.2, duration=0.5, wait=True)
        reachy.head.look_at(x,y,z, duration=0.7, wait=True)

        q = reachy.head.neck.model.find_quaternion_transform([1, 0, 0], [x, y, z])
        qA = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(np.sign(y)*5))
        q = q*qA
        reachy.head.neck.orient(q, duration=0.7, wait=False)

        time.sleep(0.2)

        dur = 1
        t = np.linspace(0, dur, dur * 100)
        pos = 10 * np.sin(2 * np.pi * 5 * t)

        for p in pos:
            reachy.head.left_antenna.goal_position = p
            reachy.head.right_antenna.goal_position = -p
            time.sleep(0.01)
            
        reachy.head.look_at(x,y,z, duration=0.5, wait=True)

        time.sleep(0.1)
        
    elif(rand_action==2):
        q = reachy.head.neck.model.find_quaternion_transform([1, 0, 0], [x, y, z])
        qA = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(np.sign(y)*10))
        q = q*qA
        traj_head = reachy.head.neck.orient(q, duration=1, wait=False)

        reachy.goto({
            'head.left_antenna' : 30,
            'head.right_antenna' : 22,
        }, duration = 1, wait= True)

        time.sleep(0.4)

        dur = 1
        t = np.linspace(0, dur, dur * 100)
        pos = 10 * np.sin(2 * np.pi * 5 * t)

        for p in pos:
            reachy.head.left_antenna.goal_position = p
            reachy.head.right_antenna.goal_position = -p
            time.sleep(0.01)

        time.sleep(0.2)
        
        gather(traj_head)

        reachy.head.look_at(x,y,z, duration=0.5, wait=True)
    
    else:
        q = reachy.head.neck.model.find_quaternion_transform([1, 0, 0], [x, y, z])
        qA = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(np.sign(y)*10))
        q = q*qA
        reachy.head.neck.orient(q, duration=0.7, wait=True)

        traj_ant = reachy.goto({
            'head.left_antenna' : 10,
            'head.right_antenna' : 20,
        },duration = 0.3,wait = False)

        time.sleep(0.5)

        traj_head = reachy.head.look_at(x,y,z, duration=0.5, wait=False)

        time.sleep(0.5)

        dur = 1
        t = np.linspace(0, dur, dur * 100)
        pos = 20 * np.sin(4 * np.pi * t)
        
        gather(traj_ant)

        for p in pos:
            reachy.head.left_antenna.goal_position = p
            reachy.head.right_antenna.goal_position = p
            time.sleep(0.01)
        
        gather(traj_head)

        reachy.head.look_at(x,y,-0.2, duration=0.5, wait=True)
        reachy.head.look_at(x,y,z, duration=0.7, wait=True)
    
    gather(traj_arm)
    for m in reachy.right_arm.motors:
        m.compliant = True

    time.sleep(0.01)
    
    

def has_been_ignored(reachy):
    
    base_pos_left(reachy)
    base_pos_right_ignored(reachy)
    
    pos = [
        (-0.5, 150),
        (-0.4, 110),
        (-0.5, 150),
        (0, 90),
        (0, 20),
    ]

    for (z, antenna_pos) in pos:
        gather(
            reachy.head.look_at(0.5, 0.0, z, duration=1.0, wait=False),
            reachy.goto({
                'head.left_antenna': antenna_pos,
                'head.right_antenna': -antenna_pos,
            }, duration=1.5, wait=True, interpolation_mode='minjerk')
        )
    
    time.sleep(1)
    
    reachy.goto({   
        'head.right_antenna' : 0,
        'head.left_antenna' : 0,
    }, duration = 1, wait=True)
    
    
def look_around(reachy):
    x = 0.5
    y = (2 * np.random.rand() - 1) * 0.4
    z = (2 * np.random.rand() - 1) * 0.1

    duration = 1

    traj = reachy.head.look_at(x, y, z, duration=duration, wait=False)

    real = []

    t0 = time.time()
    while time.time() - t0 < duration:
        real.append([d.rot_position for d in reachy.head.neck.disks])
        time.sleep(0.01)
    
    gather(traj)
    
    rand_time = random.randrange(0,15)/10
    time.sleep(rand_time)
    
    
def read_flyer(reachy):
    
    for m in reachy.right_arm.motors:
        m.compliant = False
    
    gather(
        reachy.goto({
        'right_arm.shoulder_pitch': -34,
        'right_arm.shoulder_roll': -52,
        'right_arm.arm_yaw': 60,    
        'right_arm.elbow_pitch': -87,
        'right_arm.hand.forearm_yaw': -89.5,
        'right_arm.hand.wrist_pitch': 57,
        'right_arm.hand.wrist_roll': 26.5,
        }, duration=2, wait=False, interpolation_mode='minjerk'),
    
        reachy.head.look_at(0.4,-0.2,-0.5,duration =2, wait=True)
    )
    
    time.sleep(1)

    traj_look = head_home(reachy,False)
    base_pos_right_ignored(reachy)
    
    gather(traj_look)
    
    
    
def stretch_head(reachy):
    q1 = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(15))
    q2 = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(-15))
    reachy.head.neck.orient(q1, duration=1, wait=True)
    reachy.head.neck.orient(q2, duration=1, wait=True)
    
    head_home(reachy,True)

    
def lonely(reachy):
    
    reachy.head.look_at(0.5,-0.4,0,duration=1,wait=True)
    time.sleep(0.2)
    reachy.head.look_at(0.5,0.4,0,duration=1,wait=True)
    time.sleep(0.2)
    
    traj = reachy.goto({   
            'head.right_antenna' : -150,
            'head.left_antenna' : 150,
        },duration = 0.5, wait=False)
    
    reachy.head.look_at(0.5,0.05,-0.2,duration=1,wait=True)
    reachy.head.look_at(0.5,-0.05,-0.2,duration=1,wait=True)
    reachy.head.look_at(0.5,0.05,-0.2,duration=1,wait=True)
    
    gather(traj)
    
    head_home(reachy,True)
    
        
def look_hand(reachy):
    
    for m in reachy.right_arm.motors:
        m.compliant = False
        
    qA = [-11.5,-7,25.5,-69,-99,-5,13,18]
    A = reachy.right_arm.forward_kinematics(joints_position=qA)

    x = random.randint(20,35)/100
    y = random.randint(-40,0)/100
    z = random.randint(-10,0)/100
    
    pos = [x,y,z]

    B = A.copy()
    for i in range(3):
        B[i][3] = pos[i]

    JB = reachy.right_arm.inverse_kinematics(B, q0=qA)
    
    gather(
        reachy.goto({
            m.name: j
            for j, m in zip(JB, reachy.right_arm.motors)
        }, duration=1.2, wait=False, interpolation_mode='minjerk'),

        reachy.head.look_at(x,y,z-0.1,duration=1,wait=True)
    )
    
    s=1
    for i in range(4):
        s = -s
        reachy.goto({
            'right_arm.hand.forearm_yaw' :reachy.right_arm.hand.forearm_yaw.present_position + 30*s
        }, duration = 0.7, wait = True)
        time.sleep(0.1)
    
    reachy.right_arm.hand.gripper.goal_position = 50
    time.sleep(0.1)
    reachy.right_arm.hand.gripper.goal_position = 0
    time.sleep(0.1)
    reachy.right_arm.hand.gripper.goal_position = 50
    time.sleep(0.1)
    reachy.right_arm.hand.gripper.goal_position = 0
    
    time.sleep(0.2)
    
    traj_head = head_home(reachy,False)
    base_pos_right(reachy)
    
    gather(traj_head)

    
    
def waiting(reachy):
    
    for m in reachy.right_arm.motors:
        m.compliant = False
    
    right_arm = [57.4, -45.912, 25.89, -126.637, -80.792, 9.011, 15.689, 17.449]
    
    gather(
        reachy.head.look_at(0.5,-0.4,0,duration=1,wait=False),
    
        reachy.goto({
            m.name: j
            for j, m in zip(right_arm, reachy.right_arm.motors)
        }, duration=1.5, wait=True, interpolation_mode='minjerk')
    )
    
    reachy.head.look_at(0.5,0.4,0,duration=1.5,wait=True)
    
    traj_head = head_home(reachy,False)
    base_pos_right(reachy)
    gather(traj_head)

    
def pull_flyer_idle(reachy):
    
    for m in reachy.right_arm.motors:
        m.compliant = False
    for m in reachy.left_arm.motors:
        m.compliant = False
        
    grab_flyer(reachy)
    pull_flyer_adapted(reachy,0.5,0,0)
    
    base_pos_left(reachy)
    base_pos_right_ignored(reachy)
    
    
def show_flyer(reachy):

    for m in reachy.right_arm.motors:
        m.compliant = False
        
    hold_flyer_adapted(reachy,0.5,-0.1,0)
    hold_flyer_adapted(reachy,0.5,0.1,0)
    base_pos_right_ignored(reachy)
    

def do_not_give_flyer(reachy,x,y,z):
    
    for m in reachy.right_arm.motors:
        m.compliant = False
    
    q0 = [-25,-7.5,42,-110,120,8,-38,18]
    reachy.head.look_at(0.5,0.05,-0.4, duration=0.7,wait=True)
    time.sleep(0.5)
    reachy.head.look_at(x,y,z,duration=0.7, wait=True)
    reachy.goto({
                m.name: j
                for j, m in zip(q0, reachy.right_arm.motors)
            }, duration=2, wait=True, interpolation_mode = 'minjerk')
    time.sleep(0.1)
    reachy.goto({'right_arm.hand.wrist_pitch': 50},duration = 0.5, wait=True, interpolation_mode = 'minjerk')
    reachy.goto({'right_arm.hand.wrist_pitch': 10},duration = 0.5, wait=True, interpolation_mode = 'minjerk')
    reachy.goto({'right_arm.hand.wrist_pitch': 50},duration = 0.5, wait=True, interpolation_mode = 'minjerk')
    reachy.goto({'right_arm.hand.wrist_pitch': 10},duration = 0.5, wait=True, interpolation_mode = 'minjerk')
    
    time.sleep(0.2)
    base_pos_right(reachy)

    
def initialize_gripper_threshold(reachy):
    
    for m in reachy.right_arm.motors:
        m.compliant = False
    
    qA = [-11.5,-7,25.5,-69,-99,-5,13,18]
    A = reachy.right_arm.forward_kinematics(joints_position=qA)

    x = random.randint(20,35)/100
    y = random.randint(-30,-10)/100
    z = random.randint(-10,0)/100
    
    pos = [x,y,z]

    B = A.copy()
    for i in range(3):
        B[i][3] = pos[i]

    JB = reachy.right_arm.inverse_kinematics(B, q0=qA)
    
    gather(
        reachy.goto({
        m.name: j
        for j, m in zip(JB, reachy.right_arm.motors)
        }, duration=1.2, wait=False, interpolation_mode='minjerk'),
    
        reachy.head.look_at(x,y,z-0.1,duration=1,wait=False)
    )
    
    s=1
    for i in range(4):
        s = -s
        reachy.goto({
            'right_arm.hand.forearm_yaw' :reachy.right_arm.hand.forearm_yaw.present_position + 30*s
        }, duration = 0.7, wait = True)
        time.sleep(0.1)
    
    reachy.right_arm.hand.gripper.goal_position = 50
    time.sleep(0.1)
    reachy.right_arm.hand.gripper.goal_position = 0
    time.sleep(0.1)
    reachy.right_arm.hand.gripper.goal_position = 50
    
    #base_pos_right = [13.4, -17.78, 31.692, -67.121, -101.906, -1.275, 33.871]
    
    gather(
        head_home(reachy,False),
        
        reachy.goto({
            m.name: j
            for j, m in zip(base_pos_right_array[:7], reachy.right_arm.motors[:7])
        }, duration=1.5, wait=False, interpolation_mode='minjerk')
    )
    
    time.sleep(0.1)
    
    reachy.right_arm.hand.gripper.goal_position = 0
    
    for m in reachy.right_arm.motors:
        m.compliant = True
