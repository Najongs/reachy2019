"""Validate and execute LLM-generated motions safely.

The LLM (an opus session on the broker) emits keyframes:

    [{"pose": {"right_arm.shoulder_pitch": -40, ...}, "duration": 1.5}, ...]

Nothing the model outputs is trusted. validate() whitelists joints, clamps
angles to this robot's real limits, stretches too-fast segments, and runs a
forward-kinematics sweep that rejects trajectories entering the torso, the
head, or crossing the body midline. MotionExecutor then plays the validated
keyframes with its own 50 Hz interpolation loop while pointing the head so
the moving hand stays in camera view.

Pure-python on purpose: the validator and its FK need only the standard
library, so they run identically on the Pi (3.7), on the DGX (for tests), and
inside the broker.
"""

import logging
import math
import time


logger = logging.getLogger(__name__)


# --- Static robot model -------------------------------------------------------
# Mirrors software/reachy/parts/arm.py + newcode/custom_hands.py, converted to
# the user frame with the compute_bounds formula ((limits if direct else
# -limits) - offset). Torso frame: x forward, y left, z up, meters.

# (joint name, link translation, rotation axis, (low, high) degrees)
RIGHT_CHAIN = [
    ('right_arm.shoulder_pitch', (0, -0.19, 0), (0, 1, 0), (-150, 90)),
    ('right_arm.shoulder_roll', (0, 0, 0), (1, 0, 0), (-180, 10)),
    ('right_arm.arm_yaw', (0, 0, 0), (0, 0, 1), (-90, 90)),
    ('right_arm.elbow_pitch', (0, 0, -0.28), (0, 1, 0), (-125, 0)),
    ('right_arm.hand.forearm_yaw', (0, 0, 0), (0, 0, 1), (-100, 100)),
    ('right_arm.hand.wrist_pitch', (0, 0, -0.25), (0, 1, 0), (-45, 45)),
    # wrist_roll physically absent; its -0.0325 offset is folded in here.
    ('right_arm.hand.gripper', (0, -0.01, -0.1075), (0, 0, 0), (-50, 10)),
]

LEFT_CHAIN = [
    ('left_arm.shoulder_pitch', (0, 0.19, 0), (0, 1, 0), (-150, 90)),
    ('left_arm.shoulder_roll', (0, 0, 0), (1, 0, 0), (-10, 180)),
    ('left_arm.arm_yaw', (0, 0, 0), (0, 0, 1), (-90, 90)),
    ('left_arm.elbow_pitch', (0, 0, -0.28), (0, 1, 0), (-125, 0)),
    ('left_arm.hand.forearm_yaw', (0, 0, 0), (0, 0, 1), (-100, 100)),
    ('left_arm.hand.wrist_pitch', (0, 0, -0.25), (0, 1, 0), (-45, 45)),
    # wrist_roll and gripper physically absent on the left arm.
]

CHAINS = {'right_arm': RIGHT_CHAIN, 'left_arm': LEFT_CHAIN}

ANTENNA_LIMITS = {
    'head.left_antenna': (-140, 140),
    'head.right_antenna': (-140, 140),
}

ALL_LIMITS = {}
for _chain in (RIGHT_CHAIN, LEFT_CHAIN):
    for _name, _t, _axis, _lim in _chain:
        ALL_LIMITS[_name] = _lim
ALL_LIMITS.update(ANTENNA_LIMITS)

# Safe operating range, TIGHTER than mechanical angle-limits, to avoid
# self-interference the joint limits alone don't catch (the upper arm hangs
# ~3 cm from the torso at rest). Tune against the real robot via the sim
# viewer calibration slider. Absent -> full mechanical limit.
SAFE_LIMITS = {
    'right_arm.shoulder_roll': (-80, 0),
    'left_arm.shoulder_roll':  (0, 80),
    'right_arm.arm_yaw': (-70, 70),
    'left_arm.arm_yaw':  (-70, 70),
}


def effective_limits(joint):
    lo, hi = ALL_LIMITS[joint]
    if joint in SAFE_LIMITS:
        slo, shi = SAFE_LIMITS[joint]
        lo, hi = max(lo, slo), min(hi, shi)
    return lo, hi

# Average deg/s caps (minjerk peaks at 1.875x the average, so these keep
# peaks under ~94 deg/s on the big joints).
SLOW_JOINTS = ('shoulder_pitch', 'shoulder_roll', 'arm_yaw', 'elbow_pitch')
SPEED_SLOW = 50.0
SPEED_FAST = 90.0

DURATION_MIN = 0.4
DURATION_MAX = 6.0
MAX_KEYFRAMES = 12
MAX_TOTAL_SECONDS = 20.0

# Keep-out volumes (torso frame, meters), checked for hand and wrist.
TORSO_BOX = {'x_max': 0.07, 'y_abs_max': 0.14, 'z_min': -0.60}
HEAD_SPHERE_CENTER = (0.0, 0.0, 0.09)
HEAD_SPHERE_RADIUS = 0.15
# Crossing the body midline is forbidden near the torso (clap-type moves),
# but allowed in the open space well in front of it - that is where the
# left hand's storage tray sits on the table.
MIDLINE_MARGIN = 0.04
MIDLINE_FREE_X = 0.20
MIDLINE_HARD_MAX = 0.22

NECK_POS = (0.0, 0.0, 0.09)

TEMPERATURE_LIMIT = 55

# Relaxed "ready" stance held between gestures: arms slightly forward with a
# soft elbow bend - looks attentive instead of dangling. Shoulder+elbow stay
# powered at low torque to hold it; everything else relaxes.
READY_POSE = {
    'right_arm.shoulder_pitch': -15, 'right_arm.shoulder_roll': -8,
    'right_arm.arm_yaw': 0, 'right_arm.elbow_pitch': -40,
    'right_arm.hand.forearm_yaw': 0, 'right_arm.hand.wrist_pitch': 0,
    'right_arm.hand.gripper': -15,
    'left_arm.shoulder_pitch': -15, 'left_arm.shoulder_roll': 8,
    'left_arm.arm_yaw': 0, 'left_arm.elbow_pitch': -40,
    'left_arm.hand.forearm_yaw': 0, 'left_arm.hand.wrist_pitch': 0,
}
# wrist_pitch must hold too: with the elbow bent, gravity folds a limp wrist
# and the hand droops. forearm_yaw keeps the hand's facing tidy (tiny load).
HOLD_JOINTS = ('shoulder_pitch', 'elbow_pitch', 'wrist_pitch', 'forearm_yaw')
HOLD_TORQUE = 50
# After this much idle time the arms lower to the hang pose and power off,
# so a quiet robot does not cook its shoulder motors holding a pose.
SETTLE_AFTER = 180.0

# Natural hanging pose used for full power-off. All zeros = the arm points
# straight down, which is where gravity leaves it anyway - so cutting power
# afterwards causes no visible drop.
REST_POSE = {
    'right_arm.shoulder_pitch': 0, 'right_arm.shoulder_roll': 0,
    'right_arm.arm_yaw': 0, 'right_arm.elbow_pitch': 0,
    'right_arm.hand.forearm_yaw': 0, 'right_arm.hand.wrist_pitch': 0,
    'right_arm.hand.gripper': 0,
    'left_arm.shoulder_pitch': 0, 'left_arm.shoulder_roll': 0,
    'left_arm.arm_yaw': 0, 'left_arm.elbow_pitch': 0,
    'left_arm.hand.forearm_yaw': 0, 'left_arm.hand.wrist_pitch': 0,
}


# --- Pure-python kinematics ---------------------------------------------------

def _rot(axis, deg):
    """3x3 rotation matrix about an arbitrary axis (Rodrigues), row-major."""
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    if n == 0:
        return ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    x, y, z = x / n, y / n, z / n

    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    C = 1 - c

    return (
        (c + x * x * C, x * y * C - z * s, x * z * C + y * s),
        (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
        (z * x * C - y * s, z * y * C + x * s, c + z * z * C),
    )


def _mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _mat_vec(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def forward_kinematics(chain, pose, upto=None):
    """Position of a chain point in the torso frame.

    Args:
        chain: RIGHT_CHAIN or LEFT_CHAIN
        pose (dict): full joint angles in degrees (missing joints read as 0)
        upto (int): use only the first `upto` links (None = whole chain / hand)

    Returns:
        (x, y, z) in meters. Matches software/reachy Chain.forward semantics:
        each link applies its translation in the parent frame, then rotates.
    """
    links = chain if upto is None else chain[:upto]

    p = (0.0, 0.0, 0.0)
    r = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    for name, translation, axis, _limits in links:
        p = tuple(p[i] + _mat_vec(r, translation)[i] for i in range(3))
        r = _mat_mul(r, _rot(axis, pose.get(name, 0.0)))

    return p


def minjerk(u):
    """Minimum-jerk ease on u in [0, 1] (zero endpoint velocity/accel)."""
    return u * u * u * (10 + u * (-15 + 6 * u))


# --- Validation ---------------------------------------------------------------

class ValidationError(Exception):
    """Motion rejected; str(e) is a short speakable reason."""


def _speed_cap(joint):
    return SPEED_SLOW if any(j in joint for j in SLOW_JOINTS) else SPEED_FAST


def _arm_of(joint):
    prefix = joint.split('.')[0]
    return prefix if prefix in CHAINS else None


def validate(moves, seed_pose, available_joints, check_collision=True):
    """Turn raw LLM keyframes into a safe, executable segment list.

    Args:
        moves (list): [{"pose": {joint: deg}, "duration": s}, ...]
        seed_pose (dict): current commanded angles for every available joint
        available_joints (iterable): joints that physically exist right now

    Returns:
        list of (pose_dict, duration) with clamped angles and stretched
        durations. Raises ValidationError when nothing safe remains.
    """
    if not isinstance(moves, list) or not moves:
        raise ValidationError('동작 내용이 비어 있어요')
    if len(moves) > MAX_KEYFRAMES:
        moves = moves[:MAX_KEYFRAMES]
        logger.warning('Keyframes truncated to %d', MAX_KEYFRAMES)

    available = set(available_joints)
    cleaned = []
    current = dict(seed_pose)
    dropped = set()

    for frame in moves:
        if not isinstance(frame, dict):
            raise ValidationError('동작 형식이 이상해요')

        # Grasp step: force-sensor based close/open, not an angle keyframe.
        if 'grasp' in frame and not frame.get('pose'):
            action = frame['grasp']
            if action not in ('close', 'open'):
                raise ValidationError('잡기 동작 값이 이상해요')
            if 'right_arm.hand.gripper' not in available:
                raise ValidationError('그리퍼가 없어서 잡기는 못 해요')
            cleaned.append(('grasp', action))
            continue

        if not isinstance(frame.get('pose'), dict):
            raise ValidationError('동작 형식이 이상해요')

        pose = {}
        for joint, angle in frame['pose'].items():
            if joint not in ALL_LIMITS:
                dropped.add(joint)
                continue
            if joint not in available:
                dropped.add(joint)
                continue
            if not isinstance(angle, (int, float)) or not math.isfinite(angle):
                raise ValidationError('동작 값이 이상해요')

            low, high = effective_limits(joint)
            pose[joint] = min(max(float(angle), low), high)

        if not pose:
            continue

        try:
            duration = float(frame.get('duration', 1.0))
        except (TypeError, ValueError):
            duration = 1.0
        if not math.isfinite(duration):
            duration = 1.0
        duration = min(max(duration, DURATION_MIN), DURATION_MAX)

        # Stretch the segment until every joint respects its speed cap.
        for joint, target in pose.items():
            delta = abs(target - current.get(joint, 0.0))
            needed = delta / _speed_cap(joint)
            if needed > duration:
                duration = min(needed, DURATION_MAX)

        current.update(pose)
        cleaned.append((pose, duration))

    if dropped:
        logger.warning('Dropped unknown/absent joints: %s', sorted(dropped))

    if not cleaned:
        raise ValidationError('할 수 있는 관절이 하나도 없어요')

    pose_segments = [seg for seg in cleaned if seg[0] != 'grasp']
    if not pose_segments and not any(seg[0] == 'grasp' for seg in cleaned):
        raise ValidationError('할 수 있는 관절이 하나도 없어요')

    total = sum(d for _, d in pose_segments)
    if total > MAX_TOTAL_SECONDS:
        raise ValidationError('동작이 너무 길어요')

    if check_collision:
        _collision_sweep(pose_segments, seed_pose)

    return cleaned


def _arm_points(chain, pose):
    """Points along the whole arm: shoulder base, elbow, wrist, hand, plus
    interpolated points on each segment (so a link, not just its endpoints,
    is collision-checked)."""
    # positions after each kinematic link
    js = [forward_kinematics(chain, pose, upto=i) for i in range(1, len(chain) + 1)]
    # start point (torso shoulder mount) = first link translation origin
    start = (0.0, chain[0][1][1], 0.0)   # (0, ±0.19, 0)
    nodes = [start] + js
    pts = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        for t in (0.0, 0.34, 0.67):
            pts.append(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))
    pts.append(nodes[-1])
    return pts


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


ARM_CLEARANCE = 0.09   # min centerline gap between the two arms (m)


def _check_point(point, side):
    """Raise if a hand/wrist point is inside a keep-out volume."""
    x, y, z = point

    if (x < TORSO_BOX['x_max'] and abs(y) < TORSO_BOX['y_abs_max']
            and z > TORSO_BOX['z_min']):
        raise ValidationError('몸에 부딪힐 것 같아요')

    dx = x - HEAD_SPHERE_CENTER[0]
    dy = y - HEAD_SPHERE_CENTER[1]
    dz = z - HEAD_SPHERE_CENTER[2]
    if math.sqrt(dx * dx + dy * dy + dz * dz) < HEAD_SPHERE_RADIUS:
        raise ValidationError('머리에 부딪힐 것 같아요')

    near_body = x < MIDLINE_FREE_X
    if side == 'right_arm':
        if (near_body and y > MIDLINE_MARGIN) or y > MIDLINE_HARD_MAX:
            raise ValidationError('팔이 몸을 가로지르는 동작은 못 해요')
    if side == 'left_arm':
        if (near_body and y < -MIDLINE_MARGIN) or y < -MIDLINE_HARD_MAX:
            raise ValidationError('팔이 몸을 가로지르는 동작은 못 해요')


def _collision_sweep(segments, seed_pose, samples_per_second=10):
    """FK-sample the whole interpolated trajectory through the keep-outs."""
    current = dict(seed_pose)

    for pose, duration in segments:
        targets = dict(current)
        targets.update(pose)

        steps = max(2, int(duration * samples_per_second))
        for step in range(1, steps + 1):
            k = minjerk(step / float(steps))
            sample = {
                joint: current.get(joint, 0.0)
                + (targets.get(joint, 0.0) - current.get(joint, 0.0)) * k
                for joint in set(current) | set(targets)
            }

            arm_pts = {}
            for side, chain in CHAINS.items():
                pts = _arm_points(chain, sample)
                arm_pts[side] = pts
                # keep-out volumes: check every point along this arm
                for p in pts:
                    _check_point(p, side)

            # arm-to-arm: reject if the two arms' links come too close
            for pr in arm_pts.get('right_arm', []):
                for pl in arm_pts.get('left_arm', []):
                    if _dist(pr, pl) < ARM_CLEARANCE:
                        raise ValidationError('두 팔이 서로 부딪힐 것 같아요')

        current = targets


# --- Execution ----------------------------------------------------------------

class MotionExecutor(object):
    """Play validated keyframes on the robot, head following the hand.

    Single 50 Hz thread does everything: interpolates arm joints (writing
    goal_position directly), and every 5th tick computes FK on the *commanded*
    pose (never the bus) to point the camera at the moving hand.

    Args:
        reachy (reachy.Reachy): connected robot (arms + head)
        freq (float): control rate in Hz
    """

    def __init__(self, reachy, freq=50):
        import threading

        self.reachy = reachy
        self.freq = freq

        self._lock = threading.Lock()
        self.abort = threading.Event()
        self._settle_timer = None

    # -- public ---------------------------------------------------------------

    def available_joints(self):
        """Joints that physically exist on the connected robot."""
        return [m.name for m in self.reachy.motors]

    def commanded_pose(self):
        """Read the robot's current position once (seed for interpolation)."""
        return {m.name: m.present_position for m in self.reachy.motors}

    def execute(self, segments):
        """Run validated segments. Returns (ok, spoken_reason_or_None).

        Refuses (without raising) when another motion is running or a motor
        is too hot. Always tries to return to rest and relax the arms.
        """
        self._cancel_settle()

        if not self._lock.acquire(False):
            return False, '지금 움직이는 중이에요, 잠깐만요'

        try:
            arm_motors = self._involved_arm_motors(segments)

            hot = [m for m in arm_motors
                   if (m.temperature or 0) > TEMPERATURE_LIMIT]
            if hot:
                logger.warning('Motors too hot: %s',
                               [(m.name, m.temperature) for m in hot])
                return False, '모터가 뜨거워서 조금 쉬어야 해요'

            self.abort.clear()
            self.max_divergence = 0.0

            seed = self.commanded_pose()

            self._stiffen(arm_motors, seed)
            self._play(segments, seed, follow_side=self._follow_side(segments))
            # Measure while still stiff - a relaxed arm drifts on friction
            # and would show phantom tracking error.
            self._log_tracking_error(arm_motors)

            self._enter_ready(arm_motors)
            self._home_head()
            self._schedule_settle()

            return True, None

        except Exception:
            logger.exception('Motion execution failed')
            try:
                self._settle(self._all_arm_motors())
            except Exception:
                logger.exception('Settle recovery failed too')
            return False, '동작하다가 문제가 생겨서 멈췄어요'
        finally:
            self._lock.release()

    # -- internals ------------------------------------------------------------

    def _involved_arm_motors(self, segments):
        joints = set()
        for seg in segments:
            if seg[0] == 'grasp':
                joints.add('right_arm.hand.gripper')
            else:
                joints.update(seg[0])

        motors = []
        for m in self.reachy.motors:
            if m.name in joints and _arm_of(m.name) is not None:
                motors.append(m)
            # Any other joint of the same arm must also be stiff, otherwise
            # gravity moves the uncommanded joints mid-gesture.
        arms = {_arm_of(m.name) for m in motors}
        return [m for m in self.reachy.motors if _arm_of(m.name) in arms]

    def _follow_side(self, segments):
        """Arm whose hand the camera should track (largest total travel)."""
        travel = {}
        last = {}
        for seg in segments:
            if seg[0] == 'grasp':
                travel['right_arm'] = travel.get('right_arm', 0.0) + 10.0
                continue
            pose = seg[0]
            for joint, angle in pose.items():
                side = _arm_of(joint)
                if side is None:
                    continue
                travel[side] = travel.get(side, 0.0) + abs(angle - last.get(joint, 0.0))
                last[joint] = angle
        if not travel:
            return None
        return max(travel, key=travel.get)

    def _stiffen(self, arm_motors, seed):
        """Power the arms without letting them jump.

        goal_position writes are silently dropped while compliant, so the
        stored target can be stale: write the seed back immediately after
        stiffening, before any torque builds up.
        """
        for m in arm_motors:
            m.compliant = False
        time.sleep(0.05)
        for m in arm_motors:
            m.goal_position = seed[m.name]
        for m in arm_motors:
            try:
                if m.name.endswith('.gripper'):
                    m.torque_limit = 40
                elif any(j in m.name for j in ('shoulder_pitch', 'shoulder_roll',
                                               'elbow_pitch')):
                    # Lifting the whole arm against gravity stalls at 70%.
                    m.torque_limit = 90
                else:
                    m.torque_limit = 70
            except Exception:
                logger.warning('torque_limit not settable for %s', m.name)
        time.sleep(0.05)

    def _relax_arms(self, arm_motors):
        """Release only the arms; the head stays stiff for the voice loop."""
        for m in arm_motors:
            m.compliant = True

    def _all_arm_motors(self):
        return [m for m in self.reachy.motors if _arm_of(m.name) is not None]

    def _go_pose(self, arm_motors, pose_table):
        """Glide the given motors to their entries in pose_table, slowly."""
        involved = {m.name for m in arm_motors}
        target = {j: a for j, a in pose_table.items() if j in involved}
        if not target:
            return

        seed = {m.name: m.present_position for m in arm_motors}
        delta = max(abs(target[j] - seed.get(j, 0.0)) for j in target)
        # Slow and deliberate: the gesture is over, this is the wind-down.
        # Higher floor so the return to base never feels rushed.
        duration = min(max(delta / 25.0, 2.0), 4.5)
        self._play([(target, duration)], seed, follow_side=None)

    def _enter_ready(self, gesture_motors):
        """Move BOTH arms into the ready stance and hold it cheaply.

        Shoulder+elbow keep power at HOLD_TORQUE so the forward/bent look
        stays; every other joint relaxes.
        """
        motors = self._all_arm_motors()

        extra = [m for m in motors if m not in gesture_motors]
        if extra:
            seed = {m.name: m.present_position for m in extra}
            self._stiffen(extra, seed)

        self._go_pose(motors, READY_POSE)

        for m in motors:
            if any(j in m.name for j in HOLD_JOINTS):
                try:
                    m.torque_limit = HOLD_TORQUE
                except Exception:
                    pass
            else:
                m.compliant = True

    def hold_ready(self):
        """Public: bring the arms to the ready stance (e.g. at startup)."""
        self._cancel_settle()
        if not self._lock.acquire(False):
            return
        try:
            self._enter_ready([])
            self._schedule_settle()
        except Exception:
            logger.exception('hold_ready failed')
        finally:
            self._lock.release()

    def _settle(self, arm_motors):
        """Lower to the hang pose and power everything off."""
        seed = {m.name: m.present_position for m in arm_motors}
        self._stiffen(arm_motors, seed)
        self._go_pose(arm_motors, REST_POSE)
        self._relax_arms(arm_motors)

    def _schedule_settle(self):
        import threading

        self._cancel_settle()
        self._settle_timer = threading.Timer(SETTLE_AFTER, self._settle_quietly)
        self._settle_timer.daemon = True
        self._settle_timer.start()

    def _cancel_settle(self):
        if self._settle_timer is not None:
            self._settle_timer.cancel()
            self._settle_timer = None

    def _settle_quietly(self):
        """Timer callback: idle too long, lower the arms and relax."""
        if not self._lock.acquire(False):
            return
        try:
            logger.info('Idle for %.0fs - settling arms down', SETTLE_AFTER)
            self._settle(self._all_arm_motors())
        except Exception:
            logger.exception('Idle settle failed')
        finally:
            self._lock.release()

    def shutdown(self):
        """Lower and relax the arms before program exit."""
        self._cancel_settle()
        with self._lock:
            try:
                self._settle(self._all_arm_motors())
            except Exception:
                logger.exception('Shutdown settle failed')

    def _play(self, segments, seed, follow_side):
        """The 50 Hz loop: interpolate + write arms, follow with the head."""
        from say_and_move import point_head

        motor_by_name = {m.name: m for m in self.reachy.motors}
        current = dict(seed)
        gaze = list(getattr(self.reachy.head, '_soft_gaze', (0.0, 0.0)))

        dt = 1.0 / self.freq
        chain = CHAINS.get(follow_side)

        # Ramp from wherever the arm is to the first pose keyframe.
        plan = list(segments)
        for i, seg in enumerate(plan):
            if seg[0] != 'grasp':
                first_pose = seg[0]
                delta = max(abs(first_pose[j] - current.get(j, 0.0))
                            for j in first_pose)
                ramp = min(max(delta / 40.0, 1.0), 3.0)
                plan[i] = (first_pose, ramp)
                break

        t_start = time.time()
        # Ease the head-follow in over a longer window so the neck never
        # lurches toward the hand when a gesture starts.
        follow_ramp = 1.2
        follow_ramp_end = t_start + follow_ramp
        tick = 0

        for seg in plan:
            if seg[0] == 'grasp':
                self._do_grasp(seg[1])
                continue

            pose, duration = seg
            targets = dict(current)
            targets.update(pose)
            seg_start = time.time()

            while not self.abort.is_set():
                now = time.time()
                u = (now - seg_start) / duration
                if u >= 1.0:
                    break
                k = minjerk(u)

                for joint in pose:
                    a = current.get(joint, 0.0)
                    value = a + (targets[joint] - a) * k
                    motor = motor_by_name.get(joint)
                    if motor is not None:
                        motor.goal_position = value

                # Head follow: FK every 5th tick, smooth toward it at 50 Hz.
                if chain is not None:
                    if tick % 5 == 0:
                        sample = {
                            j: current.get(j, 0.0) + (targets.get(j, current.get(j, 0.0))
                                                      - current.get(j, 0.0)) * k
                            for j in set(current) | set(targets)
                        }
                        self._gaze_target = self._hand_gaze(chain, sample)

                    target = getattr(self, '_gaze_target', None)
                    if target is not None:
                        # Slower smoothing (tau ~0.35s) keeps the gaze gliding
                        # rather than tracking the hand abruptly.
                        alpha = min(1.0, dt / 0.35)
                        if now < follow_ramp_end:
                            alpha *= (now - t_start) / follow_ramp
                        gaze[0] += (target[0] - gaze[0]) * alpha
                        gaze[1] += (target[1] - gaze[1]) * alpha
                        point_head(self.reachy, 0.5, gaze[0], gaze[1], tilt=False)

                tick += 1
                next_t = seg_start + (int((now - seg_start) / dt) + 1) * dt
                sleep = next_t - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                elif sleep < -dt * 0.5:
                    logger.debug('tick overrun %.1f ms', -sleep * 1000)

            if self.abort.is_set():
                logger.warning('Motion aborted')
                break

            # Land exactly on the keyframe.
            for joint, value in pose.items():
                motor = motor_by_name.get(joint)
                if motor is not None:
                    motor.goal_position = value
            current = targets

            # Reality check at each keyframe: a few bus reads of the followed
            # arm. If it stalled (gravity, obstacle), gaze and interpolation
            # continue from where the arm REALLY is, not the fantasy pose.
            if chain is not None:
                divergence = 0.0
                for joint in list(current):
                    if not joint.startswith(follow_side):
                        continue
                    motor = motor_by_name.get(joint)
                    if motor is None:
                        continue
                    actual = motor.present_position
                    divergence = max(divergence, abs(actual - current[joint]))
                    current[joint] = actual
                if divergence > self.max_divergence:
                    self.max_divergence = divergence
                if divergence > 25:
                    logger.warning('Arm lagging command by %.0f deg - '
                                   'stall? (gravity/obstacle)', divergence)

        self._gaze_target = None

    def _home_head(self, duration=1.8):
        """Glide the gaze and antennas back to neutral after a gesture.

        Hand-follow leaves the head aimed at wherever the hand ended, and some
        presets end with the antennas up - without this the robot keeps
        staring sideways until idle happens to recenter it.
        """
        from say_and_move import point_head

        head = self.reachy.head
        y0, z0 = getattr(head, '_soft_gaze', (0.0, 0.0))
        left0 = head.left_antenna.goal_position
        right0 = head.right_antenna.goal_position

        steps = max(2, int(duration * self.freq))
        for i in range(1, steps + 1):
            if self.abort.is_set():
                break
            k = 0.5 - 0.5 * math.cos(math.pi * i / steps)
            point_head(self.reachy, 0.5, y0 * (1 - k), z0 * (1 - k))
            head.left_antenna.goal_position = left0 * (1 - k)
            head.right_antenna.goal_position = right0 * (1 - k)
            time.sleep(1.0 / self.freq)

    def _do_grasp(self, action):
        """Force-sensor grasp via the stock ForceGripper close()/open().

        close() ramps the gripper until the load cell reports contact and
        returns whether something was actually caught; we remember that on
        self.last_grasp_ok so the caller can react ("못 잡았어요").
        """
        hand = getattr(getattr(self.reachy, 'right_arm', None), 'hand', None)
        if hand is None:
            logger.warning('No right hand for grasp step')
            return

        try:
            if action == 'close':
                self.last_grasp_ok = bool(hand.close())
                logger.info('Grasp close -> caught=%s', self.last_grasp_ok)
            else:
                hand.open()
                self.last_grasp_ok = None
                logger.info('Grasp open')
        except Exception:
            logger.exception('Grasp step failed')

    def _hand_gaze(self, chain, pose):
        """Gaze (y, z at x=0.5) that keeps the hand low in the camera frame."""
        px, py, pz = forward_kinematics(chain, pose)

        vx = px - NECK_POS[0]
        vy = py - NECK_POS[1]
        vz = pz - NECK_POS[2]

        vx = max(vx, 0.15)
        scale = 0.5 / vx
        y = min(max(vy * scale, -0.40), 0.40)
        z = min(max(vz * scale + 0.06, -0.35), 0.15)
        return (y, z)

    def _log_tracking_error(self, arm_motors):
        """One bus read at the end: how far reality lags its own command.

        goal_position still holds the last commanded value for each motor, so
        present-vs-goal is the true tracking error regardless of what pose the
        gesture ended in.
        """
        worst = 0.0
        worst_joint = None
        for m in arm_motors:
            try:
                err = abs(m.present_position - m.goal_position)
            except Exception:
                continue
            if err > worst:
                worst, worst_joint = err, m.name
        logger.info('Post-motion tracking error: %.1f deg (%s)', worst, worst_joint)
