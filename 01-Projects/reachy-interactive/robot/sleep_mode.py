"""밤에는 재운다 - 불이 꺼지고 아무도 없으면 소리도 동작도 내지 않는다.

빈 사무실에서 로봇이 혼자 인사하고 팔을 움직이면 무섭다. 그래서 어두운데
사람도 없으면 '자는' 상태로 들어가, 스스로는 아무 소리도 내지 않고 움직이지도
않는다.

**통신은 끄지 않는다.** 터널, 브로커, ssh, 뷰어, 상태 미러, 로그, 카메라
감시는 자는 동안에도 그대로 돌아간다. 자는 동안 멈추는 것은 로봇이 '스스로
시작하는' 소리와 움직임뿐이다 - 원격으로 들여다보고 손보는 길은 열려 있다.

깨우는 조건은 셋이다. 어느 하나만 맞아도 깬다.

  * 불이 켜진다        - 밝기가 문턱을 넘어 잠시 유지되면
  * 사람이 보인다      - 사람 검출기가 잡으면 (불이 어느 정도 있어야 보인다)
  * 말을 건다          - 알아들은 말이 잡음이 아니면

카메라를 못 쓸 때를 위한 안전망으로 '조용한 시간대'도 둔다. 퇴근할 때 모터
전원을 내리면 로봇 연결 자체가 실패해 카메라도 없다 - 그러면 밝기를 볼 수
없으므로, 밝기만 믿는 설계는 정작 필요한 밤에 작동하지 않는다. 그 시간대에는
사람이 안 보이면(또는 볼 눈이 없으면) 잔다. 말을 걸면 그때도 깬다.

마지막 조건이 중요하다. 불이 꺼진 채로도 사람이 말을 걸면 대답해야 한다.
잡음으로는 깨지 않는다 - 어두운 복도에서 로봇이 혼자 말하기 시작하는 것이
바로 피하려던 일이다.

문턱값은 현장마다 다르다. presence.PresenceWatcher 가 화면 밝기를 5분마다
로그에 남기므로, 밤낮 값을 보고 --dark-below / --light-above 로 맞춘다.
"""

import logging
import time


logger = logging.getLogger('reachy.sleep')


class SleepWatcher(object):
    """불이 꺼지고 사람이 없으면 자야 한다고 알려 준다.

    스스로 스레드를 돌리지 않는다. 대화 루프와 인사 스레드가 각자 돌다가
    asleep 을 물어보는 방식이라, 잠들고 깨는 판단이 한 군데에만 있다.

    Args:
        watcher (PresenceWatcher): brightness / person 을 읽어 온다
        dark_below (float): 이보다 어두우면 '불이 꺼졌다'
        light_above (float): 이보다 밝으면 '불이 켜졌다'. dark_below 보다
            높게 둔다 - 문턱 하나로 재면 경계에서 껐다 켰다 한다
        quiet_for (float): 어둡고 사람도 없는 상태가 이만큼 이어져야 잠든다
        wake_for (float): 밝아진 상태가 이만큼 이어져야 깬다 (지나가는
            자동차 불빛, 휴대폰 화면 따위로 깨지 않게)
        quiet_hours ((int, int)): (시작시, 끝시) 이 사이에는 사람이 안 보이면
            밝기와 무관하게 잔다. 카메라가 없을 때의 유일한 안전망이다.
            None 이면 시간대 규칙을 쓰지 않는다. 기본 (18, 9) = 근무시간
            아침 9시~저녁 6시 밖에는 사람이 없으면 잔다 (사용자 지정
            2026-09-08).
        watcher: None 이어도 된다 (카메라 없이 시간대만으로 판단)
    """

    def __init__(self, watcher, dark_below=0.15, light_above=0.25,
                 quiet_for=180.0, wake_for=3.0, quiet_hours=(18, 9)):
        self.watcher = watcher
        self.quiet_hours = quiet_hours
        self.dark_below = dark_below
        self.light_above = light_above
        self.quiet_for = quiet_for
        self.wake_for = wake_for

        self._asleep = False
        self._dark_since = None     # 어둡고 사람도 없어진 시각
        self._light_since = None    # 밝아진 시각
        self._changed_at = 0.0

    @property
    def asleep(self):
        return self._asleep

    def brightness(self):
        """지금 화면 밝기. 아직 프레임을 못 받았으면 None."""
        return getattr(self.watcher, 'brightness', None) if self.watcher else None

    def wake(self, why):
        """밖에서 깨운다 (사람이 말을 걸었을 때).

        근무시간 밖에는 깨우지 못한다 - 무조건 절전 (사용자 지정).
        """
        if self.in_quiet_hours():
            return False
        self._dark_since = None
        if not self._asleep:
            return False
        self._asleep = False
        self._changed_at = time.time()
        logger.info('깨어납니다 (%s)', why)
        return True

    def in_quiet_hours(self, now=None):
        """지금이 '조용한 시간대'인가. 자정을 넘는 구간도 다룬다."""
        if not self.quiet_hours:
            return False
        start, end = self.quiet_hours
        hour = time.localtime(now if now is not None else time.time()).tm_hour
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end      # 20시~8시처럼 자정을 넘는 구간

    def update(self):
        """지금 자야 하는지 다시 판단한다. asleep 을 돌려준다.

        대화 루프가 한 바퀴 돌 때마다 부른다. 값을 읽기만 하므로 싸다.
        """
        now = time.time()
        level = self.brightness()
        person = bool(getattr(self.watcher, 'person', False)) if self.watcher else False
        night = self.in_quiet_hours(now)

        # 근무시간(9~18시) 밖은 무조건 절전 (사용자 지정 2026-09-08).
        # 사람이 보여도, 불이 켜져도, 말을 걸어도 자는 상태를 유지한다 -
        # 터널·로그 같은 통신만 살아 있다. wake() 도 이 시간대엔 거부한다.
        if night:
            if not self._asleep:
                self._asleep = True
                self._changed_at = now
                logger.info('잠듭니다 (근무시간 밖 - 무조건 절전). '
                            '통신은 그대로 열어 둡니다.')
            return True

        # 어둡다고 볼 근거: 실제로 어둡거나, 조용한 시간대이거나.
        # 밝기를 못 재는데 낮이면(카메라 고장) 자지 않는다 - 못 보는 것을
        # 어둡다고 우기면 대낮에 로봇이 통째로 잠들어 버린다.
        if level is None:
            dark = night
        else:
            dark = level < self.dark_below or night

        if self._asleep:
            if person:
                self.wake('사람이 보입니다')
                return False
            # 조용한 시간대에는 불이 켜졌다고 깨지 않는다. 청소나 잠깐
            # 들른 사람의 불빛으로 밤새 깼다 잠들었다 하지 않게.
            if not night and level is not None and level > self.light_above:
                if self._light_since is None:
                    self._light_since = now
                elif now - self._light_since >= self.wake_for:
                    self.wake('불이 켜졌습니다')
                    return False
            else:
                self._light_since = None
            return True

        self._light_since = None
        if person or not dark:
            self._dark_since = None
            return False

        if self._dark_since is None:
            self._dark_since = now
            return False

        if now - self._dark_since >= self.quiet_for:
            self._asleep = True
            self._changed_at = now
            why = ('조용한 시간대' if level is None else
                   '밝기 %.3f%s' % (level, ', 조용한 시간대' if night else ''))
            logger.info('잠듭니다 (%s, %.0f분째 사람 없음). '
                        '통신은 그대로 열어 둡니다.',
                        why, (now - self._dark_since) / 60.0)
            return True

        return False
