import rospy
from collections import deque

class CurriculumManager:
    def __init__(self, max_level=4, window_size=20,
                 advance_threshold=0.5, regress_threshold=0.3,
                 consecutive_fails_for_regress=None,
                 min_episodes_at_level=None):
        self.max_level = max_level
        self.current_level = 1
        self.window = deque(maxlen=window_size)
        self.advance_threshold = advance_threshold
        self.regress_threshold = regress_threshold

        if isinstance(consecutive_fails_for_regress, dict):
            self.consecutive_fails_for_regress = consecutive_fails_for_regress
        elif consecutive_fails_for_regress is not None:
            self.consecutive_fails_for_regress = consecutive_fails_for_regress
        else:
            self.consecutive_fails_for_regress = {1: 10, 2: 15, 3: 20, 4: 25}
        self.min_episodes_at_level = (
            min_episodes_at_level
            if min_episodes_at_level is not None
            else {1: 500, 2: 1000, 3: 1500, 4: 0}
        )

        self._consecutive_fails = 0
        self._level_episode_count = 0

    @property
    def spawn_index(self):
        return self.current_level

    def _get_fail_threshold(self):
        if isinstance(self.consecutive_fails_for_regress, dict):
            return self.consecutive_fails_for_regress.get(self.current_level, 10)
        return self.consecutive_fails_for_regress

    def update(self, success: bool) -> None:
        self.window.append(success)
        self._level_episode_count += 1

        if success:
            self._consecutive_fails = 0
        else:
            self._consecutive_fails += 1

        fail_th = self._get_fail_threshold()
        if self._consecutive_fails >= fail_th:
            if self.current_level > 1:
                self.current_level -= 1
                self._consecutive_fails = 0
                self._level_episode_count = 0
                self.window.clear()
                rospy.loginfo(f"[Curriculum] Regressed to level {self.current_level} "
                              f"({fail_th} consecutive fails)")
            return

        if len(self.window) < self.window.maxlen:
            return

        min_ep = self.min_episodes_at_level.get(self.current_level, 0)
        if self._level_episode_count < min_ep:
            return

        rate = sum(self.window) / len(self.window)

        if rate >= self.advance_threshold and self.current_level < self.max_level:
            self.current_level += 1
            self._consecutive_fails = 0
            self._level_episode_count = 0
            self.window.clear()
            rospy.loginfo(f"[Curriculum] Advanced to level {self.current_level} "
                          f"(success rate: {rate:.2f})")

        elif rate <= self.regress_threshold and self.current_level > 1:
            self.current_level -= 1
            self._consecutive_fails = 0
            self._level_episode_count = 0
            self.window.clear()
            rospy.loginfo(f"[Curriculum] Regressed to level {self.current_level} "
                          f"(success rate: {rate:.2f})")

    @property
    def success_rate(self):
        return sum(self.window) / len(self.window) if self.window else 0.0
