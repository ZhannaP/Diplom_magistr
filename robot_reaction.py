from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, Optional, Protocol


class Action(Enum):
    CONTINUE = "continue"        
    SUPPORT = "support"          
    PAUSE = "pause" 
    SLOW_DOWN = "slow_down"
    EXPLAIN = "explain"
    CHANGE_TASK = "change_task" 
    RETRY = "retry"
    NONE = "none"


@dataclass(frozen=True)
class Reaction:
    scenario: str          
    message: str           
    action: Action         
    speed_factor: float    


SCENARIOS: Dict[str, Reaction] = {
    "happy": Reaction(
        "Радість", "Чудово виходить! Продовжуємо вправу.", Action.CONTINUE, 1.0),
    "sad": Reaction(
        "Смуток", "Ви молодець. Якщо потрібна допомога — я поруч.", Action.SUPPORT, 1.0),
    "angry": Reaction(
        "Злість", "Давайте зробимо паузу або зменшимо темп. Як ви почуваєтесь?",
        Action.PAUSE, 0.5),
    "surprise": Reaction(
        "Подив", "Поясню завдання ще раз детальніше.", Action.EXPLAIN, 1.0),
    "fear": Reaction(
        "Страх", "Все гаразд, не поспішайте. Продовжимо у комфортному темпі.",
        Action.SLOW_DOWN, 0.7),
    "disgust": Reaction(
        "Відраза", "Як ви почуваєтесь? Можемо змінити завдання.",
        Action.CHANGE_TASK, 0.8),
    "neutral": Reaction(
        "Нейтральний", "", Action.CONTINUE, 1.0),
}

UNCERTAIN = Reaction(
    "Невизначений результат",
    "Не вдалося впевнено визначити стан. Продовжуємо за планом.",
    Action.RETRY, 1.0,
)


class RobotInterface(Protocol):
    def execute(self, reaction: Reaction) -> None: ...


class ConsoleRobot:

    def execute(self, reaction: Reaction) -> None:
        print(f"[ROBOT] {reaction.action.value} | speed x{reaction.speed_factor} | "
              f"\"{reaction.message}\"")


class RobotReactionManager:
    def __init__(
        self,
        robot: Optional[RobotInterface] = None,
        min_confidence: float = 0.5,
        window: int = 8,
        min_votes: int = 5,
        cooldown_s: float = 8.0,
    ):
        self.robot = robot or ConsoleRobot()
        self.min_confidence = min_confidence
        self.window = window
        self.min_votes = min_votes
        self.cooldown_s = cooldown_s

        self._history: Deque[Optional[str]] = deque(maxlen=window)
        self._last_time = 0.0
        self._last_key: Optional[str] = None

    def reset(self) -> None:
        self._history.clear()
        self._last_time = 0.0
        self._last_key = None

    # ------------------------------------------------------------------ #
    def react(
        self, label: Optional[str], confidence: Optional[float], stream: bool = False
    ) -> Optional[Reaction]:

        key = self._reliable_key(label, confidence)

        if not stream:
            reaction = SCENARIOS[key] if key else UNCERTAIN
            self._execute(reaction)
            return reaction

        self._history.append(key)
        stable = self._stable_key()
        if stable is None:
            return None

        now = time.monotonic()
        same = stable == self._last_key
        if now - self._last_time < self.cooldown_s and same:
            return None 
        if now - self._last_time < self.cooldown_s / 2:
            return None

        self._last_key, self._last_time = stable, now
        reaction = SCENARIOS[stable]
        self._execute(reaction)
        return reaction

    def _reliable_key(self, label: Optional[str], conf: Optional[float]) -> Optional[str]:
        if label is None or conf is None or conf < self.min_confidence:
            return None
        key = label.lower()
        return key if key in SCENARIOS else None

    def _stable_key(self) -> Optional[str]:
        valid = [k for k in self._history if k is not None]
        if not valid:
            return None
        key, votes = Counter(valid).most_common(1)[0]
        return key if votes >= self.min_votes else None

    def _execute(self, reaction: Reaction) -> None:
        if reaction.action is Action.NONE:
            return
        self.robot.execute(reaction)
