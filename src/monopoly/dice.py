from __future__ import annotations

"""Dice utilities for deterministic and serialized game rolls.

The project uses this module both for normal gameplay randomness and for fully
reproducible execution in saves, tests, and RL rollouts. Scripted rolls are
consumed first so scenarios can pin exact outcomes without disturbing the random
generator state that follows.
"""

import base64
from collections import deque
from dataclasses import dataclass
import pickle
import random
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class DiceRoll:
    """One two-die Monopoly roll with convenience accessors."""

    die_one: int
    die_two: int

    @property
    def total(self) -> int:
        """Return the sum of both dice."""
        return self.die_one + self.die_two

    @property
    def is_double(self) -> bool:
        """Whether both dice show the same face."""
        return self.die_one == self.die_two


class Dice:
    """Roll Monopoly dice with optional seeding and scripted deterministic outputs."""

    def __init__(self, seed: int | None = None, scripted_rolls: list[tuple[int, int]] | None = None) -> None:
        """Create dice backed by a PRNG plus an optional queue of scripted rolls."""
        self._random = random.Random(seed)
        self._scripted_rolls = deque(scripted_rolls or [])

    def roll(self) -> DiceRoll:
        """Return the next scripted roll if present, otherwise roll both random dice."""
        if self._scripted_rolls:
            die_one, die_two = self._scripted_rolls.popleft()
            return DiceRoll(die_one=die_one, die_two=die_two)
        return DiceRoll(die_one=self._random.randint(1, 6), die_two=self._random.randint(1, 6))

    def to_dict(self) -> dict[str, Any]:
        """Serialize remaining scripted rolls and PRNG state for save/load round-trips."""
        random_state = pickle.dumps(self._random.getstate())
        return {
            "scripted_rolls": [list(roll) for roll in self._scripted_rolls],
            "random_state": base64.b64encode(random_state).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Dice:
        """Restore dice from `to_dict()` output, including generator progress."""
        dice = cls(scripted_rolls=[tuple(int(value) for value in roll) for roll in data.get("scripted_rolls", [])])
        random_state = data.get("random_state")
        if random_state is not None:
            decoded_state = base64.b64decode(str(random_state).encode("ascii"))
            dice._random.setstate(pickle.loads(decoded_state))
        return dice
