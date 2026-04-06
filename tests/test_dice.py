from __future__ import annotations

import unittest

from monopoly.dice import Dice, DiceRoll


class DiceTests(unittest.TestCase):
    def test_dice_roll_properties_report_total_and_double_status(self) -> None:
        roll = DiceRoll(die_one=3, die_two=3)

        self.assertEqual(6, roll.total)
        self.assertTrue(roll.is_double)

    def test_scripted_rolls_are_used_before_random_rolls_without_advancing_rng(self) -> None:
        dice = Dice(seed=7, scripted_rolls=[(6, 6)])
        reference = Dice(seed=7)

        scripted = dice.roll()
        random_after_script = dice.roll()
        reference_random = reference.roll()

        self.assertEqual(DiceRoll(6, 6), scripted)
        self.assertEqual(reference_random, random_after_script)

    def test_dice_round_trip_preserves_remaining_scripted_rolls_and_random_state(self) -> None:
        dice = Dice(seed=11, scripted_rolls=[(1, 2), (3, 4)])

        self.assertEqual(DiceRoll(1, 2), dice.roll())
        restored = Dice.from_dict(dice.to_dict())

        self.assertEqual(dice.roll(), restored.roll())
        self.assertEqual(dice.roll(), restored.roll())


if __name__ == "__main__":
    unittest.main()