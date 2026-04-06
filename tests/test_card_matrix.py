from __future__ import annotations

from dataclasses import dataclass
import unittest

from monopoly.dice import Dice
from monopoly.game import Game


@dataclass(frozen=True, slots=True)
class CardCase:
    index: int
    expected_name: str
    setup_name: str
    player_count: int = 2
    dice_rolls: tuple[tuple[int, int], ...] = ()


class CardMatrixTests(unittest.TestCase):
    def test_community_chest_cards_apply_expected_effects(self) -> None:
        cases = [
            CardCase(0, "Advance to Go", "_setup_advance_to_go"),
            CardCase(1, "Bank Error", "_setup_collect_200"),
            CardCase(2, "Doctor's Fee", "_setup_doctors_fee"),
            CardCase(3, "Sale of Stock", "_setup_collect_50"),
            CardCase(4, "Get Out of Jail Free", "_setup_get_out_of_jail"),
            CardCase(5, "Go to Jail", "_setup_go_to_jail"),
            CardCase(6, "Holiday Fund Matures", "_setup_collect_100"),
            CardCase(7, "Income Tax Refund", "_setup_collect_20"),
            CardCase(8, "Birthday", "_setup_birthday", player_count=3),
            CardCase(9, "Life Insurance Matures", "_setup_collect_100"),
            CardCase(10, "Hospital Fees", "_setup_hospital_fees"),
            CardCase(11, "School Fees", "_setup_school_fees"),
            CardCase(12, "Consultancy Fee", "_setup_collect_25"),
            CardCase(13, "Street Repairs", "_setup_street_repairs"),
            CardCase(14, "Beauty Contest", "_setup_collect_10"),
            CardCase(15, "Inheritance", "_setup_collect_100"),
        ]

        self._assert_card_cases(deck_name="community_chest", cases=cases)

    def test_chance_cards_apply_expected_effects(self) -> None:
        cases = [
            CardCase(0, "Advance to Go", "_setup_advance_to_go"),
            CardCase(1, "Advance to Trafalgar Square", "_setup_trafalgar_square"),
            CardCase(2, "Advance to Mayfair", "_setup_mayfair"),
            CardCase(3, "Advance to Pall Mall", "_setup_pall_mall"),
            CardCase(4, "Advance to Nearest Station", "_setup_nearest_station"),
            CardCase(5, "Advance to Nearest Station", "_setup_nearest_station"),
            CardCase(6, "Advance to Nearest Utility", "_setup_nearest_utility", dice_rolls=((3, 4),)),
            CardCase(7, "Bank Pays Dividend", "_setup_collect_50"),
            CardCase(8, "Get Out of Jail Free", "_setup_get_out_of_jail"),
            CardCase(9, "Go Back Three Spaces", "_setup_go_back_three_spaces"),
            CardCase(10, "Go to Jail", "_setup_go_to_jail"),
            CardCase(11, "General Repairs", "_setup_general_repairs"),
            CardCase(12, "Speeding Fine", "_setup_speeding_fine"),
            CardCase(13, "Take a Trip to King's Cross Station", "_setup_kings_cross_station"),
            CardCase(14, "Chairman of the Board", "_setup_chairman_of_the_board", player_count=3),
            CardCase(15, "Building Loan Matures", "_setup_collect_150"),
        ]

        self._assert_card_cases(deck_name="chance", cases=cases)

    def _assert_card_cases(self, deck_name: str, cases: list[CardCase]) -> None:
        for case in cases:
            with self.subTest(deck=deck_name, card_index=case.index, card_name=case.expected_name):
                game = self._create_game(case.player_count, case.dice_rolls)
                player = game.players[0]
                expected = getattr(self, case.setup_name)(game, player)
                deck = game.board.chance_deck if deck_name == "chance" else game.board.community_chest_deck
                card = list(deck)[case.index]

                self.assertEqual(case.expected_name, card.name)
                messages = card.apply(game, player)

                self._assert_expected_state(game, player, messages, expected)

    @staticmethod
    def _create_game(player_count: int, dice_rolls: tuple[tuple[int, int], ...]) -> Game:
        player_names = [chr(ord("A") + index) for index in range(player_count)]
        if dice_rolls:
            return Game(player_names, dice=Dice(scripted_rolls=list(dice_rolls)))
        return Game(player_names)

    def _assert_expected_state(self, game: Game, player, messages: list[str], expected: dict[str, object]) -> None:
        if "position" in expected:
            self.assertEqual(expected["position"], player.position)
        if "cash" in expected:
            self.assertEqual(expected["cash"], player.cash)
        if "in_jail" in expected:
            self.assertEqual(expected["in_jail"], player.in_jail)
        if "jail_turns" in expected:
            self.assertEqual(expected["jail_turns"], player.jail_turns)
        if "jail_cards" in expected:
            self.assertEqual(expected["jail_cards"], player.get_out_of_jail_cards)
        if "other_cash" in expected:
            for player_index, cash in expected["other_cash"].items():
                self.assertEqual(cash, game.players[player_index].cash)
        if "owned_properties" in expected:
            self.assertEqual(expected["owned_properties"], tuple(space.name for space in player.properties))
        for fragment in expected.get("message_contains", ()):  # type: ignore[union-attr]
            self.assertTrue(any(fragment in message for message in messages), msg=f"Expected {fragment!r} in {messages!r}")
        for fragment, count in expected.get("message_count_contains", {}).items():  # type: ignore[union-attr]
            self.assertEqual(count, sum(1 for message in messages if fragment in message))

    def _setup_advance_to_go(self, game: Game, player) -> dict[str, object]:
        player.position = 17
        return {"position": 0, "cash": 1700, "message_contains": ("moves to Go and collects $200.",)}

    def _setup_collect_200(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1700, "message_contains": ("collects $200.",)}

    def _setup_doctors_fee(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1450, "message_contains": ("Doctor's Fee", "pays $50")}

    def _setup_collect_50(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1550, "message_contains": ("collects $50.",)}

    def _setup_get_out_of_jail(self, game: Game, player) -> dict[str, object]:
        return {"jail_cards": 1, "message_contains": ("keeps a Get Out of Jail Free card.",)}

    def _setup_go_to_jail(self, game: Game, player) -> dict[str, object]:
        player.position = 7
        return {"position": 10, "in_jail": True, "jail_turns": 0, "message_contains": ("goes directly to jail.",)}

    def _setup_collect_100(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1600, "message_contains": ("collects $100.",)}

    def _setup_collect_20(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1520, "message_contains": ("collects $20.",)}

    def _setup_birthday(self, game: Game, player) -> dict[str, object]:
        return {
            "cash": 1520,
            "other_cash": {1: 1490, 2: 1490},
            "message_count_contains": {"pays $10": 2},
        }

    def _setup_hospital_fees(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1400, "message_contains": ("Hospital Fees", "pays $100")}

    def _setup_school_fees(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1450, "message_contains": ("School Fees", "pays $50")}

    def _setup_collect_25(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1525, "message_contains": ("collects $25.",)}

    def _setup_street_repairs(self, game: Game, player) -> dict[str, object]:
        game.board.get_space(1).assign_owner(player)
        game.board.get_space(3).assign_owner(player)
        game.board.get_space(1).building_count = 2
        game.board.get_space(3).building_count = 5
        return {"cash": 1305, "message_contains": ("Street Repairs", "pays $195")}

    def _setup_collect_10(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1510, "message_contains": ("collects $10.",)}

    def _setup_trafalgar_square(self, game: Game, player) -> dict[str, object]:
        player.position = 30
        game.board.get_space(24).assign_owner(player)
        return {
            "position": 24,
            "cash": 1700,
            "owned_properties": ("Trafalgar Square",),
            "message_contains": ("advances to Trafalgar Square", "already owns Trafalgar Square"),
        }

    def _setup_mayfair(self, game: Game, player) -> dict[str, object]:
        player.position = 20
        game.board.get_space(39).assign_owner(player)
        return {
            "position": 39,
            "cash": 1500,
            "owned_properties": ("Mayfair",),
            "message_contains": ("advances to Mayfair", "already owns Mayfair"),
        }

    def _setup_pall_mall(self, game: Game, player) -> dict[str, object]:
        player.position = 20
        game.board.get_space(11).assign_owner(player)
        return {
            "position": 11,
            "cash": 1700,
            "owned_properties": ("Pall Mall",),
            "message_contains": ("advances to Pall Mall", "already owns Pall Mall"),
        }

    def _setup_nearest_station(self, game: Game, player) -> dict[str, object]:
        owner = game.players[1]
        game.board.get_space(5).assign_owner(owner)
        player.position = 36
        return {
            "position": 5,
            "cash": 1650,
            "other_cash": {1: 1550},
            "message_contains": ("nearest station", "pays $50"),
        }

    def _setup_nearest_utility(self, game: Game, player) -> dict[str, object]:
        owner = game.players[1]
        game.board.get_space(12).assign_owner(owner)
        player.position = 7
        return {
            "position": 12,
            "cash": 1430,
            "other_cash": {1: 1570},
            "message_contains": ("nearest utility", "utility card", "pays $70"),
        }

    def _setup_go_back_three_spaces(self, game: Game, player) -> dict[str, object]:
        player.position = 7
        return {
            "position": 4,
            "cash": 1300,
            "message_contains": ("goes back three spaces", "Income Tax"),
        }

    def _setup_general_repairs(self, game: Game, player) -> dict[str, object]:
        game.board.get_space(1).assign_owner(player)
        game.board.get_space(3).assign_owner(player)
        game.board.get_space(1).building_count = 2
        game.board.get_space(3).building_count = 5
        return {"cash": 1350, "message_contains": ("General Repairs", "pays $150")}

    def _setup_speeding_fine(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1485, "message_contains": ("Speeding Fine", "pays $15")}

    def _setup_kings_cross_station(self, game: Game, player) -> dict[str, object]:
        player.position = 36
        game.board.get_space(5).assign_owner(player)
        return {
            "position": 5,
            "cash": 1700,
            "owned_properties": ("King's Cross Station",),
            "message_contains": ("King's Cross Station", "already owns King's Cross Station"),
        }

    def _setup_chairman_of_the_board(self, game: Game, player) -> dict[str, object]:
        return {
            "cash": 1400,
            "other_cash": {1: 1550, 2: 1550},
            "message_count_contains": {"pays $50": 2},
        }

    def _setup_collect_150(self, game: Game, player) -> dict[str, object]:
        return {"cash": 1650, "message_contains": ("collects $150.",)}


if __name__ == "__main__":
    unittest.main()