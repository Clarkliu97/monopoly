from __future__ import annotations

import unittest

from monopoly.board import create_standard_board
from monopoly.spaces import RailroadPropertySpace, StreetPropertySpace, UtilityPropertySpace


class BoardTests(unittest.TestCase):
    def test_standard_board_has_forty_spaces(self) -> None:
        board = create_standard_board()
        self.assertEqual(40, len(board.spaces))

    def test_board_contains_expected_property_types(self) -> None:
        board = create_standard_board()
        railroads = [space for space in board.spaces if isinstance(space, RailroadPropertySpace)]
        utilities = [space for space in board.spaces if isinstance(space, UtilityPropertySpace)]
        streets = [space for space in board.spaces if isinstance(space, StreetPropertySpace)]

        self.assertEqual(4, len(railroads))
        self.assertEqual(2, len(utilities))
        self.assertEqual(22, len(streets))

    def test_next_space_of_type_wraps_around_board(self) -> None:
        board = create_standard_board()

        next_railroad = board.next_space_of_type(36, "railroad")

        self.assertEqual("King's Cross Station", next_railroad.name)

    def test_commonwealth_decks_have_sixteen_cards_each(self) -> None:
        board = create_standard_board()

        self.assertEqual(16, len(board.chance_deck))
        self.assertEqual(16, len(board.community_chest_deck))

    def test_kept_card_is_removed_until_returned(self) -> None:
        board = create_standard_board()
        keep_card = next(card for card in board.chance_deck if card.keep_until_used)
        board.chance_deck.remove(keep_card)
        board.chance_deck.appendleft(keep_card)

        drawn = board.draw_card("chance")

        self.assertIs(drawn, keep_card)
        self.assertNotIn(keep_card, board.chance_deck)

        board.return_kept_card(drawn)

        self.assertIn(keep_card, board.chance_deck)

    def test_non_kept_card_cycles_to_back_of_deck_after_draw(self) -> None:
        board = create_standard_board()
        first_card = board.chance_deck[0]
        self.assertFalse(first_card.keep_until_used)

        drawn = board.draw_card("chance")

        self.assertIs(drawn, first_card)
        self.assertIs(board.chance_deck[-1], first_card)

    def test_properties_in_color_group_returns_complete_group(self) -> None:
        board = create_standard_board()

        brown_group = board.properties_in_color_group("Brown")

        self.assertEqual(["Old Kent Road", "Whitechapel Road"], [space.name for space in brown_group])


if __name__ == "__main__":
    unittest.main()
