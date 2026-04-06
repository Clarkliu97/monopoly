from __future__ import annotations

import unittest

from monopoly.agent.board_analysis import (
    analyze_board,
    analyze_transition,
    max_opponent_buildability,
    max_opponent_rent_pressure,
    relative_board_strength,
    strongest_opponent_name,
)
from monopoly.game import Game, PendingAuctionState


class BoardAnalysisTests(unittest.TestCase):
    def test_analyze_board_uses_actor_relative_order_and_marks_buildable_monopoly_spaces(self) -> None:
        game = Game(["A", "B", "C"])
        old_kent = game.board.get_space(1)
        baltic = game.board.get_space(3)
        old_kent.assign_owner(game.players[1])
        baltic.assign_owner(game.players[1])

        analysis = analyze_board(game.get_frontend_state(), "B")

        self.assertEqual(("B", "C", "A"), tuple(player.name for player in analysis.ordered_players))
        self.assertEqual(0, analysis.slot_by_name["B"])
        self.assertEqual(1.0, analysis.owner_group_progress_by_space_index[old_kent.index])
        self.assertEqual(1.0, analysis.owner_group_buildable_by_space_index[old_kent.index])
        self.assertGreater(analysis.estimated_space_pressure_by_index[old_kent.index], 0.0)

    def test_board_analysis_helper_functions_identify_strongest_opponent_and_pressure(self) -> None:
        game = Game(["A", "B", "C"])
        old_kent = game.board.get_space(1)
        baltic = game.board.get_space(3)
        whitechapel = game.board.get_space(16)
        northumberland = game.board.get_space(28)
        old_kent.assign_owner(game.players[1])
        baltic.assign_owner(game.players[1])
        whitechapel.assign_owner(game.players[2])
        whitechapel.building_count = 2
        northumberland.assign_owner(game.players[2])

        analysis = analyze_board(game.get_frontend_state(), "A")

        self.assertEqual("C", strongest_opponent_name(analysis, "A"))
        self.assertGreater(max_opponent_rent_pressure(analysis, "A"), 0.0)
        self.assertGreater(max_opponent_buildability(analysis, "A"), 0.0)
        self.assertLess(relative_board_strength(analysis, "A"), 0.0)

    def test_analyze_transition_reports_monopoly_denial_when_near_monopoly_is_broken(self) -> None:
        previous_game = Game(["A", "B"])
        previous_game.board.get_space(1).assign_owner(previous_game.players[0])

        next_game = Game(["A", "B"])

        diagnostics = analyze_transition(previous_game.get_frontend_state(), next_game.get_frontend_state())

        self.assertGreater(diagnostics.monopoly_denial_events, 0.0)

    def test_analyze_transition_reports_auction_bid_quality_for_resolved_win(self) -> None:
        game = Game(["A", "B"])
        property_space = game.board.get_space(1)
        game.pending_auction = PendingAuctionState(
            property_index=property_space.index,
            property_name=property_space.name,
            eligible_player_names=["A", "B"],
            active_player_names=["A", "B"],
            current_bid=120,
            current_winner_name="A",
            current_bidder_index=1,
        )
        previous_state = game.get_frontend_state()

        game.submit_auction_bid("B", None)
        next_state = game.get_frontend_state()

        diagnostics = analyze_transition(previous_state, next_state)

        self.assertIsNotNone(diagnostics.auction_bid_quality)
        self.assertGreater(diagnostics.auction_bid_quality, 1.0)


if __name__ == "__main__":
    unittest.main()