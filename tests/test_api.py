from __future__ import annotations

import unittest

from monopoly.api import (
    AIPlayerSetup,
    AuctionView,
    BoardSpaceView,
    FrontendStateView,
    GameSetup,
    GameView,
    InteractionResult,
    JailDecisionView,
    LegalActionOption,
    OnlineSeatView,
    OnlineSessionView,
    PendingActionView,
    PlayerView,
    PropertyActionView,
    TradeDecisionView,
    TurnPlanView,
)


class ApiViewTests(unittest.TestCase):
    def test_pending_action_round_trip_preserves_nested_prompt_views(self) -> None:
        pending = PendingActionView(
            action_type="auction",
            player_name="Alice",
            player_role="human",
            turn_phase="in_turn",
            prompt="Bid or pass.",
            available_actions=("bid", "pass"),
            property_name="Mayfair",
            property_index=39,
            price=400,
            auction=AuctionView(
                property_name="Mayfair",
                property_index=39,
                current_bid=150,
                current_winner_name="Bob",
                current_bidder_name="Alice",
                eligible_player_names=("Alice", "Bob"),
                active_player_names=("Alice", "Bob"),
                minimum_bid=155,
            ),
            jail=JailDecisionView(
                player_name="Alice",
                player_role="human",
                available_actions=("roll", "pay_fine"),
                fine_amount=50,
                has_get_out_of_jail_card=False,
                jail_turns=1,
            ),
            property_action=PropertyActionView(
                action_type="mortgage",
                player_name="Alice",
                player_role="human",
                property_name="Mayfair",
                property_index=39,
                cash_effect=200,
                description="Mortgage Mayfair",
            ),
            trade=TradeDecisionView(
                proposer_name="Alice",
                proposer_role="human",
                receiver_name="Bob",
                receiver_role="ai",
                proposer_cash=100,
                receiver_cash=50,
                proposer_property_names=("Mayfair",),
                receiver_property_names=("Park Lane",),
                proposer_jail_cards=0,
                receiver_jail_cards=1,
                note="Swap dark blues",
            ),
        )

        restored = PendingActionView.from_dict(pending.to_dict())

        self.assertEqual(pending, restored)

    def test_frontend_state_round_trip_preserves_game_board_and_turn_plan(self) -> None:
        player = PlayerView(
            name="Alice",
            role="human",
            cash=1500,
            position=0,
            in_jail=False,
            jail_turns=0,
            get_out_of_jail_cards=0,
            is_bankrupt=False,
            properties=("Mayfair",),
        )
        pending = PendingActionView(
            action_type="purchase",
            player_name="Alice",
            player_role="human",
            turn_phase="in_turn",
            prompt="Buy Mayfair?",
            available_actions=("buy", "decline"),
            property_name="Mayfair",
            property_index=39,
            price=400,
        )
        game_view = GameView(
            turn_counter=4,
            current_player_name="Alice",
            current_player_role="human",
            current_turn_phase="in_turn",
            starting_cash=1500,
            houses_remaining=32,
            hotels_remaining=12,
            players=(player,),
            pending_action=pending,
            blocked_trade_offer_signatures=("abc",),
        )
        turn_plan = TurnPlanView(
            player_name="Alice",
            player_role="human",
            turn_phase="in_turn",
            is_current_player=True,
            has_pending_action=True,
            pending_action=pending,
            legal_actions=(
                LegalActionOption(
                    action_type="buy_property",
                    actor_name="Alice",
                    actor_role="human",
                    handler_name="resolve_purchase",
                    description="Buy Mayfair",
                    property_name="Mayfair",
                ),
            ),
            reason=None,
        )
        state = FrontendStateView(
            game_view=game_view,
            active_turn_plan=turn_plan,
            board_spaces=(
                BoardSpaceView(
                    index=39,
                    name="Mayfair",
                    space_type="street",
                    occupant_names=("Alice",),
                    owner_name="Alice",
                    color_group="Dark Blue",
                    mortgaged=False,
                    building_count=0,
                    price=400,
                    house_cost=200,
                    notes="Prime property",
                ),
            ),
        )

        restored = FrontendStateView.from_dict(state.to_dict())

        self.assertEqual(state, restored)

    def test_game_setup_round_trip_preserves_per_player_ai_configuration(self) -> None:
        setup = GameSetup(
            player_names=("Alice", "Bot"),
            starting_cash=1800,
            player_roles=("human", "ai"),
            ai_checkpoint_path="scripted:auction_value_shark",
            ai_player_setups=(
                AIPlayerSetup(
                    player_name="Bot",
                    checkpoint_path="scripted:auction_value_shark",
                    action_cooldown_seconds=0.5,
                ),
            ),
        )

        payload = setup.to_dict()
        restored = GameSetup.from_dict(
            {
                **payload,
                "ai_player_setups": [
                    {
                        "player_name": "Bot",
                        "checkpoint_path": "scripted:auction_value_shark",
                        "action_cooldown_seconds": 0.5,
                    }
                ],
            }
        )

        self.assertEqual(("human", "ai"), restored.resolved_player_roles())
        self.assertEqual("scripted:auction_value_shark", restored.ai_checkpoint_path)
        self.assertEqual("Bot", restored.ai_player_setups[0].player_name)

    def test_online_session_round_trip_preserves_optional_pause_and_ai_fields(self) -> None:
        session = OnlineSessionView(
            session_code="ABC123",
            state="paused",
            host_player_name="Alice",
            seat_count=2,
            starting_cash=1500,
            seats=(
                OnlineSeatView(seat_index=0, status="host", player_name="Alice", player_role="human", is_host=True, is_claimable=False),
                OnlineSeatView(
                    seat_index=1,
                    status="ai",
                    player_name="Bot",
                    player_role="ai",
                    is_host=False,
                    is_claimable=False,
                    checkpoint_path="scripted:expansionist_builder",
                    action_cooldown_seconds=0.75,
                ),
            ),
            paused_reason="player_disconnected",
            paused_seat_index=1,
        )

        restored = OnlineSessionView.from_dict(session.to_dict())

        self.assertEqual(session, restored)

    def test_interaction_result_round_trip_preserves_embedded_state(self) -> None:
        interaction = InteractionResult(
            messages=("Alice buys Mayfair.",),
            game_view=GameView(
                turn_counter=1,
                current_player_name="Alice",
                current_player_role="human",
                current_turn_phase="post_roll",
                starting_cash=1500,
                houses_remaining=32,
                hotels_remaining=12,
                players=(
                    PlayerView(
                        name="Alice",
                        role="human",
                        cash=1100,
                        position=39,
                        in_jail=False,
                        jail_turns=0,
                        get_out_of_jail_cards=0,
                        is_bankrupt=False,
                        properties=("Mayfair",),
                    ),
                ),
            ),
        )

        restored = InteractionResult.from_dict(interaction.to_dict())

        self.assertEqual(interaction, restored)


if __name__ == "__main__":
    unittest.main()