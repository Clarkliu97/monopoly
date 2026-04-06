from __future__ import annotations

import unittest

from monopoly.dice import Dice
from monopoly.game import Game
from monopoly.trading import TradeOffer
from monopoly.api import FrontendStateView, GameView, InteractionResult, TurnPlanView


class InteractiveApiTests(unittest.TestCase):
    def test_game_view_can_round_trip_through_dict(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1)]))
        game.players[0].position = 39
        game.start_turn_interactive()

        payload = game.get_game_view().to_dict()
        restored = GameView.from_dict(payload)

        self.assertEqual(game.get_game_view(), restored)

    def test_turn_plan_can_round_trip_through_dict(self) -> None:
        game = Game(["A", "B"])

        payload = game.get_turn_plan("A").to_dict()
        restored = TurnPlanView.from_dict(payload)

        self.assertEqual(game.get_turn_plan("A"), restored)

    def test_frontend_state_can_round_trip_through_dict(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1)]))
        game.players[0].position = 39
        game.start_turn_interactive()

        payload = game.get_frontend_state().to_dict()
        restored = FrontendStateView.from_dict(payload)

        self.assertEqual(game.get_frontend_state(), restored)

    def test_execute_legal_action_can_run_purchase_flow_and_end_turn(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1), (1, 2)]))
        game.players[0].position = 39

        start_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "start_turn")
        start_result = game.execute_legal_action(start_action)
        self.assertEqual("property_purchase", start_result.pending_action.action_type)

        buy_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "buy_property")
        buy_result = game.execute_legal_action(buy_action)
        self.assertEqual("post_roll", buy_result.game_view.current_turn_phase)
        self.assertIs(game.board.get_space(1).owner, game.players[0])

        end_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "end_turn")
        end_result = game.execute_legal_action(end_action)
        self.assertEqual("B", end_result.game_view.current_player_name)

    def test_execute_legal_action_can_drive_auction_flow(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 1), (1, 2)]))
        game.players[0].position = 39

        start_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "start_turn")
        game.execute_legal_action(start_action)
        decline_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "decline_property")
        game.execute_legal_action(decline_action)

        pass_action_a = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "pass_auction")
        game.execute_legal_action(pass_action_a)
        bid_action_b = next(action for action in game.get_turn_plan("B").legal_actions if action.action_type == "place_auction_bid")
        game.execute_legal_action(bid_action_b, bid_amount=80)
        pass_action_c = next(action for action in game.get_turn_plan("C").legal_actions if action.action_type == "pass_auction")
        result = game.execute_legal_action(pass_action_c)

        self.assertIs(game.board.get_space(1).owner, game.players[1])
        self.assertEqual("post_roll", result.game_view.current_turn_phase)

    def test_execute_legal_action_can_handle_jail_decisions(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 2)]))
        player = game.players[0]
        player.in_jail = True
        player.position = 10
        player.get_out_of_jail_cards = 1

        start_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "start_turn")
        game.execute_legal_action(start_action)
        jail_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "jail_use_card")
        result = game.execute_legal_action(jail_action)

        self.assertFalse(player.in_jail)
        self.assertEqual("property_purchase", result.pending_action.action_type)

    def test_execute_legal_action_can_request_and_confirm_property_action(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        for index in (1, 3):
            game.board.get_space(index).assign_owner(player)
        player.cash = 1000

        request_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "request_build")
        request_result = game.execute_legal_action(request_action)
        self.assertEqual("property_action", request_result.pending_action.action_type)

        confirm_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "confirm_property_action")
        game.execute_legal_action(confirm_action)
        self.assertEqual(1, game.board.get_space(1).building_count)

    def test_execute_legal_action_can_propose_and_accept_trade(self) -> None:
        game = Game(["A", "B"])
        player_a = game.players[0]
        player_b = game.players[1]
        oriental = game.board.get_space(6)
        oriental.assign_owner(player_b)
        player_b.pay(oriental.price)

        propose_action = next(
            action
            for action in game.get_turn_plan("A").legal_actions
            if action.action_type == "propose_trade" and action.target_player_name == "B"
        )
        offer = TradeOffer(
            proposer=player_a,
            receiver=player_b,
            proposer_cash=200,
            receiver_properties=[oriental],
        )
        propose_result = game.execute_legal_action(propose_action, trade_offer=offer)
        self.assertEqual("trade_decision", propose_result.pending_action.action_type)

        accept_action = next(action for action in game.get_turn_plan("B").legal_actions if action.action_type == "accept_trade")
        game.execute_legal_action(accept_action)
        self.assertIs(oriental.owner, player_a)

    def test_execute_legal_action_rejects_stale_action(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1)]))
        game.players[0].position = 39
        start_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "start_turn")
        stale_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "propose_trade")

        game.execute_legal_action(start_action)

        with self.assertRaises(ValueError):
            game.execute_legal_action(stale_action, trade_offer=TradeOffer(proposer=game.players[0], receiver=game.players[1]))

    def test_execute_serialized_action_can_run_purchase_flow(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1), (1, 2)]))
        game.players[0].position = 39

        start_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "start_turn")
        start_result = InteractionResult.from_dict(game.execute_serialized_action(start_action.to_dict()))
        self.assertEqual("property_purchase", start_result.pending_action.action_type)

        buy_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "buy_property")
        buy_result = InteractionResult.from_dict(game.execute_serialized_action(buy_action.to_dict()))
        self.assertEqual("post_roll", buy_result.game_view.current_turn_phase)
        self.assertIs(game.board.get_space(1).owner, game.players[0])

    def test_serialized_helpers_support_trade_payloads(self) -> None:
        game = Game(["A", "B"])
        player_a = game.players[0]
        player_b = game.players[1]
        oriental = game.board.get_space(6)
        oriental.assign_owner(player_b)
        player_b.pay(oriental.price)

        propose_action = next(
            action
            for action in game.get_turn_plan("A").legal_actions
            if action.action_type == "propose_trade" and action.target_player_name == "B"
        )
        trade_offer = TradeOffer(
            proposer=player_a,
            receiver=player_b,
            proposer_cash=200,
            receiver_properties=[oriental],
            note="serialized trade",
        )

        payload = game.serialize_trade_offer(trade_offer)
        restored_offer = game.deserialize_trade_offer(payload)
        self.assertEqual("A", restored_offer.proposer.name)
        self.assertEqual("B", restored_offer.receiver.name)
        self.assertEqual(["The Angel Islington"], [space.name for space in restored_offer.receiver_properties])

        result = InteractionResult.from_dict(
            game.execute_serialized_action(propose_action.to_dict(), trade_offer_payload=payload)
        )
        self.assertEqual("trade_decision", result.pending_action.action_type)

    def test_game_view_starts_in_pre_roll_phase(self) -> None:
        game = Game(["A", "B"], player_roles=["human", "ai"])

        view = game.get_game_view()

        self.assertEqual("pre_roll", view.current_turn_phase)

    def test_game_view_exposes_player_roles(self) -> None:
        game = Game(["Human", "Bot"], player_roles={"Human": "human", "Bot": "ai"})

        view = game.get_game_view()

        self.assertEqual("human", view.players[0].role)
        self.assertEqual("ai", view.players[1].role)
        self.assertEqual("human", view.current_player_role)

    def test_start_turn_interactive_creates_purchase_decision(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1)]), player_roles=["human", "ai"])
        game.players[0].position = 39

        result = game.start_turn_interactive()

        self.assertIsNotNone(result.pending_action)
        self.assertEqual("property_purchase", result.pending_action.action_type)
        self.assertEqual("human", result.pending_action.player_role)
        self.assertEqual("Old Kent Road", result.pending_action.property_name)
        self.assertIsNone(game.board.get_space(1).owner)

    def test_pending_action_role_helpers_route_to_human_and_ai(self) -> None:
        human_game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1)]), player_roles=["human", "ai"])
        human_game.players[0].position = 39
        human_game.start_turn_interactive()

        self.assertTrue(human_game.is_pending_action_for_human())
        self.assertFalse(human_game.is_pending_action_for_ai())
        self.assertEqual("human", human_game.get_pending_action_role())

        ai_game = Game(["Human", "Bot"], dice=Dice(scripted_rolls=[(1, 1)]), player_roles={"Human": "human", "Bot": "ai"})
        ai_game.current_player_index = 1
        ai_game.players[1].position = 39
        ai_game.start_turn_interactive()

        self.assertTrue(ai_game.is_pending_action_for_ai())
        self.assertFalse(ai_game.is_pending_action_for_human())
        self.assertEqual("ai", ai_game.get_pending_action_role())

    def test_resolve_property_purchase_can_buy_and_finish_turn(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1), (1, 2)]))
        game.players[0].position = 39

        game.start_turn_interactive()
        result = game.resolve_property_decision(True)

        self.assertIs(game.board.get_space(1).owner, game.players[0])
        self.assertIsNone(result.pending_action)

    def test_unmortgage_action_is_hidden_when_player_cannot_afford_interest(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        mediterranean = game.board.get_space(1)
        mediterranean.assign_owner(player)
        mediterranean.mortgaged = True
        player.cash = mediterranean.mortgage_value
        game.current_turn_phase = "post_roll"

        turn_plan = game.get_turn_plan(player.name)

        self.assertFalse(any(action.action_type == "request_unmortgage" for action in turn_plan.legal_actions))

    def test_request_unmortgage_rejects_player_without_interest_cash(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        mediterranean = game.board.get_space(1)
        mediterranean.assign_owner(player)
        mediterranean.mortgaged = True
        player.cash = mediterranean.mortgage_value

        with self.assertRaises(ValueError):
            game.request_property_action(player, mediterranean.name, "unmortgage")

    def test_declining_property_opens_interactive_auction(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 1)]), player_roles=["human", "ai", "human"])
        game.players[0].position = 39

        game.start_turn_interactive()
        result = game.resolve_property_decision(False)

        self.assertIsNotNone(result.pending_action)
        self.assertEqual("auction", result.pending_action.action_type)
        self.assertEqual("A", result.pending_action.auction.current_bidder_name)
        self.assertEqual("human", result.pending_action.player_role)

    def test_interactive_auction_can_be_driven_by_ui_or_ai_calls(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 1), (1, 2)]))
        game.players[0].position = 39

        game.start_turn_interactive()
        game.resolve_property_decision(False)
        game.submit_auction_bid("A", None)
        game.submit_auction_bid("B", 80)
        result = game.submit_auction_bid("C", None)

        self.assertIs(game.board.get_space(1).owner, game.players[1])
        self.assertIsNone(result.pending_action)
        self.assertEqual("A", result.game_view.current_player_name)
        self.assertEqual("post_roll", result.game_view.current_turn_phase)

    def test_game_view_exposes_bank_and_pending_state(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1)]))
        game.players[0].position = 39

        game.start_turn_interactive()
        view = game.get_game_view()

        self.assertEqual(32, view.houses_remaining)
        self.assertEqual(12, view.hotels_remaining)
        self.assertIsNotNone(view.pending_action)

    def test_start_turn_interactive_can_request_jail_decision(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        player.in_jail = True
        player.position = 10
        player.get_out_of_jail_cards = 1

        result = game.start_turn_interactive()

        self.assertEqual("jail_decision", result.pending_action.action_type)
        self.assertIn("use_card", result.pending_action.available_actions)
        self.assertIn("roll", result.pending_action.available_actions)

    def test_resolve_jail_decision_with_card_continues_to_next_pending_action(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 2)]))
        player = game.players[0]
        player.in_jail = True
        player.position = 10
        player.get_out_of_jail_cards = 1

        game.start_turn_interactive()
        result = game.resolve_jail_decision("use_card")

        self.assertFalse(player.in_jail)
        self.assertEqual("property_purchase", result.pending_action.action_type)
        self.assertEqual("Whitehall", result.pending_action.property_name)

    def test_request_property_action_build_uses_pending_action_contract(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        for index in (1, 3):
            game.board.get_space(index).assign_owner(player)
        player.cash = 1000

        request = game.request_property_action("A", "build", "Old Kent Road")
        result = game.resolve_property_action(True)

        self.assertEqual("property_action", request.pending_action.action_type)
        self.assertEqual(-50, request.pending_action.property_action.cash_effect)
        self.assertEqual(1, game.board.get_space(1).building_count)
        self.assertIsNone(result.pending_action)

    def test_request_property_action_mortgage_can_be_cancelled(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        oriental = game.board.get_space(6)
        oriental.assign_owner(player)

        request = game.request_property_action("A", "mortgage", "The Angel Islington")
        result = game.resolve_property_action(False)

        self.assertEqual("property_action", request.pending_action.action_type)
        self.assertFalse(oriental.mortgaged)
        self.assertIsNone(result.pending_action)

    def test_cancelling_property_action_blocks_same_request_until_turn_ends(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        game.players[1].is_bankrupt = True
        oriental = game.board.get_space(6)
        oriental.assign_owner(player)
        game.current_turn_phase = "post_roll"

        game.request_property_action("A", "mortgage", "The Angel Islington")
        game.resolve_property_action(False)

        self.assertFalse(
            any(
                action.action_type == "request_mortgage" and action.property_name == "The Angel Islington"
                for action in game.get_turn_plan("A").legal_actions
            )
        )
        with self.assertRaises(ValueError):
            game.request_property_action("A", "mortgage", "The Angel Islington")

        game.end_turn_interactive()

        self.assertTrue(
            any(
                action.action_type == "request_mortgage" and action.property_name == "The Angel Islington"
                for action in game.get_turn_plan("A").legal_actions
            )
        )

    def test_execute_serialized_action_can_cancel_property_action(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        oriental = game.board.get_space(6)
        oriental.assign_owner(player)

        request_action = next(
            action
            for action in game.get_turn_plan("A").legal_actions
            if action.action_type == "request_mortgage" and action.property_name == "The Angel Islington"
        )
        request_result = InteractionResult.from_dict(game.execute_serialized_action(request_action.to_dict()))
        self.assertEqual("property_action", request_result.pending_action.action_type)

        cancel_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "cancel_property_action")
        cancel_result = InteractionResult.from_dict(game.execute_serialized_action(cancel_action.to_dict()))

        self.assertFalse(oriental.mortgaged)
        self.assertIsNone(cancel_result.pending_action)

    def test_turn_plan_for_current_player_exposes_start_turn_and_trade_options(self) -> None:
        game = Game(["Human", "Bot"], player_roles={"Human": "human", "Bot": "ai"})

        plan = game.get_turn_plan("Human")

        self.assertTrue(plan.is_current_player)
        self.assertFalse(plan.has_pending_action)
        self.assertEqual("pre_roll", plan.turn_phase)
        self.assertTrue(any(action.action_type == "start_turn" for action in plan.legal_actions))
        self.assertTrue(any(action.action_type == "propose_trade" for action in plan.legal_actions))
        self.assertFalse(any(action.action_type == "end_turn" for action in plan.legal_actions))

    def test_turn_plan_for_pending_ai_purchase_exposes_resolver_options(self) -> None:
        game = Game(["Human", "Bot"], dice=Dice(scripted_rolls=[(1, 1)]), player_roles={"Human": "human", "Bot": "ai"})
        game.current_player_index = 1
        game.players[1].position = 39
        game.start_turn_interactive()

        plan = game.get_turn_plan("Bot")

        self.assertTrue(plan.has_pending_action)
        self.assertEqual("in_turn", plan.turn_phase)
        self.assertEqual("property_purchase", plan.pending_action.action_type)
        self.assertEqual({"buy_property", "decline_property"}, {action.action_type for action in plan.legal_actions})

    def test_turn_plan_for_pending_auction_exposes_bid_range_and_pass(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 1)]), player_roles=["human", "ai", "human"])
        game.players[0].position = 39
        game.start_turn_interactive()
        game.resolve_property_decision(False)

        plan = game.get_turn_plan("A")

        bid_action = next(action for action in plan.legal_actions if action.action_type == "place_auction_bid")
        pass_action = next(action for action in plan.legal_actions if action.action_type == "pass_auction")
        self.assertEqual(1, bid_action.min_bid)
        self.assertEqual(game.players[0].cash, bid_action.max_bid)
        self.assertEqual("submit_auction_bid", pass_action.handler_name)

    def test_turn_plan_for_pending_auction_scales_minimum_raise_after_an_opening_bid(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 1)]), player_roles=["human", "ai", "human"])
        game.players[0].position = 39

        game.start_turn_interactive()
        game.resolve_property_decision(False)
        game.submit_auction_bid("A", 1)

        plan = game.get_turn_plan("B")

        bid_action = next(action for action in plan.legal_actions if action.action_type == "place_auction_bid")
        self.assertEqual(6, bid_action.min_bid)

    def test_turn_plan_for_non_active_player_explains_why_they_cannot_act(self) -> None:
        game = Game(["A", "B"], player_roles=["human", "ai"])

        plan = game.get_turn_plan("B")

        self.assertFalse(plan.is_current_player)
        self.assertEqual((), plan.legal_actions)
        self.assertIn("A", plan.reason)

    def test_turn_plan_for_post_roll_phase_exposes_end_turn(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 2)]), player_roles=["human", "ai"])
        game.players[0].position = 17

        game.start_turn_interactive()
        plan = game.get_turn_plan("A")

        self.assertEqual("post_roll", plan.turn_phase)
        self.assertTrue(any(action.action_type == "end_turn" for action in plan.legal_actions))
        self.assertFalse(any(action.action_type == "start_turn" for action in plan.legal_actions))

    def test_end_turn_interactive_requires_post_roll_phase(self) -> None:
        game = Game(["A", "B"])

        with self.assertRaises(ValueError):
            game.end_turn_interactive()

    def test_interactive_trade_acceptance_uses_pending_action_contract(self) -> None:
        game = Game(["A", "B"], player_roles={"A": "human", "B": "ai"})
        player_a = game.players[0]
        player_b = game.players[1]
        oriental = game.board.get_space(6)
        oriental.assign_owner(player_b)
        player_b.pay(oriental.price)

        request = game.propose_trade_interactive(
            TradeOffer(
                proposer=player_a,
                receiver=player_b,
                proposer_cash=200,
                receiver_properties=[oriental],
                note="Buy The Angel Islington",
            )
        )
        result = game.resolve_trade_decision(True)

        self.assertEqual("trade_decision", request.pending_action.action_type)
        self.assertEqual("B", request.pending_action.player_name)
        self.assertEqual("ai", request.pending_action.player_role)
        self.assertEqual("human", request.pending_action.trade.proposer_role)
        self.assertEqual("ai", request.pending_action.trade.receiver_role)
        self.assertEqual(("accept", "reject", "counter"), request.pending_action.available_actions)
        self.assertIs(oriental.owner, player_a)
        self.assertIsNone(result.pending_action)

    def test_execute_legal_action_can_counter_trade(self) -> None:
        game = Game(["A", "B"], player_roles={"A": "human", "B": "human"})
        player_a = game.players[0]
        player_b = game.players[1]
        old_kent = game.board.get_space(1)
        whitechapel = game.board.get_space(3)
        old_kent.assign_owner(player_a)
        whitechapel.assign_owner(player_b)
        player_a.pay(old_kent.price)
        player_b.pay(whitechapel.price)

        propose_action = next(
            action
            for action in game.get_turn_plan("A").legal_actions
            if action.action_type == "propose_trade" and action.target_player_name == "B"
        )
        game.execute_legal_action(
            propose_action,
            trade_offer=TradeOffer(
                proposer=player_a,
                receiver=player_b,
                proposer_cash=100,
                receiver_properties=[whitechapel],
                note="Initial offer",
            ),
        )

        counter_action = next(action for action in game.get_turn_plan("B").legal_actions if action.action_type == "counter_trade")
        result = game.execute_legal_action(
            counter_action,
            trade_offer=TradeOffer(
                proposer=player_b,
                receiver=player_a,
                proposer_cash=0,
                receiver_cash=50,
                proposer_properties=[whitechapel],
                receiver_properties=[old_kent],
                note="Counter offer",
            ),
        )

        self.assertIsNotNone(result.pending_action)
        self.assertEqual("trade_decision", result.pending_action.action_type)
        self.assertEqual("A", result.pending_action.player_name)
        self.assertEqual("B", result.pending_action.trade.proposer_name)
        self.assertEqual(("Whitechapel Road",), result.pending_action.trade.proposer_property_names)
        self.assertEqual(("Old Kent Road",), result.pending_action.trade.receiver_property_names)
        self.assertIn("counter-offer", result.messages[0])
        self.assertEqual(("accept", "reject"), result.pending_action.available_actions)
        self.assertFalse(any(action.action_type == "counter_trade" for action in game.get_turn_plan("A").legal_actions))

    def test_trade_exchange_cannot_be_countered_twice(self) -> None:
        game = Game(["A", "B"], player_roles={"A": "human", "B": "human"})
        old_kent = game.board.get_space(1)
        whitechapel = game.board.get_space(3)
        old_kent.assign_owner(game.players[0])
        whitechapel.assign_owner(game.players[1])

        propose_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "propose_trade")
        game.execute_legal_action(
            propose_action,
            trade_offer=TradeOffer(
                proposer=game.players[0],
                receiver=game.players[1],
                proposer_cash=50,
                receiver_properties=[whitechapel],
            ),
        )
        counter_action = next(action for action in game.get_turn_plan("B").legal_actions if action.action_type == "counter_trade")
        game.execute_legal_action(
            counter_action,
            trade_offer=TradeOffer(
                proposer=game.players[1],
                receiver=game.players[0],
                proposer_properties=[whitechapel],
                receiver_properties=[old_kent],
            ),
        )

        self.assertFalse(any(action.action_type == "counter_trade" for action in game.get_turn_plan("A").legal_actions))

    def test_interactive_trade_rejection_keeps_assets_unchanged(self) -> None:
        game = Game(["A", "B"])
        player_a = game.players[0]
        player_b = game.players[1]
        oriental = game.board.get_space(6)
        oriental.assign_owner(player_b)
        player_b.pay(oriental.price)

        game.propose_trade_interactive(
            TradeOffer(
                proposer=player_a,
                receiver=player_b,
                proposer_cash=200,
                receiver_properties=[oriental],
            )
        )
        result = game.resolve_trade_decision(False)

        self.assertIs(oriental.owner, player_b)
        self.assertIsNone(result.pending_action)

    def test_start_turn_interactive_exposes_ai_role_for_pending_decision(self) -> None:
        game = Game(["Human", "Bot"], dice=Dice(scripted_rolls=[(1, 1)]), player_roles={"Human": "human", "Bot": "ai"})
        game.current_player_index = 1
        game.players[1].position = 39

        result = game.start_turn_interactive()

        self.assertEqual("Bot", result.pending_action.player_name)
        self.assertEqual("ai", result.pending_action.player_role)

    def test_invalid_player_role_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Game(["A", "B"], player_roles={"A": "human", "B": "robot"})

    def test_property_actions_are_allowed_in_post_roll_phase(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 2)]))
        player = game.players[0]
        mediterranean = game.board.get_space(1)
        baltic = game.board.get_space(3)
        mediterranean.assign_owner(player)
        baltic.assign_owner(player)
        player.cash = 1000
        player.position = 17

        game.start_turn_interactive()
        request = game.request_property_action("A", "build", "Old Kent Road")

        self.assertEqual("property_action", request.pending_action.action_type)
        self.assertEqual("in_turn", request.pending_action.turn_phase)


if __name__ == "__main__":
    unittest.main()