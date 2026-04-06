from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from evaluate_agent import main as evaluate_main
import train_agent
import numpy as np
import torch
from monopoly.agent import (
    AgentPolicyController,
    AgentActionChoice,
    CheckpointEvaluator,
    GameProcessAgentHost,
    HeuristicScorer,
    HeuristicWeights,
    LeagueEpisodeAssignment,
    LeagueManager,
    LeaguePolicySpec,
    MonopolyActionSpace,
    MonopolySelfPlayEnvironment,
    ObservationEncoder,
    ParallelSelfPlayTrainer,
    PolicyConfig,
    RewardFunction,
    RewardWeights,
    build_scripted_controller,
    TorchPolicyModel,
    TrainingExample,
    TrainingConfig,
    load_agent_host_from_checkpoint,
)
from monopoly.agent.board_analysis import analyze_board, estimate_actor_threat
from monopoly.api import LegalActionOption
from monopoly.constants import AI_ROLE
from monopoly.dice import Dice
from monopoly.game import Game, PendingAuctionState, PurchaseDecisionState
from monopoly.agent.worker_pool import PersistentRolloutWorkerPool
from tournament_checkpoints import main as tournament_main
from train_agent import main as train_agent_main


@dataclass
class _ScriptedPolicyDecision:
    action_id: int
    value: float = 0.0
    log_probability: float = 0.0


class _ScriptedController:
    def __init__(self, scripts: dict[str, list[str]]) -> None:
        self._scripts = {player_name: list(labels) for player_name, labels in scripts.items()}
        self._encoder = ObservationEncoder()
        self._action_space = MonopolyActionSpace()

    def choose_action(self, game: Game, actor_name: str, explore: bool = True):
        del explore
        frontend_state = game.get_frontend_state()
        turn_plan = game.get_turn_plan(actor_name)
        observation = self._encoder.encode(frontend_state, actor_name)
        action_mask, choices = self._action_space.build_mask(turn_plan, frontend_state)
        desired_label = self._scripts[actor_name].pop(0)
        choice = next(option for option in choices.values() if option.action_label == desired_label)
        return SimpleNamespace(
            choice=choice,
            observation=observation,
            action_mask=action_mask,
            heuristic_bias=np.zeros(self._action_space.action_count, dtype=np.float32),
            decision=_ScriptedPolicyDecision(action_id=choice.action_id),
        )

    def evaluate_state_values(self, frontend_state, actor_names: tuple[str, ...]):
        del frontend_state
        return {actor_name: 0.0 for actor_name in actor_names}


class _FixedActionModel:
    def __init__(self, action_id: int) -> None:
        self._action_id = action_id

    def act(self, observation, action_mask, heuristic_bias, explore=True, use_heuristic_bias=True):
        del observation, heuristic_bias, explore, use_heuristic_bias
        candidate_ids = np.flatnonzero(action_mask)
        action_id = self._action_id if bool(action_mask[self._action_id]) else int(candidate_ids[0])
        return SimpleNamespace(action_id=action_id, log_probability=0.0, value=0.0)


class _AuctionEnvironment(MonopolySelfPlayEnvironment):
    def create_game(self, seed: int | None = None) -> Game:
        del seed
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        property_space = game.board.get_space(1)
        game.pending_auction = PendingAuctionState(
            property_index=property_space.index,
            property_name=property_space.name,
            eligible_player_names=["A", "B"],
            active_player_names=["A", "B"],
            current_bid=0,
            current_winner_name=None,
            current_bidder_index=0,
        )
        return game


class _MultiBidAuctionEnvironment(MonopolySelfPlayEnvironment):
    def create_game(self, seed: int | None = None) -> Game:
        del seed
        game = Game(["A", "B", "C"], player_roles=[AI_ROLE, AI_ROLE, AI_ROLE])
        property_space = game.board.get_space(1)
        game.pending_auction = PendingAuctionState(
            property_index=property_space.index,
            property_name=property_space.name,
            eligible_player_names=["A", "B", "C"],
            active_player_names=["A", "B", "C"],
            current_bid=0,
            current_winner_name=None,
            current_bidder_index=0,
        )
        return game


class AgentTests(unittest.TestCase):
    def _zeroed_model(self, encoder: ObservationEncoder, action_space: MonopolyActionSpace) -> TorchPolicyModel:
        model = TorchPolicyModel(encoder.observation_size, action_space.action_count, seed=3, hidden_sizes=(64, 64))
        model.zero_parameters()
        return model

    def _write_checkpoint(self, directory: str, *, seed: int = 3) -> Path:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        trainer = ParallelSelfPlayTrainer(
            policy_model=TorchPolicyModel(
                encoder.observation_size,
                action_space.action_count,
                seed=seed,
                hidden_sizes=(64, 64),
            ),
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=4, players_per_game=2, checkpoint_interval=1),
            policy_config=PolicyConfig(seed=seed, hidden_sizes=(64, 64), ppo_epochs=1, minibatch_size=8),
        )
        checkpoint_path = Path(directory) / f"checkpoint_{seed}.pt"
        trainer.save_checkpoint(checkpoint_path)
        return checkpoint_path

    def test_action_space_encodes_purchase_and_auction_actions(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 1)]), player_roles=[AI_ROLE, AI_ROLE, AI_ROLE])
        game.players[0].position = 39
        game.start_turn_interactive()
        action_space = MonopolyActionSpace()

        purchase_mask, purchase_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())

        self.assertTrue(any(choice.action_label == "buy_property" for choice in purchase_choices.values()))
        self.assertTrue(any(choice.action_label == "decline_property" for choice in purchase_choices.values()))
        self.assertGreater(int(purchase_mask.sum()), 1)

        game.resolve_property_decision(False)
        auction_mask, auction_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        bid_labels = {choice.action_label for choice in auction_choices.values()}

        self.assertIn("pass_auction", bid_labels)
        self.assertIn("auction_bid_min", bid_labels)
        self.assertIn("auction_bid_step", bid_labels)
        self.assertIn("auction_bid_denial", bid_labels)
        self.assertGreaterEqual(int(auction_mask.sum()), 2)

    def test_action_space_skips_unaffordable_auction_bid_choices(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        action_space = MonopolyActionSpace()
        property_space = game.board.get_space(1)
        game.pending_auction = PendingAuctionState(
            property_index=property_space.index,
            property_name=property_space.name,
            eligible_player_names=["A", "B"],
            active_player_names=["A", "B"],
            current_bid=150,
            current_winner_name="B",
            current_bidder_index=0,
        )
        game.players[0].cash = 100

        auction_mask, auction_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        bid_labels = {choice.action_label for choice in auction_choices.values()}

        self.assertEqual({"pass_auction"}, bid_labels)
        self.assertEqual(1, int(auction_mask.sum()))

    def test_action_space_anchors_auction_bid_buckets_to_property_value(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        action_space = MonopolyActionSpace()
        property_space = game.board.get_space(1)
        game.pending_auction = PendingAuctionState(
            property_index=property_space.index,
            property_name=property_space.name,
            eligible_player_names=["A", "B"],
            active_player_names=["A", "B"],
            current_bid=0,
            current_winner_name=None,
            current_bidder_index=0,
        )

        _, auction_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        bids_by_label = {
            choice.action_label: choice.bid_amount
            for choice in auction_choices.values()
            if choice.bid_amount is not None
        }

        self.assertEqual(1, bids_by_label["auction_bid_min"])
        self.assertEqual(11, bids_by_label["auction_bid_step"])
        self.assertEqual(51, bids_by_label["auction_bid_reserve"])
        self.assertEqual(60, bids_by_label["auction_bid_value"])
        self.assertEqual(72, bids_by_label["auction_bid_premium"])
        self.assertEqual(87, bids_by_label["auction_bid_aggressive"])
        self.assertEqual(93, bids_by_label["auction_bid_denial"])

    def test_action_space_raises_denial_anchor_when_opponent_has_color_progress(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        action_space = MonopolyActionSpace()
        game.board.get_space(3).assign_owner(game.players[1])
        property_space = game.board.get_space(1)
        game.pending_auction = PendingAuctionState(
            property_index=property_space.index,
            property_name=property_space.name,
            eligible_player_names=["A", "B"],
            active_player_names=["A", "B"],
            current_bid=0,
            current_winner_name=None,
            current_bidder_index=0,
        )

        _, auction_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        bids_by_label = {
            choice.action_label: choice.bid_amount
            for choice in auction_choices.values()
            if choice.bid_amount is not None
        }

        self.assertGreater(bids_by_label["auction_bid_denial"], bids_by_label["auction_bid_aggressive"])

    def test_action_space_generates_board_aware_trade_templates(self) -> None:
        game = Game(["A", "B", "C"], player_roles=[AI_ROLE, AI_ROLE, AI_ROLE])
        old_kent = game.board.get_space(1)
        kings_cross = game.board.get_space(5)
        angel = game.board.get_space(6)
        euston = game.board.get_space(8)
        whitechapel = game.board.get_space(3)
        old_kent.assign_owner(game.players[0])
        kings_cross.assign_owner(game.players[0])
        angel.assign_owner(game.players[0])
        whitechapel.assign_owner(game.players[1])
        euston.assign_owner(game.players[1])
        game.board.get_space(9).assign_owner(game.players[2])
        action_space = MonopolyActionSpace()

        _, choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        trade_choices = [choice for choice in choices.values() if choice.trade_offer_payload is not None]
        trade_labels = {choice.action_label for choice in trade_choices}

        self.assertTrue(any(label.startswith("trade_request_completion_cash_light") for label in trade_labels))
        self.assertTrue(any(label.startswith("trade_request_completion_cash_heavy") for label in trade_labels))
        self.assertTrue(any(label.startswith("trade_request_completion_property_cash") for label in trade_labels))
        self.assertTrue(any(label.startswith("trade_request_expansion_cash_light") for label in trade_labels))
        self.assertTrue(any(label.startswith("trade_swap_low_for_best_fit") for label in trade_labels))

        for choice in trade_choices:
            payload = choice.trade_offer_payload
            assert payload is not None
            proposer_gives = bool(payload.get("proposer_cash", 0) or payload.get("proposer_property_names"))
            receiver_gives = bool(payload.get("receiver_cash", 0) or payload.get("receiver_property_names"))
            self.assertTrue(proposer_gives)
            self.assertTrue(receiver_gives)

        completion_choice = next(
            choice
            for choice in trade_choices
            if choice.action_label.startswith("trade_request_completion_cash_light")
        )
        self.assertEqual(["Whitechapel Road"], completion_choice.trade_offer_payload["receiver_property_names"])
        self.assertGreater(completion_choice.trade_offer_payload["proposer_cash"], 0)

        expansion_choice = next(
            choice
            for choice in trade_choices
            if choice.action_label.startswith("trade_request_expansion_cash_light")
        )
        self.assertEqual(["Euston Road"], expansion_choice.trade_offer_payload["receiver_property_names"])
        self.assertGreater(expansion_choice.trade_offer_payload["proposer_cash"], 0)

    def test_trade_template_payload_executes_through_game(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        old_kent = game.board.get_space(1)
        old_kent.assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        action_space = MonopolyActionSpace()

        _, choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        trade_choice = next(choice for choice in choices.values() if choice.trade_offer_payload is not None)
        interaction = game.execute_legal_action(
            trade_choice.legal_action,
            trade_offer=game.deserialize_trade_offer(trade_choice.trade_offer_payload),
        )

        self.assertIsNotNone(interaction.pending_action)
        self.assertEqual("trade_decision", interaction.pending_action.action_type)

    def test_action_space_encodes_counter_trade_choices_for_pending_decision(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        action_space = MonopolyActionSpace()

        propose_action = next(
            action
            for action in game.get_turn_plan("A").legal_actions
            if action.action_type == "propose_trade" and action.target_player_name == "B"
        )
        trade_choice = next(
            choice
            for choice in action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())[1].values()
            if choice.trade_offer_payload is not None
        )
        game.execute_legal_action(
            propose_action,
            trade_offer=game.deserialize_trade_offer(trade_choice.trade_offer_payload),
        )

        _, counter_choices = action_space.build_mask(game.get_turn_plan("B"), game.get_frontend_state())

        self.assertIn("accept_trade", {choice.action_label for choice in counter_choices.values() if choice.trade_offer_payload is None})
        self.assertIn("reject_trade", {choice.action_label for choice in counter_choices.values() if choice.trade_offer_payload is None})
        counter_trade_choices = [choice for choice in counter_choices.values() if choice.trade_offer_payload is not None]
        self.assertTrue(counter_trade_choices)
        self.assertTrue(all(choice.legal_action.action_type == "counter_trade" for choice in counter_trade_choices))

    def test_trade_proposals_are_limited_to_one_pre_roll_and_one_post_roll_per_turn(self) -> None:
        game = Game(["A", "B", "C"], player_roles=[AI_ROLE, AI_ROLE, AI_ROLE])
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        game.board.get_space(6).assign_owner(game.players[2])
        action_space = MonopolyActionSpace()

        pre_roll_trade_actions = [action for action in game.get_turn_plan("A").legal_actions if action.action_type == "propose_trade"]
        self.assertEqual(2, len(pre_roll_trade_actions))

        _, pre_roll_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        pre_roll_trade_choice = next(choice for choice in pre_roll_choices.values() if choice.trade_offer_payload is not None)
        game.execute_legal_action(
            pre_roll_trade_choice.legal_action,
            trade_offer=game.deserialize_trade_offer(pre_roll_trade_choice.trade_offer_payload),
        )
        game.resolve_trade_decision(False)

        pre_roll_trade_actions_after_rejection = [action for action in game.get_turn_plan("A").legal_actions if action.action_type == "propose_trade"]
        self.assertEqual([], pre_roll_trade_actions_after_rejection)

        game.current_turn_phase = "post_roll"
        post_roll_trade_actions = [action for action in game.get_turn_plan("A").legal_actions if action.action_type == "propose_trade"]
        self.assertEqual(2, len(post_roll_trade_actions))

        _, post_roll_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        post_roll_trade_choice = next(choice for choice in post_roll_choices.values() if choice.trade_offer_payload is not None)
        game.execute_legal_action(
            post_roll_trade_choice.legal_action,
            trade_offer=game.deserialize_trade_offer(post_roll_trade_choice.trade_offer_payload),
        )
        game.resolve_trade_decision(False)

        post_roll_trade_actions_after_rejection = [action for action in game.get_turn_plan("A").legal_actions if action.action_type == "propose_trade"]
        self.assertEqual([], post_roll_trade_actions_after_rejection)

    def test_rejected_trade_offer_is_not_encodable_again_later_in_the_same_turn(self) -> None:
        game = Game(["A", "B", "C"], player_roles=[AI_ROLE, AI_ROLE, AI_ROLE])
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        game.board.get_space(6).assign_owner(game.players[2])
        action_space = MonopolyActionSpace()

        _, pre_roll_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        rejected_choice = next(choice for choice in pre_roll_choices.values() if choice.trade_offer_payload is not None)
        rejected_signature = game._trade_offer_signature_from_payload(rejected_choice.trade_offer_payload)
        game.execute_legal_action(
            rejected_choice.legal_action,
            trade_offer=game.deserialize_trade_offer(rejected_choice.trade_offer_payload),
        )
        game.resolve_trade_decision(False)

        game.current_turn_phase = "post_roll"
        _, post_roll_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        remaining_trade_choices = [choice for choice in post_roll_choices.values() if choice.trade_offer_payload is not None]

        self.assertTrue(remaining_trade_choices)
        self.assertNotIn(
            rejected_signature,
            {game._trade_offer_signature_from_payload(choice.trade_offer_payload) for choice in remaining_trade_choices},
        )

    def test_action_space_drops_trade_target_when_all_templates_for_that_opponent_are_blocked(self) -> None:
        game = Game(["A", "B", "C"], player_roles=[AI_ROLE, AI_ROLE, AI_ROLE])
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        game.board.get_space(6).assign_owner(game.players[2])
        action_space = MonopolyActionSpace()

        _, initial_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        blocked_opponent_choices = [
            choice
            for choice in initial_choices.values()
            if choice.trade_offer_payload is not None and choice.legal_action.target_player_name == "B"
        ]
        self.assertTrue(blocked_opponent_choices)

        game._blocked_trade_offer_signatures = {
            game._trade_offer_signature_from_payload(choice.trade_offer_payload)
            for choice in blocked_opponent_choices
        }

        _, filtered_choices = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())
        filtered_trade_choices = [choice for choice in filtered_choices.values() if choice.trade_offer_payload is not None]

        self.assertFalse(any(choice.legal_action.target_player_name == "B" for choice in filtered_trade_choices))
        self.assertTrue(any(choice.legal_action.target_player_name == "C" for choice in filtered_trade_choices))

    def test_agent_policy_controller_filters_invalid_trade_templates_before_sampling(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        turn_plan = game.get_turn_plan("A")
        start_turn_action = next(action for action in turn_plan.legal_actions if action.action_type == "start_turn")
        trade_action = next(action for action in turn_plan.legal_actions if action.action_type == "propose_trade")
        invalid_trade_choice = AgentActionChoice(
            action_id=1,
            action_label="trade_cash_request_100:1",
            legal_action=trade_action,
            trade_offer_payload={
                "proposer_name": "A",
                "receiver_name": "B",
                "proposer_cash": 0,
                "receiver_cash": 999999,
                "proposer_property_names": [],
                "receiver_property_names": [],
                "proposer_jail_cards": 0,
                "receiver_jail_cards": 0,
                "note": "invalid",
            },
        )
        valid_start_turn_choice = AgentActionChoice(
            action_id=0,
            action_label="start_turn",
            legal_action=start_turn_action,
        )

        class _FixedActionSpace:
            action_count = 2

            def build_mask(self, turn_plan, frontend_state):
                del turn_plan, frontend_state
                return np.asarray([True, True], dtype=bool), {0: valid_start_turn_choice, 1: invalid_trade_choice}

        controller = AgentPolicyController(
            policy_model=_FixedActionModel(action_id=1),
            observation_encoder=ObservationEncoder(),
            action_space=_FixedActionSpace(),
            heuristic_scorer=HeuristicScorer(),
        )

        decision = controller.choose_action(game, "A", explore=False)

        self.assertEqual("start_turn", decision.choice.action_label)

    def test_bankrupt_player_pending_purchase_is_cleared_before_ai_choice(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        controller = AgentPolicyController(
            policy_model=self._zeroed_model(encoder, action_space),
            observation_encoder=encoder,
            action_space=action_space,
            heuristic_scorer=HeuristicScorer(),
        )
        game = Game(["A", "B", "C", "D"], player_roles=[AI_ROLE, AI_ROLE, AI_ROLE, AI_ROLE])
        game.players[2].is_bankrupt = True
        game.players[2].cash = 30
        game.current_player_index = 2
        game.current_turn_phase = "in_turn"
        game.pending_purchase_decision = PurchaseDecisionState(
            player_name="C",
            property_index=1,
            property_name="Old Kent Road",
            price=60,
        )

        decision = controller.choose_action(game, game.get_active_turn_plan().player_name, explore=False)

        self.assertEqual("D", decision.choice.legal_action.actor_name)
        self.assertEqual("start_turn", decision.choice.action_label)

    def test_observation_encoder_returns_stable_vector_length(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        encoder = ObservationEncoder()

        observation = encoder.encode(game.get_frontend_state(), "A")

        self.assertEqual(encoder.observation_size, len(observation))
        self.assertGreater(len(observation), 200)
        self.assertEqual("float32", str(observation.dtype))

    def test_observation_encoder_adds_trade_specific_features(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        old_kent = game.board.get_space(1)
        whitechapel = game.board.get_space(3)
        old_kent.assign_owner(game.players[0])
        whitechapel.assign_owner(game.players[1])
        encoder = ObservationEncoder()

        baseline = encoder.encode(game.get_frontend_state(), "A")
        trade_action = next(
            action
            for action in game.get_turn_plan("A").legal_actions
            if action.action_type == "propose_trade" and action.target_player_name == "B"
        )
        game.execute_legal_action(
            trade_action,
            trade_offer=game.deserialize_trade_offer(
                {
                    "proposer_name": "A",
                    "receiver_name": "B",
                    "proposer_cash": 50,
                    "receiver_cash": 0,
                    "proposer_property_names": ["Old Kent Road"],
                    "receiver_property_names": [],
                    "proposer_jail_cards": 0,
                    "receiver_jail_cards": 0,
                    "note": "test",
                }
            ),
        )
        trade_state = encoder.encode(game.get_frontend_state(), "B")

        self.assertEqual(encoder.observation_size, len(trade_state))
        self.assertFalse(np.array_equal(baseline, trade_state))

    def test_observation_encoder_distinguishes_opponent_owner_identity_and_board_metrics(self) -> None:
        game = Game(["A", "B", "C"], player_roles=[AI_ROLE, AI_ROLE, AI_ROLE])
        old_kent = game.board.get_space(1)
        whitechapel = game.board.get_space(16)
        old_kent.assign_owner(game.players[1])
        whitechapel.assign_owner(game.players[2])
        whitechapel.building_count = 2
        encoder = ObservationEncoder()

        observation = encoder.encode(game.get_frontend_state(), "A")
        analysis = analyze_board(game.get_frontend_state(), "A")

        self.assertEqual(encoder.observation_size, len(observation))
        self.assertNotEqual(
            analysis.owner_slot_by_space_index[old_kent.index],
            analysis.owner_slot_by_space_index[whitechapel.index],
        )
        self.assertGreater(analysis.player_metrics["C"].developed_value, analysis.player_metrics["B"].developed_value)

    def test_board_analysis_reports_positive_threat_when_strongest_opponent_is_ahead(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        dark_blue_one = game.board.get_space(37)
        dark_blue_two = game.board.get_space(39)
        dark_blue_one.assign_owner(game.players[1])
        dark_blue_two.assign_owner(game.players[1])

        threat = estimate_actor_threat(game.get_frontend_state(), "A")

        self.assertGreater(threat, 0.0)

    def test_cancelled_property_action_is_blocked_for_the_rest_of_the_turn(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        property_space = game.board.get_space(1)
        property_space.assign_owner(game.players[0])
        game.current_turn_phase = "post_roll"

        game.request_property_action("A", "mortgage", property_space.name)
        game.resolve_property_action(False)

        legal_actions = game.get_turn_plan("A").legal_actions
        mortgage_actions = [
            action for action in legal_actions if action.action_type == "request_mortgage" and action.property_name == property_space.name
        ]

        self.assertEqual([], mortgage_actions)
        self.assertTrue(any(action.action_type == "end_turn" for action in legal_actions))

    def test_reward_function_respects_adjustable_weights(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1), (1, 2)]), player_roles=[AI_ROLE, AI_ROLE])
        game.players[0].position = 39
        previous_state = game.get_frontend_state()
        game.start_turn_interactive()
        game.resolve_property_decision(True)
        next_state = game.get_frontend_state()
        reward_function = RewardFunction(
            RewardWeights(
                win_reward=0.0,
                loss_penalty=0.0,
                cash_delta_weight=0.0,
                net_worth_delta_weight=0.0,
                property_gain_weight=1.0,
                monopoly_gain_weight=0.0,
                rent_potential_weight=0.0,
                buildable_monopoly_weight=0.0,
                near_monopoly_weight=0.0,
                cluster_strength_weight=0.0,
                relative_board_strength_weight=0.0,
                opponent_rent_pressure_weight=0.0,
                opponent_buildability_denial_weight=0.0,
                strategic_property_weight=0.0,
                bankruptcy_penalty=0.0,
                opponent_bankruptcy_reward=0.0,
                jail_enter_penalty=0.0,
                turn_completion_reward=0.0,
                auction_overpay_weight=0.0,
                auction_liquidity_weight=0.0,
            )
        )

        reward = reward_function.score_transition(previous_state, next_state, "A")

        self.assertEqual(1.0, reward.total)
        self.assertEqual(1.0, reward.property_delta)

    def test_reward_function_penalizes_auction_overpay_and_low_liquidity_on_resolution(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        property_space = game.board.get_space(1)
        game.pending_auction = PendingAuctionState(
            property_index=property_space.index,
            property_name=property_space.name,
            eligible_player_names=["A", "B"],
            active_player_names=["A", "B"],
            current_bid=200,
            current_winner_name="A",
            current_bidder_index=1,
        )
        game.players[0].cash = 250
        previous_state = game.get_frontend_state()
        game.submit_auction_bid("B", None)
        next_state = game.get_frontend_state()
        reward_function = RewardFunction(
            RewardWeights(
                win_reward=0.0,
                loss_penalty=0.0,
                cash_delta_weight=0.0,
                net_worth_delta_weight=0.0,
                property_gain_weight=0.0,
                monopoly_gain_weight=0.0,
                bankruptcy_penalty=0.0,
                opponent_bankruptcy_reward=0.0,
                jail_enter_penalty=0.0,
                turn_completion_reward=0.0,
                auction_overpay_weight=0.01,
                auction_liquidity_weight=0.01,
                auction_cash_reserve_ratio=0.2,
            )
        )

        reward = reward_function.score_transition(previous_state, next_state, "A")

        self.assertLess(reward.total, 0.0)
        self.assertLess(reward.auction_overpay_delta, 0.0)
        self.assertLess(reward.auction_liquidity_delta, 0.0)

    def test_reward_function_rewards_buildable_cluster_progress_and_relative_strength(self) -> None:
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        old_kent = game.board.get_space(1)
        baltic = game.board.get_space(3)
        old_kent.assign_owner(game.players[0])
        previous_state = game.get_frontend_state()
        baltic.assign_owner(game.players[0])
        next_state = game.get_frontend_state()
        reward_function = RewardFunction(
            RewardWeights(
                win_reward=0.0,
                loss_penalty=0.0,
                cash_delta_weight=0.0,
                net_worth_delta_weight=0.0,
                property_gain_weight=0.0,
                monopoly_gain_weight=0.0,
                rent_potential_weight=0.01,
                buildable_monopoly_weight=1.0,
                near_monopoly_weight=0.5,
                cluster_strength_weight=0.5,
                relative_board_strength_weight=0.001,
                opponent_rent_pressure_weight=0.0,
                opponent_buildability_denial_weight=0.0,
                strategic_property_weight=0.001,
                bankruptcy_penalty=0.0,
                opponent_bankruptcy_reward=0.0,
                jail_enter_penalty=0.0,
                turn_completion_reward=0.0,
                auction_overpay_weight=0.0,
                auction_liquidity_weight=0.0,
            )
        )

        reward = reward_function.score_transition(previous_state, next_state, "A")

        self.assertGreater(reward.total, 0.0)
        self.assertGreater(reward.buildable_monopoly_delta, 0.0)
        self.assertGreater(reward.cluster_strength_delta, 0.0)
        self.assertGreater(reward.relative_board_strength_delta, 0.0)

    def test_auction_heuristics_are_neutral_by_default(self) -> None:
        weights = HeuristicWeights()

        self.assertEqual(0.0, weights.auction_min_bid_bias)
        self.assertEqual(0.0, weights.auction_value_bid_bias)
        self.assertEqual(0.0, weights.auction_premium_bid_bias)
        self.assertEqual(0.0, weights.auction_aggressive_bid_bias)
        self.assertEqual(0.0, weights.pass_auction_bias)

    def test_heuristic_scorer_respects_scale(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 2)]), player_roles=[AI_ROLE, AI_ROLE])
        frontend_state = game.get_frontend_state()
        turn_plan = game.get_turn_plan("A")
        action_space = MonopolyActionSpace()
        action_mask, choices = action_space.build_mask(turn_plan, frontend_state)
        del action_mask

        scorer = HeuristicScorer(HeuristicWeights())
        full_bias = scorer.score(frontend_state, choices, "A", action_space.action_count, heuristic_scale=1.0)
        zero_bias = scorer.score(frontend_state, choices, "A", action_space.action_count, heuristic_scale=0.0)

        self.assertGreater(np.abs(full_bias).sum(), 0.0)
        self.assertEqual(0.0, float(np.abs(zero_bias).sum()))

    def test_policy_model_can_ignore_heuristic_bias(self) -> None:
        model = TorchPolicyModel(observation_size=4, action_count=2, seed=5, hidden_sizes=(8, 8))
        model.zero_parameters()
        observation = np.zeros(4, dtype=np.float32)
        action_mask = np.asarray([True, True], dtype=np.bool_)
        heuristic_bias = np.asarray([0.0, 10.0], dtype=np.float32)

        biased_decision = model.act(observation, action_mask, heuristic_bias, explore=False, use_heuristic_bias=True)
        unbiased_decision = model.act(observation, action_mask, heuristic_bias, explore=False, use_heuristic_bias=False)

        self.assertEqual(1, biased_decision.action_id)
        self.assertEqual(0, unbiased_decision.action_id)

    def test_trainer_resolves_linear_heuristic_anneal_scale(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        trainer = ParallelSelfPlayTrainer(
            policy_model=self._zeroed_model(encoder, action_space),
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=4, players_per_game=2),
            policy_config=PolicyConfig(
                seed=17,
                hidden_sizes=(64, 64),
                minibatch_size=8,
                ppo_epochs=1,
                use_heuristic_bias=True,
                heuristic_anneal_schedule="linear",
                heuristic_bias_start=1.0,
                heuristic_bias_end=0.25,
                heuristic_anneal_iterations=4,
            ),
        )

        self.assertEqual(1.0, trainer.current_heuristic_scale(0))
        self.assertAlmostEqual(0.625, trainer.current_heuristic_scale(2), places=6)
        self.assertEqual(0.25, trainer.current_heuristic_scale(4))

        trainer.policy_config.use_heuristic_bias = False
        self.assertEqual(0.0, trainer.current_heuristic_scale(2))

    def test_game_process_agent_host_can_drive_ai_turns(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        controller = AgentPolicyController(
            policy_model=self._zeroed_model(encoder, action_space),
            observation_encoder=encoder,
            action_space=action_space,
            heuristic_scorer=HeuristicScorer(HeuristicWeights()),
        )
        host = GameProcessAgentHost(controller)
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1), (1, 2)]), player_roles=[AI_ROLE, AI_ROLE])
        game.players[0].position = 39

        decisions = host.play_ai_actions(game, explore=False, max_actions=4)

        self.assertGreaterEqual(len(decisions), 2)
        self.assertIs(game.board.get_space(1).owner, game.players[0])
        self.assertEqual("B", game.current_player.name)

    def test_self_play_environment_runs_episode_and_collects_training_examples(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        controller = AgentPolicyController(
            policy_model=self._zeroed_model(encoder, action_space),
            observation_encoder=encoder,
            action_space=action_space,
            heuristic_scorer=HeuristicScorer(),
        )
        environment = MonopolySelfPlayEnvironment(
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=20, players_per_game=2),
            reward_function=RewardFunction(),
            controller=controller,
            discount_gamma=0.95,
        )

        episode = environment.run_episode(seed=11, explore=False)

        self.assertGreater(episode.step_count, 0)
        self.assertEqual(episode.step_count, episode.macro_step_count)
        self.assertGreaterEqual(episode.raw_action_count, episode.macro_step_count)
        self.assertGreater(len(episode.training_examples), 0)
        self.assertLessEqual(episode.step_count, 20)

    def test_self_play_environment_attributes_delayed_auction_reward_to_bid_action(self) -> None:
        controller = _ScriptedController({"A": ["auction_bid_value"], "B": ["pass_auction"]})
        reward_function = RewardFunction(
            RewardWeights(
                win_reward=0.0,
                loss_penalty=0.0,
                cash_delta_weight=0.0,
                net_worth_delta_weight=0.0,
                property_gain_weight=1.0,
                monopoly_gain_weight=0.0,
                rent_potential_weight=0.0,
                buildable_monopoly_weight=0.0,
                near_monopoly_weight=0.0,
                cluster_strength_weight=0.0,
                relative_board_strength_weight=0.0,
                opponent_rent_pressure_weight=0.0,
                opponent_buildability_denial_weight=0.0,
                strategic_property_weight=0.0,
                bankruptcy_penalty=0.0,
                opponent_bankruptcy_reward=0.0,
                jail_enter_penalty=0.0,
                turn_completion_reward=0.0,
                auction_overpay_weight=0.0,
                auction_liquidity_weight=0.0,
                auction_cash_reserve_ratio=0.0,
            )
        )
        environment = _AuctionEnvironment(
            training_config=TrainingConfig(
                worker_count=1,
                episodes_per_worker=1,
                max_steps_per_episode=2,
                players_per_game=2,
                use_auction_macro_steps=False,
            ),
            reward_function=reward_function,
            controller=controller,
            discount_gamma=0.9,
            gae_lambda=1.0,
        )

        episode = environment.run_episode(explore=False)
        bid_example = next(example for example in episode.training_examples if example.action_id == MonopolyActionSpace()._generic_action_ids["auction_bid_value"])

        self.assertAlmostEqual(0.9, bid_example.discounted_return, places=5)
        self.assertAlmostEqual(0.9, bid_example.advantage, places=5)

    def test_self_play_environment_macro_step_counts_full_auction_once(self) -> None:
        controller = _ScriptedController({
            "A": ["auction_bid_min", "auction_bid_min"],
            "B": ["auction_bid_value", "pass_auction"],
            "C": ["pass_auction"],
        })
        reward_function = RewardFunction(
            RewardWeights(
                win_reward=0.0,
                loss_penalty=0.0,
                cash_delta_weight=0.0,
                net_worth_delta_weight=0.0,
                property_gain_weight=1.0,
                monopoly_gain_weight=0.0,
                bankruptcy_penalty=0.0,
                opponent_bankruptcy_reward=0.0,
                jail_enter_penalty=0.0,
                turn_completion_reward=0.0,
                auction_overpay_weight=0.0,
                auction_liquidity_weight=0.0,
                auction_cash_reserve_ratio=0.0,
            )
        )
        environment = _MultiBidAuctionEnvironment(
            training_config=TrainingConfig(
                worker_count=1,
                episodes_per_worker=1,
                max_steps_per_episode=1,
                max_actions_per_episode=8,
                players_per_game=3,
                use_auction_macro_steps=True,
            ),
            reward_function=reward_function,
            controller=controller,
            discount_gamma=0.9,
        )

        episode = environment.run_episode(explore=False)

        self.assertEqual(1, episode.step_count)
        self.assertEqual(1, episode.macro_step_count)
        self.assertEqual(5, episode.raw_action_count)
        self.assertEqual(1, episode.auction_count)
        self.assertEqual(0, episode.truncated_auction_count)
        self.assertEqual(5, len(episode.training_examples))
        repeated_bid_returns = [
            example.discounted_return
            for example in episode.training_examples
            if example.action_id == MonopolyActionSpace()._generic_action_ids["auction_bid_min"]
        ]
        self.assertEqual(2, len(repeated_bid_returns))
        self.assertEqual(repeated_bid_returns[0], repeated_bid_returns[1])

    def test_self_play_environment_macro_step_honors_raw_action_cap(self) -> None:
        controller = _ScriptedController({
            "A": ["auction_bid_min"],
            "B": ["auction_bid_value"],
        })
        environment = _MultiBidAuctionEnvironment(
            training_config=TrainingConfig(
                worker_count=1,
                episodes_per_worker=1,
                max_steps_per_episode=1,
                max_actions_per_episode=2,
                players_per_game=3,
                use_auction_macro_steps=True,
            ),
            reward_function=RewardFunction(),
            controller=controller,
            discount_gamma=0.9,
        )

        episode = environment.run_episode(explore=False)

        self.assertEqual(1, episode.step_count)
        self.assertEqual(1, episode.macro_step_count)
        self.assertEqual(2, episode.raw_action_count)
        self.assertEqual(1, episode.auction_count)
        self.assertEqual(1, episode.truncated_auction_count)
        self.assertEqual(2, len(episode.training_examples))

    def test_train_agent_rejects_macro_step_mode_mismatch_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir, seed=41)

            with self.assertRaisesRegex(ValueError, "auction macro-step setting"):
                train_agent_main([
                    "--resume",
                    str(checkpoint_path),
                    "--iterations",
                    "0",
                    "--checkpoint-dir",
                    temp_dir,
                    "--threads",
                    "1",
                    "--no-auction-macro-steps",
                ])

    def test_parallel_trainer_can_train_and_round_trip_checkpoint(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        model = self._zeroed_model(encoder, action_space)
        trainer = ParallelSelfPlayTrainer(
            policy_model=model,
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=12, players_per_game=2, checkpoint_interval=1),
            policy_config=PolicyConfig(seed=5, hidden_sizes=(64, 64), minibatch_size=16, ppo_epochs=1),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.training_config.output_directory = temp_dir
            stats = trainer.train(1)
            checkpoint_path = Path(temp_dir) / "iteration_0001.pt"

            self.assertEqual(1, len(stats))
            self.assertTrue(checkpoint_path.exists())
            self.assertGreaterEqual(stats[0].average_macro_steps, 0.0)
            self.assertGreaterEqual(stats[0].average_raw_actions, stats[0].average_macro_steps)
            self.assertGreaterEqual(stats[0].auction_truncation_rate, 0.0)
            self.assertLessEqual(stats[0].auction_truncation_rate, 1.0)
            self.assertIsInstance(stats[0].average_rent_potential_trend, float)
            self.assertIsInstance(stats[0].average_monopoly_denial_events, float)
            self.assertIsInstance(stats[0].average_board_strength_trend, float)
            self.assertGreaterEqual(stats[0].average_auction_bid_quality, 0.0)
            self.assertGreaterEqual(stats[0].truncated_bootstrap_rate, 0.0)
            self.assertLessEqual(stats[0].truncated_bootstrap_rate, 1.0)
            self.assertGreaterEqual(stats[0].rollout_seconds, 0.0)
            self.assertGreaterEqual(stats[0].update_seconds, 0.0)
            restored_trainer = ParallelSelfPlayTrainer.load_checkpoint(checkpoint_path)

        self.assertEqual(model.observation_size, restored_trainer.policy_model.observation_size)
        self.assertEqual(model.action_count, restored_trainer.policy_model.action_count)
        self.assertEqual((64, 64), restored_trainer.policy_model.hidden_sizes)
        self.assertEqual(1, restored_trainer.completed_iterations)

    def test_parallel_trainer_reuses_persistent_worker_pool_across_iterations(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        trainer = ParallelSelfPlayTrainer(
            policy_model=self._zeroed_model(encoder, action_space),
            training_config=TrainingConfig(worker_count=2, episodes_per_worker=1, max_steps_per_episode=8, players_per_game=2),
            policy_config=PolicyConfig(seed=13, hidden_sizes=(64, 64), minibatch_size=8, ppo_epochs=1),
        )

        try:
            first_results = trainer._collect_worker_results(0)
            first_pool = trainer._worker_pool
            second_results = trainer._collect_worker_results(1)

            self.assertIsNotNone(first_pool)
            self.assertIs(first_pool, trainer._worker_pool)
            self.assertEqual(2, len(first_results))
            self.assertEqual(2, len(second_results))
            self.assertTrue(all(len(result.episodes) == 1 for result in first_results))
            self.assertTrue(all(len(result.episodes) == 1 for result in second_results))
        finally:
            trainer.close()

    def test_parallel_trainer_close_is_idempotent(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        trainer = ParallelSelfPlayTrainer(
            policy_model=self._zeroed_model(encoder, action_space),
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=4, players_per_game=2),
            policy_config=PolicyConfig(seed=21, hidden_sizes=(64, 64), minibatch_size=8, ppo_epochs=1),
        )

        trainer._collect_worker_results(0)
        trainer.close()
        trainer.close()

        self.assertIsNone(trainer._worker_pool)

    def test_policy_config_defaults_to_available_device_and_larger_minibatch(self) -> None:
        config = PolicyConfig()

        self.assertEqual(("cuda" if __import__("torch").cuda.is_available() else "cpu"), config.device)
        self.assertEqual(512, config.minibatch_size)
        self.assertEqual("linear", config.learning_rate_schedule)
        self.assertEqual(5.0, config.advantage_clip)
        self.assertEqual("transformer", config.model_type)
        self.assertFalse(config.use_heuristic_bias)
        self.assertEqual(0.95, config.gae_lambda)

    def test_train_agent_parser_starts_fresh_by_default(self) -> None:
        args = train_agent.build_argument_parser().parse_args([])

        self.assertIsNone(args.resume)

    def test_environment_computes_gae_for_terminal_episode(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        environment = MonopolySelfPlayEnvironment(
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=4, players_per_game=2),
            reward_function=RewardFunction(),
            controller=AgentPolicyController(
                policy_model=self._zeroed_model(encoder, action_space),
                observation_encoder=encoder,
                action_space=action_space,
                heuristic_scorer=HeuristicScorer(),
            ),
            discount_gamma=0.9,
            gae_lambda=0.8,
        )

        returns, advantages = environment._compute_gae(
            rewards=[1.0, 2.0],
            state_values=[0.5, 1.0],
            next_state_values=[1.0, 0.0],
            terminal_flags=[False, True],
            bootstrap_final_transition=False,
        )

        self.assertAlmostEqual(2.62, returns[0], places=6)
        self.assertAlmostEqual(2.0, returns[1], places=6)
        self.assertAlmostEqual(2.12, advantages[0], places=6)
        self.assertAlmostEqual(1.0, advantages[1], places=6)

    def test_environment_bootstraps_gae_for_truncated_episode(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        environment = MonopolySelfPlayEnvironment(
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=4, players_per_game=2),
            reward_function=RewardFunction(),
            controller=AgentPolicyController(
                policy_model=self._zeroed_model(encoder, action_space),
                observation_encoder=encoder,
                action_space=action_space,
                heuristic_scorer=HeuristicScorer(),
            ),
            discount_gamma=0.9,
            gae_lambda=0.95,
            bootstrap_truncated_episodes=True,
        )

        returns, advantages = environment._compute_gae(
            rewards=[1.0],
            state_values=[0.5],
            next_state_values=[2.0],
            terminal_flags=[False],
            bootstrap_final_transition=True,
        )

        self.assertAlmostEqual(2.8, returns[0], places=6)
        self.assertAlmostEqual(2.3, advantages[0], places=6)

    def test_policy_model_iter_minibatches_covers_full_batch(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        model = self._zeroed_model(encoder, action_space)
        examples = [
            TrainingExample(
                observation=[0.0] * encoder.observation_size,
                action_mask=[True] * action_space.action_count,
                heuristic_bias=[0.0] * action_space.action_count,
                action_id=0,
                discounted_return=float(index),
                advantage=float(index),
                old_log_probability=0.0,
                threat_target=float(index) / 10.0,
            )
            for index in range(5)
        ]

        batch = model.batch_tensors(examples)
        minibatches = model.iter_minibatches(batch, minibatch_size=2, shuffle=False)

        self.assertEqual(3, len(minibatches))
        self.assertEqual([2, 2, 1], [int(minibatch["observations"].shape[0]) for minibatch in minibatches])
        self.assertEqual([2, 2, 1], [int(minibatch["threat_targets"].shape[0]) for minibatch in minibatches])

    def test_transformer_policy_model_can_act_and_round_trip_state(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        model = TorchPolicyModel(
            encoder.observation_size,
            action_space.action_count,
            seed=11,
            device="cpu",
            model_type="transformer",
            transformer_embedding_size=64,
            transformer_heads=4,
            transformer_layers=1,
            input_layout=encoder.observation_layout,
        )
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        observation = encoder.encode(game.get_frontend_state(), "A")
        action_mask, _ = action_space.build_mask(game.get_turn_plan("A"), game.get_frontend_state())

        decision = model.act(observation, action_mask, np.zeros(action_space.action_count, dtype=np.float32), explore=False, use_heuristic_bias=False)
        restored = TorchPolicyModel.from_state(model.to_state(), device_override="cpu")

        self.assertIsInstance(decision.action_id, int)
        self.assertEqual("transformer", restored.model_type)
        self.assertEqual(encoder.observation_layout, restored.input_layout)

    def test_league_manager_builds_mixed_rollout_plan(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        model = self._zeroed_model(encoder, action_space)
        training_config = TrainingConfig(
            worker_count=2,
            episodes_per_worker=2,
            players_per_game=3,
            use_league_self_play=True,
            league_snapshot_interval=1,
            league_recent_snapshot_count=2,
        )
        manager = LeagueManager(training_config, PolicyConfig(seed=17, hidden_sizes=(64, 64), minibatch_size=8, ppo_epochs=1))
        manager.record_snapshot(model.to_state(), 1)

        plan = manager.build_rollout_plan(model.to_state(), 2)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(2, len(plan.assignments_by_worker))
        self.assertIn("current", {spec.label for spec in plan.policy_specs})
        self.assertTrue(any(spec.source == "scripted" for spec in plan.policy_specs))
        self.assertIn("scripted", plan.source_weights)
        self.assertTrue(
            any(
                any(label != "current" for label in assignment.seat_labels)
                for worker_assignments in plan.assignments_by_worker
                for assignment in worker_assignments
            )
        )

    def test_league_manager_interpolates_weighted_sources(self) -> None:
        manager = LeagueManager(
            TrainingConfig(
                worker_count=1,
                episodes_per_worker=1,
                players_per_game=2,
                use_league_self_play=True,
            ),
            PolicyConfig(seed=17, hidden_sizes=(64, 64), minibatch_size=8, ppo_epochs=1),
        )

        self.assertEqual({"best": 0.0, "recent": 0.0, "scripted": 1.0}, manager.source_weights(0))
        self.assertEqual({"best": 0.2, "recent": 0.1, "scripted": 0.7}, manager.source_weights(20))
        self.assertEqual({"best": 0.4, "recent": 0.1, "scripted": 0.5}, manager.source_weights(50))
        self.assertEqual({"best": 0.25, "recent": 0.25, "scripted": 0.5}, manager.source_weights(150))
        self.assertEqual({"best": 0.4, "recent": 0.4, "scripted": 0.2}, manager.source_weights(250))

    def test_checkpoint_evaluator_supports_scripted_policy_specs(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        current_spec = LeaguePolicySpec(
            label="current",
            source="current",
            policy_state=self._zeroed_model(encoder, action_space).to_state(),
        )
        scripted_spec = LeaguePolicySpec(
            label="scripted_expansionist_builder",
            source="scripted",
            scripted_variant="expansionist_builder",
        )

        summary = CheckpointEvaluator(device="cpu").run_policy_benchmark(
            [current_spec, scripted_spec],
            seeds=[100],
            players_per_game=2,
            max_steps=8,
        )

        self.assertEqual(2, len(summary.participants))
        self.assertIn("current", summary.elo_ratings)
        self.assertIn("scripted_expansionist_builder", summary.elo_ratings)

    def test_scripted_controller_selects_legal_action(self) -> None:
        controller = build_scripted_controller("auction_value_shark", seed=13)
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])

        decision = controller.choose_action(game, "A", explore=False)

        legal_actions = {action.action_type for action in game.get_turn_plan("A").legal_actions}
        self.assertIn(decision.choice.legal_action.action_type, legal_actions)

    def test_scripted_controller_prefers_trade_that_completes_monopoly(self) -> None:
        controller = build_scripted_controller("expansionist_builder", seed=13)
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(5).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        frontend_state = game.get_frontend_state()
        action_mask, choices = MonopolyActionSpace().build_mask(game.get_turn_plan("A"), frontend_state)
        del action_mask
        analysis = analyze_board(frontend_state, "A")
        swap_choice = next(
            choice
            for choice in choices.values()
            if choice.trade_offer_payload is not None
            and choice.trade_offer_payload.get("proposer_property_names") == ["King's Cross Station"]
            and choice.trade_offer_payload.get("receiver_property_names") == ["Whitechapel Road"]
            and choice.trade_offer_payload.get("proposer_cash", 0) == 0
        )
        trade_choice = next(
            choice
            for choice in choices.values()
            if choice.trade_offer_payload is not None
            and choice.trade_offer_payload.get("receiver_property_names") == ["Whitechapel Road"]
            and choice.trade_offer_payload.get("proposer_cash", 0) > 0
        )

        trade_score = controller._score_choice(frontend_state, game.get_turn_plan("A"), analysis, "A", trade_choice)
        swap_score = controller._score_choice(frontend_state, game.get_turn_plan("A"), analysis, "A", swap_choice)

        self.assertGreater(trade_score, 0.0)
        self.assertGreater(trade_score, swap_score)

    def test_conservative_scripted_controller_rejects_liquidity_draining_offer(self) -> None:
        controller = build_scripted_controller("conservative_liquidity_manager", seed=13)
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        game.players[0].cash = 50
        game.board.get_space(6).assign_owner(game.players[0])
        game.board.get_space(8).assign_owner(game.players[1])
        game.current_player_index = 1
        propose_action = next(
            action
            for action in game.get_turn_plan("B").legal_actions
            if action.action_type == "propose_trade" and action.target_player_name == "A"
        )
        game.execute_legal_action(
            propose_action,
            trade_offer=game.deserialize_trade_offer(
                {
                    "proposer_name": "B",
                    "receiver_name": "A",
                    "proposer_cash": 0,
                    "receiver_cash": 50,
                    "proposer_property_names": ["Euston Road"],
                    "receiver_property_names": [],
                    "proposer_jail_cards": 0,
                    "receiver_jail_cards": 0,
                    "note": "too expensive",
                }
            ),
        )

        frontend_state = game.get_frontend_state()
        analysis = analyze_board(frontend_state, "A")
        accept_choice = next(choice for choice in MonopolyActionSpace().build_mask(game.get_turn_plan("A"), frontend_state)[1].values() if choice.action_label == "accept_trade")
        reject_choice = next(choice for choice in MonopolyActionSpace().build_mask(game.get_turn_plan("A"), frontend_state)[1].values() if choice.action_label == "reject_trade")

        accept_score = controller._score_choice(frontend_state, game.get_turn_plan("A"), analysis, "A", accept_choice)
        reject_score = controller._score_choice(frontend_state, game.get_turn_plan("A"), analysis, "A", reject_choice)

        self.assertGreater(reject_score, accept_score)

    def test_auction_value_shark_prefers_denial_trade_over_expansion_trade(self) -> None:
        controller = build_scripted_controller("auction_value_shark", seed=13)
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(6).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        game.board.get_space(8).assign_owner(game.players[1])
        frontend_state = game.get_frontend_state()
        analysis = analyze_board(frontend_state, "A")
        legal_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "propose_trade")

        denial_choice = AgentActionChoice(
            action_id=-3,
            action_label="trade_request_denial_cash_light:1",
            legal_action=legal_action,
            trade_offer_payload={
                "proposer_name": "A",
                "receiver_name": "B",
                "proposer_cash": 25,
                "receiver_cash": 0,
                "proposer_property_names": [],
                "receiver_property_names": ["Whitechapel Road"],
                "proposer_jail_cards": 0,
                "receiver_jail_cards": 0,
                "note": "synthetic",
            },
        )
        expansion_choice = AgentActionChoice(
            action_id=-4,
            action_label="trade_request_expansion_cash_light:1",
            legal_action=legal_action,
            trade_offer_payload={
                "proposer_name": "A",
                "receiver_name": "B",
                "proposer_cash": 25,
                "receiver_cash": 0,
                "proposer_property_names": [],
                "receiver_property_names": ["Euston Road"],
                "proposer_jail_cards": 0,
                "receiver_jail_cards": 0,
                "note": "synthetic",
            },
        )

        denial_score = controller._score_choice(frontend_state, game.get_turn_plan("A"), analysis, "A", denial_choice)
        expansion_score = controller._score_choice(frontend_state, game.get_turn_plan("A"), analysis, "A", expansion_choice)

        self.assertGreater(denial_score, expansion_score)

    def test_monopoly_denial_disruptor_prefers_denial_family_over_completion_family(self) -> None:
        controller = build_scripted_controller("monopoly_denial_disruptor", seed=13)
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        frontend_state = game.get_frontend_state()
        analysis = analyze_board(frontend_state, "A")
        legal_action = next(action for action in game.get_turn_plan("A").legal_actions if action.action_type == "propose_trade")
        payload = {
            "proposer_name": "A",
            "receiver_name": "B",
            "proposer_cash": 25,
            "receiver_cash": 0,
            "proposer_property_names": [],
            "receiver_property_names": ["Whitechapel Road"],
            "proposer_jail_cards": 0,
            "receiver_jail_cards": 0,
            "note": "synthetic",
        }

        denial_choice = AgentActionChoice(
            action_id=-5,
            action_label="trade_request_denial_cash_light:1",
            legal_action=legal_action,
            trade_offer_payload=dict(payload),
        )
        completion_choice = AgentActionChoice(
            action_id=-6,
            action_label="trade_request_completion_cash_light:1",
            legal_action=legal_action,
            trade_offer_payload=dict(payload),
        )

        denial_score = controller._score_choice(frontend_state, game.get_turn_plan("A"), analysis, "A", denial_choice)
        completion_score = controller._score_choice(frontend_state, game.get_turn_plan("A"), analysis, "A", completion_choice)

        self.assertGreater(denial_score, completion_score)

    def test_scripted_controller_prefers_mortgaging_isolated_property_first(self) -> None:
        controller = build_scripted_controller("conservative_liquidity_manager", seed=13)
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[0])
        game.board.get_space(5).assign_owner(game.players[0])
        game.players[0].cash = 0
        frontend_state = game.get_frontend_state()
        _, choices = MonopolyActionSpace().build_mask(game.get_turn_plan("A"), frontend_state)
        analysis = analyze_board(frontend_state, "A")
        isolated_mortgage = next(choice for choice in choices.values() if choice.action_label.startswith("mortgage:") and choice.legal_action.property_name == "King's Cross Station")
        monopoly_mortgage = next(choice for choice in choices.values() if choice.action_label.startswith("mortgage:") and choice.legal_action.property_name == "Old Kent Road")

        isolated_score = controller._score_choice(frontend_state, game.get_turn_plan("A"), analysis, "A", isolated_mortgage)
        monopoly_score = controller._score_choice(frontend_state, game.get_turn_plan("A"), analysis, "A", monopoly_mortgage)

        self.assertGreater(isolated_score, monopoly_score)

    def test_scripted_controller_prefers_counter_trade_over_accepting_bad_offer(self) -> None:
        controller = build_scripted_controller("expansionist_builder", seed=13)
        game = Game(["A", "B"], player_roles=[AI_ROLE, AI_ROLE])
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(5).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        game.current_player_index = 1

        propose_action = next(
            action
            for action in game.get_turn_plan("B").legal_actions
            if action.action_type == "propose_trade" and action.target_player_name == "A"
        )
        game.execute_legal_action(
            propose_action,
            trade_offer=game.deserialize_trade_offer(
                {
                    "proposer_name": "B",
                    "receiver_name": "A",
                    "proposer_cash": 25,
                    "receiver_cash": 0,
                    "proposer_property_names": [],
                    "receiver_property_names": ["King's Cross Station"],
                    "proposer_jail_cards": 0,
                    "receiver_jail_cards": 0,
                    "note": "bad opening",
                }
            ),
        )

        decision = controller.choose_action(game, "A", explore=False)

        self.assertEqual("counter_trade", decision.choice.legal_action.action_type)
        self.assertEqual(["Whitechapel Road"], decision.choice.trade_offer_payload["receiver_property_names"])

    def test_environment_league_mode_records_learning_seat_only(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        current_controller = AgentPolicyController(
            policy_model=self._zeroed_model(encoder, action_space),
            observation_encoder=encoder,
            action_space=action_space,
            heuristic_scorer=HeuristicScorer(),
        )
        opponent_controller = AgentPolicyController(
            policy_model=self._zeroed_model(encoder, action_space),
            observation_encoder=encoder,
            action_space=action_space,
            heuristic_scorer=HeuristicScorer(),
        )
        environment = MonopolySelfPlayEnvironment(
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=1, players_per_game=2),
            reward_function=RewardFunction(),
            controller=current_controller,
        )

        episode = environment.run_episode(
            seed=5,
            explore=False,
            controller_by_player={"AI1": current_controller, "AI2": opponent_controller},
            learning_player_names=("AI1",),
        )

        self.assertEqual(1, episode.step_count)
        self.assertEqual(1, len(episode.training_examples))

    def test_worker_pool_collects_mixed_rollout_with_scripted_seats(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        model = self._zeroed_model(encoder, action_space)
        training_config = TrainingConfig(
            worker_count=1,
            episodes_per_worker=1,
            max_steps_per_episode=3,
            max_actions_per_episode=8,
            players_per_game=3,
            use_league_self_play=True,
        )
        policy_config = PolicyConfig(
            seed=31,
            hidden_sizes=(64, 64),
            minibatch_size=8,
            ppo_epochs=1,
            device="cpu",
        )
        recent_spec = LeaguePolicySpec(
            label="recent_snapshot",
            source="recent",
            policy_state=model.to_state(),
        )
        scripted_spec = LeaguePolicySpec(
            label="scripted_auction_value_shark",
            source="scripted",
            scripted_variant="auction_value_shark",
        )
        pool = PersistentRolloutWorkerPool(
            initial_policy_state=model.to_state(),
            training_config=training_config,
            policy_config=policy_config,
            reward_weights=RewardWeights(),
            heuristic_weights=HeuristicWeights(),
        )

        try:
            results = pool.collect(
                model.to_state(),
                0,
                heuristic_scale=0.0,
                use_heuristic_bias=False,
                league_policy_specs=(
                    LeaguePolicySpec(label="current", source="current", policy_state=model.to_state()),
                    recent_spec,
                    scripted_spec,
                ),
                league_assignments_by_worker=(
                    (
                        LeagueEpisodeAssignment(
                            seat_labels=("current", "scripted_auction_value_shark", "recent_snapshot"),
                            learning_player_names=("AI1",),
                        ),
                    ),
                ),
            )
        finally:
            pool.close()

        self.assertEqual(1, len(results))
        self.assertEqual(1, len(results[0].episodes))
        episode = results[0].episodes[0]
        self.assertGreaterEqual(episode.step_count, 1)
        self.assertGreaterEqual(len(episode.training_examples), 1)

    def test_trainer_iteration_stats_include_effective_league_source_weights(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        trainer = ParallelSelfPlayTrainer(
            policy_model=self._zeroed_model(encoder, action_space),
            training_config=TrainingConfig(
                worker_count=1,
                episodes_per_worker=1,
                max_steps_per_episode=2,
                max_actions_per_episode=4,
                players_per_game=2,
                use_league_self_play=True,
            ),
            policy_config=PolicyConfig(seed=29, hidden_sizes=(64, 64), minibatch_size=8, ppo_epochs=1, device="cpu"),
        )

        try:
            stats = trainer.train(1)
        finally:
            trainer.close()

        self.assertEqual(1, len(stats))
        self.assertEqual({"scripted": 1.0}, stats[0].league_source_weights)

    def test_checkpoint_evaluator_builds_benchmark_suite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_a = self._write_checkpoint(temp_dir, seed=41)
            checkpoint_b = self._write_checkpoint(temp_dir, seed=42)
            summary = CheckpointEvaluator(device="cpu").run_benchmark_suite(
                [checkpoint_a, checkpoint_b],
                seeds=[100],
                players_per_game=2,
                max_steps=8,
            )

            self.assertEqual(2, len(summary.participants))
            self.assertIn(Path(checkpoint_a).stem, summary.elo_ratings)
            self.assertIn(Path(checkpoint_b).stem, summary.cross_play_win_rates[Path(checkpoint_a).stem])

    def test_trainer_normalizes_and_clips_advantages(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        trainer = ParallelSelfPlayTrainer(
            policy_model=self._zeroed_model(encoder, action_space),
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=4, players_per_game=2),
            policy_config=PolicyConfig(seed=23, hidden_sizes=(64, 64), minibatch_size=8, ppo_epochs=1, advantage_clip=0.5),
        )

        normalized = trainer._normalize_advantages(__import__("torch").tensor([0.0, 1.0, 100.0], dtype=__import__("torch").float32))

        self.assertTrue(bool((normalized.abs() <= 0.5 + 1e-6).all()))

    def test_trainer_applies_linear_learning_rate_schedule(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        trainer = ParallelSelfPlayTrainer(
            policy_model=self._zeroed_model(encoder, action_space),
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=4, players_per_game=2),
            policy_config=PolicyConfig(
                seed=25,
                hidden_sizes=(64, 64),
                minibatch_size=8,
                ppo_epochs=1,
                learning_rate=0.01,
                minimum_learning_rate=0.001,
                learning_rate_schedule="linear",
            ),
        )

        trainer._apply_learning_rate_schedule(0, 4)
        first_learning_rate = trainer._optimizer.param_groups[0]["lr"]
        trainer._apply_learning_rate_schedule(3, 4)
        final_learning_rate = trainer._optimizer.param_groups[0]["lr"]

        self.assertEqual(0.01, first_learning_rate)
        self.assertAlmostEqual(0.001, final_learning_rate)

    def test_detect_default_worker_count_leaves_headroom_for_cuda(self) -> None:
        original_cpu_count = train_agent.os.cpu_count
        try:
            train_agent.os.cpu_count = lambda: 16

            worker_count = train_agent._detect_default_worker_count("cuda")
        finally:
            train_agent.os.cpu_count = original_cpu_count

        self.assertEqual(8, worker_count)

    def test_detect_default_worker_count_leaves_headroom_for_cpu(self) -> None:
        original_cpu_count = train_agent.os.cpu_count
        try:
            train_agent.os.cpu_count = lambda: 16

            worker_count = train_agent._detect_default_worker_count("cpu")
        finally:
            train_agent.os.cpu_count = original_cpu_count

        self.assertEqual(12, worker_count)

    def test_detect_default_worker_count_caps_small_machines_safely(self) -> None:
        original_cpu_count = train_agent.os.cpu_count
        try:
            train_agent.os.cpu_count = lambda: 2

            worker_count = train_agent._detect_default_worker_count("cuda")
        finally:
            train_agent.os.cpu_count = original_cpu_count

        self.assertEqual(1, worker_count)

    def test_train_agent_can_resume_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir, seed=9)

            exit_code = train_agent_main([
                "--resume",
                str(checkpoint_path),
                "--iterations",
                "0",
                "--checkpoint-dir",
                temp_dir,
                "--threads",
                "1",
            ])

            self.assertEqual(0, exit_code)
            self.assertTrue((Path(temp_dir) / "latest.pt").exists())

    def test_parallel_trainer_resume_continues_checkpoint_numbering(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        trainer = ParallelSelfPlayTrainer(
            policy_model=self._zeroed_model(encoder, action_space),
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=6, players_per_game=2, checkpoint_interval=5),
            policy_config=PolicyConfig(seed=29, hidden_sizes=(64, 64), minibatch_size=8, ppo_epochs=1),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer.training_config.output_directory = temp_dir
            trainer.completed_iterations = 50
            resume_path = Path(temp_dir) / "latest.pt"
            trainer.save_checkpoint(resume_path)

            restored_trainer = ParallelSelfPlayTrainer.load_checkpoint(resume_path)
            restored_trainer.training_config.output_directory = temp_dir
            stats = restored_trainer.train(5)

            self.assertEqual(50, stats[0].iteration_index)
            self.assertTrue((Path(temp_dir) / "iteration_0055.pt").exists())
            self.assertEqual(55, restored_trainer.completed_iterations)

    def test_parallel_trainer_resume_infers_iteration_from_existing_latest_checkpoint(self) -> None:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        trainer = ParallelSelfPlayTrainer(
            policy_model=self._zeroed_model(encoder, action_space),
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=6, players_per_game=2, checkpoint_interval=5),
            policy_config=PolicyConfig(seed=31, hidden_sizes=(64, 64), minibatch_size=8, ppo_epochs=1),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            trainer.training_config.output_directory = temp_dir
            trainer.completed_iterations = 50
            trainer.save_checkpoint(checkpoint_dir / "iteration_0050.pt")

            trainer.completed_iterations = 0
            trainer.save_checkpoint(checkpoint_dir / "latest.pt")

            restored_trainer = ParallelSelfPlayTrainer.load_checkpoint(checkpoint_dir / "latest.pt")

            self.assertEqual(50, restored_trainer.completed_iterations)

    def test_evaluation_and_tournament_scripts_run_against_saved_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_a = self._write_checkpoint(temp_dir, seed=1)
            checkpoint_b = self._write_checkpoint(temp_dir, seed=2)

            evaluate_exit = evaluate_main([
                str(checkpoint_a),
                "--games",
                "1",
                "--players",
                "2",
                "--max-steps",
                "8",
            ])
            tournament_exit = tournament_main([
                str(checkpoint_a),
                str(checkpoint_b),
                "--games",
                "1",
                "--players",
                "2",
                "--max-steps",
                "8",
            ])
            benchmark_exit = evaluate_main([
                str(checkpoint_a),
                "--benchmark-opponents",
                str(checkpoint_b),
                "--games",
                "1",
                "--players",
                "2",
                "--max-steps",
                "8",
            ])

            self.assertEqual(0, evaluate_exit)
            self.assertEqual(0, tournament_exit)
            self.assertEqual(0, benchmark_exit)

    def test_load_agent_host_rejects_incompatible_checkpoint_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            incompatible_path = Path(temp_dir) / "incompatible.pt"
            action_space = MonopolyActionSpace()
            smaller_model = TorchPolicyModel(ObservationEncoder().observation_size - 1, action_space.action_count, seed=33, hidden_sizes=(64, 64))
            trainer = ParallelSelfPlayTrainer(
                policy_model=smaller_model,
                training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=4, players_per_game=2),
                policy_config=PolicyConfig(seed=33, hidden_sizes=(64, 64), ppo_epochs=1, minibatch_size=8),
            )
            trainer.save_checkpoint(incompatible_path)

            with self.assertRaisesRegex(ValueError, "observation size"):
                load_agent_host_from_checkpoint(incompatible_path)


if __name__ == "__main__":
    unittest.main()