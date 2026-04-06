from __future__ import annotations

"""Discrete action encoding for the RL agent stack.

This module maps legal game actions into a fixed action-id space and expands
macro choices such as auctions and trade proposals into concrete candidate
payloads that can be scored or sampled by policies.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from monopoly.agent.board_analysis import analyze_board, strongest_opponent_name
from monopoly.api import FrontendStateView, LegalActionOption, TurnPlanView
from monopoly.constants import MAX_PLAYERS
from monopoly.board import create_standard_board


_GENERIC_ACTION_LABELS = (
    "start_turn",
    "end_turn",
    "buy_property",
    "decline_property",
    "pass_auction",
    "auction_bid_min",
    "auction_bid_step",
    "auction_bid_value",
    "auction_bid_reserve",
    "auction_bid_premium",
    "auction_bid_aggressive",
    "auction_bid_denial",
    "jail_use_card",
    "jail_pay_fine",
    "jail_roll",
    "confirm_property_action",
    "cancel_property_action",
    "accept_trade",
    "reject_trade",
)

_PROPERTY_ACTION_PREFIXES = (
    ("request_build", "build"),
    ("request_sell_building", "sell"),
    ("request_mortgage", "mortgage"),
    ("request_unmortgage", "unmortgage"),
)

_TRADE_TEMPLATE_LABELS = (
    "trade_request_completion_cash_light",
    "trade_request_completion_cash_heavy",
    "trade_request_completion_property_cash",
    "trade_request_expansion_cash_light",
    "trade_request_expansion_cash_heavy",
    "trade_request_expansion_property_cash",
    "trade_request_denial_cash_light",
    "trade_request_denial_cash_heavy",
    "trade_request_denial_property_cash",
    "trade_swap_low_for_best_fit",
    "trade_swap_low_plus_cash_for_best_fit",
    "trade_sell_isolated_for_cash",
    "trade_sell_mortgaged_for_cash",
)


@dataclass(frozen=True, slots=True)
class AgentActionChoice:
    """One concrete encodable action candidate for the acting player."""

    action_id: int
    action_label: str
    legal_action: LegalActionOption
    bid_amount: int | None = None
    trade_offer_payload: dict[str, Any] | None = None


class MonopolyActionSpace:
    """Translate legal engine actions into a stable discrete policy space."""

    def __init__(self) -> None:
        self._board = create_standard_board()
        self._generic_action_ids = {label: index for index, label in enumerate(_GENERIC_ACTION_LABELS)}
        self._action_labels_by_id = {index: label for label, index in self._generic_action_ids.items()}
        self._property_name_to_index = {space.name: space.index for space in self._board.spaces}
        self._property_name_to_price = {space.name: int(getattr(space, "price", 0) or 0) for space in self._board.spaces}
        self._property_action_labels_by_prefix = {legal_prefix: property_label for legal_prefix, property_label in _PROPERTY_ACTION_PREFIXES}
        offset = len(_GENERIC_ACTION_LABELS)
        self._trade_action_ids: dict[tuple[int, str], int] = {}
        for target_slot in range(1, MAX_PLAYERS):
            for template_index, template_label in enumerate(_TRADE_TEMPLATE_LABELS):
                action_id = offset + (target_slot - 1) * len(_TRADE_TEMPLATE_LABELS) + template_index
                action_label = f"{template_label}:{target_slot}"
                self._trade_action_ids[(target_slot, template_label)] = action_id
                self._action_labels_by_id[action_id] = action_label
        offset += (MAX_PLAYERS - 1) * len(_TRADE_TEMPLATE_LABELS)
        self._property_action_ids: dict[tuple[str, int], int] = {}
        for prefix_index, (_, property_label) in enumerate(_PROPERTY_ACTION_PREFIXES):
            for space in self._board.spaces:
                action_id = offset + prefix_index * len(self._board.spaces) + space.index
                action_label = f"{property_label}:{space.index}"
                self._property_action_ids[(property_label, space.index)] = action_id
                self._action_labels_by_id[action_id] = action_label
        self._action_count = offset + len(_PROPERTY_ACTION_PREFIXES) * len(self._board.spaces)

    @property
    def action_count(self) -> int:
        """Total number of action ids exposed to the policy model."""
        return self._action_count

    def action_label(self, action_id: int) -> str:
        """Return the human-readable label associated with an action id."""
        try:
            return self._action_labels_by_id[action_id]
        except KeyError as error:
            raise KeyError(f"Unknown action id: {action_id}") from error

    def _expand_legal_action(self, legal_action: LegalActionOption, frontend_state: FrontendStateView | None = None) -> list[AgentActionChoice]:
        """Expand one legal action into one or more encodable policy choices."""
        action_type = legal_action.action_type
        if action_type in self._generic_action_ids:
            action_id = self._generic_action_ids[action_type]
            return [AgentActionChoice(action_id=action_id, action_label=action_type, legal_action=legal_action)]

        if action_type == "place_auction_bid":
            if frontend_state is None:
                return []
            return self._expand_auction_bid_choices(legal_action, frontend_state)

        if action_type == "propose_trade":
            if frontend_state is None:
                return []
            return self._expand_trade_proposal_choices(legal_action, frontend_state)

        if action_type == "counter_trade":
            if frontend_state is None:
                return []
            return self._expand_trade_proposal_choices(legal_action, frontend_state)

        property_label = self._property_action_labels_by_prefix.get(action_type)
        if property_label is not None and legal_action.property_name is not None:
            space_index = self._property_name_to_index[legal_action.property_name]
            action_id = self._property_action_ids[(property_label, space_index)]
            return [AgentActionChoice(action_id=action_id, action_label=self._action_labels_by_id[action_id], legal_action=legal_action)]

        return []

    def build_mask(self, turn_plan: TurnPlanView, frontend_state: FrontendStateView) -> tuple[np.ndarray, dict[int, AgentActionChoice]]:
        """Build a boolean action mask and lookup table for the current turn plan."""
        mask = np.zeros(self._action_count, dtype=bool)
        choices: dict[int, AgentActionChoice] = {}

        for legal_action in turn_plan.legal_actions:
            for choice in self._expand_legal_action(legal_action, frontend_state):
                mask[choice.action_id] = True
                choices[choice.action_id] = choice

        if not choices:
            raise ValueError(f"No encodable legal actions were found for {turn_plan.player_name}.")
        return mask, choices

    def _expand_auction_bid_choices(self, legal_action: LegalActionOption, frontend_state: FrontendStateView) -> list[AgentActionChoice]:
        """Generate a small set of auction bid buckets from board-aware anchors."""
        minimum_bid = 1 if legal_action.min_bid is None else legal_action.min_bid
        maximum_bid = minimum_bid if legal_action.max_bid is None else legal_action.max_bid
        if maximum_bid < minimum_bid:
            return []
        actor = next((player for player in frontend_state.game_view.players if player.name == legal_action.actor_name), None)
        if actor is None:
            return []
        property_price = self._property_name_to_price.get(legal_action.property_name or "", minimum_bid)
        pending_action = frontend_state.game_view.pending_action
        current_bid = 0 if pending_action is None or pending_action.auction is None else pending_action.auction.current_bid
        property_index = self._property_name_to_index.get(legal_action.property_name or "", -1)
        property_space = None if property_index < 0 else frontend_state.board_spaces[property_index]
        analysis = analyze_board(frontend_state, legal_action.actor_name)
        actor_metrics = analysis.player_metrics[legal_action.actor_name]
        strongest_opponent_name_value = strongest_opponent_name(analysis, legal_action.actor_name)
        strongest_opponent_strength = 0.0 if strongest_opponent_name_value is None else analysis.player_metrics[strongest_opponent_name_value].board_strength
        actor_color_progress, opponent_color_progress = self._auction_color_progress(frontend_state, legal_action.actor_name, None if property_space is None else property_space.color_group)
        progress_pressure = 1.0 + 0.25 * max(actor_color_progress, opponent_color_progress)
        step_size = max(5, int(round(property_price * (0.10 + 0.08 * progress_pressure))))
        reserve_buffer = max(50, int(round(actor.cash * 0.2)))
        reserve_anchor = max(minimum_bid, min(maximum_bid, max(current_bid + step_size, min(actor.cash - reserve_buffer, int(round(property_price * (0.85 + 0.25 * actor_color_progress)))))))
        denial_multiplier = 1.55 + 0.35 * opponent_color_progress + (0.10 if strongest_opponent_strength > actor_metrics.board_strength else 0.0)
        denial_anchor = max(minimum_bid, int(round(property_price * denial_multiplier)))
        bucket_values = {
            "auction_bid_min": minimum_bid,
            "auction_bid_step": min(maximum_bid, max(minimum_bid, current_bid + step_size)),
            "auction_bid_value": min(maximum_bid, max(minimum_bid, property_price)),
            "auction_bid_reserve": min(maximum_bid, reserve_anchor),
            "auction_bid_premium": min(maximum_bid, max(minimum_bid, int(round(property_price * 1.2)))),
            "auction_bid_aggressive": min(maximum_bid, max(minimum_bid, int(round(property_price * (1.45 + 0.15 * actor_color_progress))))),
            "auction_bid_denial": min(maximum_bid, denial_anchor),
        }
        seen_bid_amounts: set[int] = set()
        choices: list[AgentActionChoice] = []
        for label, bid_amount in bucket_values.items():
            if bid_amount in seen_bid_amounts:
                continue
            seen_bid_amounts.add(bid_amount)
            choices.append(
                AgentActionChoice(
                    action_id=self._generic_action_ids[label],
                    action_label=label,
                    legal_action=legal_action,
                    bid_amount=bid_amount,
                )
            )
        return choices

    def _expand_trade_proposal_choices(self, legal_action: LegalActionOption, frontend_state: FrontendStateView) -> list[AgentActionChoice]:
        """Build trade-template candidates for a legal trade or counter-trade action."""
        actor_name = legal_action.actor_name
        target_name = legal_action.target_player_name
        if target_name is None:
            return []
        actor = next((player for player in frontend_state.game_view.players if player.name == actor_name), None)
        target = next((player for player in frontend_state.game_view.players if player.name == target_name), None)
        if actor is None or target is None:
            return []
        target_slot = self._target_slot(frontend_state, actor_name, target_name)
        if target_slot is None:
            return []
        analysis = analyze_board(frontend_state, actor_name)
        blocked_signatures = set(frontend_state.game_view.blocked_trade_offer_signatures)
        actor_property_names = self._owned_property_names(frontend_state, actor_name)
        target_property_names = self._owned_property_names(frontend_state, target_name)
        if not actor_property_names and not target_property_names:
            return []

        actor_low_property = self._lowest_value_tradeable_property(frontend_state, analysis, actor_name, actor_property_names)
        actor_mortgaged_property = self._best_mortgaged_tradeable_property(frontend_state, analysis, actor_name, actor_property_names)
        actor_isolated_property = self._best_isolated_tradeable_property(frontend_state, analysis, actor_name, actor_property_names)
        completion_target = self._best_completion_target(frontend_state, analysis, actor_name, target_property_names)
        expansion_target = self._best_expansion_target(frontend_state, analysis, actor_name, target_property_names)
        denial_target = self._best_denial_target(frontend_state, analysis, actor_name, target_name, target_property_names)
        best_fit_target = self._best_fit_target(frontend_state, analysis, actor_name, target_property_names)

        candidate_templates = (
            ("trade_request_completion_cash_light", self._build_cash_for_property_payload(actor_name, target_name, completion_target, actor.cash, 0.35)),
            ("trade_request_completion_cash_heavy", self._build_cash_for_property_payload(actor_name, target_name, completion_target, actor.cash, 0.55)),
            (
                "trade_request_completion_property_cash",
                self._build_property_plus_cash_for_property_payload(actor_name, target_name, actor_low_property, completion_target, actor.cash, 0.30),
            ),
            ("trade_request_expansion_cash_light", self._build_cash_for_property_payload(actor_name, target_name, expansion_target, actor.cash, 0.30)),
            ("trade_request_expansion_cash_heavy", self._build_cash_for_property_payload(actor_name, target_name, expansion_target, actor.cash, 0.45)),
            (
                "trade_request_expansion_property_cash",
                self._build_property_plus_cash_for_property_payload(actor_name, target_name, actor_low_property, expansion_target, actor.cash, 0.20),
            ),
            ("trade_request_denial_cash_light", self._build_cash_for_property_payload(actor_name, target_name, denial_target, actor.cash, 0.30)),
            ("trade_request_denial_cash_heavy", self._build_cash_for_property_payload(actor_name, target_name, denial_target, actor.cash, 0.50)),
            (
                "trade_request_denial_property_cash",
                self._build_property_plus_cash_for_property_payload(actor_name, target_name, actor_low_property, denial_target, actor.cash, 0.25),
            ),
            ("trade_swap_low_for_best_fit", self._build_property_swap_payload(actor_name, target_name, actor_low_property, best_fit_target)),
            (
                "trade_swap_low_plus_cash_for_best_fit",
                self._build_property_plus_cash_for_property_payload(actor_name, target_name, actor_low_property, best_fit_target, actor.cash, 0.20),
            ),
            ("trade_sell_isolated_for_cash", self._build_property_for_cash_payload(actor_name, target_name, actor_isolated_property, target.cash, 0.45)),
            ("trade_sell_mortgaged_for_cash", self._build_property_for_cash_payload(actor_name, target_name, actor_mortgaged_property, target.cash, 0.35)),
        )

        choices: list[AgentActionChoice] = []
        seen_signatures: set[str] = set()
        for template_label, payload in candidate_templates:
            if payload is None:
                continue
            signature = self._trade_payload_signature(payload)
            if signature in blocked_signatures or signature in seen_signatures:
                continue
            if not self._is_two_sided_trade_payload(payload):
                continue
            if self._estimate_trade_payload_score(frontend_state, analysis, actor_name, payload) <= 0.0:
                continue
            action_id = self._trade_action_ids.get((target_slot, template_label))
            if action_id is None:
                continue
            seen_signatures.add(signature)
            choices.append(
                AgentActionChoice(
                    action_id=action_id,
                    action_label=self._action_labels_by_id[action_id],
                    legal_action=legal_action,
                    trade_offer_payload=payload,
                )
            )
        return choices

    @staticmethod
    def _trade_payload(
        proposer_name: str,
        receiver_name: str,
        *,
        proposer_cash: int = 0,
        receiver_cash: int = 0,
        proposer_property_names: list[str] | None = None,
        receiver_property_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create the serialized trade payload used by engine action execution."""
        return {
            "proposer_name": proposer_name,
            "receiver_name": receiver_name,
            "proposer_cash": proposer_cash,
            "receiver_cash": receiver_cash,
            "proposer_property_names": [] if proposer_property_names is None else list(proposer_property_names),
            "receiver_property_names": [] if receiver_property_names is None else list(receiver_property_names),
            "proposer_jail_cards": 0,
            "receiver_jail_cards": 0,
            "note": "rl-template",
        }

    @staticmethod
    def _trade_payload_signature(payload: dict[str, Any]) -> str:
        """Create a deterministic signature for blocked and deduplicated trade payloads."""
        proposer_properties = sorted(str(name) for name in payload.get("proposer_property_names", []))
        receiver_properties = sorted(str(name) for name in payload.get("receiver_property_names", []))
        return "|".join(
            (
                str(payload.get("proposer_name", "")),
                str(payload.get("receiver_name", "")),
                str(int(payload.get("proposer_cash", 0))),
                str(int(payload.get("receiver_cash", 0))),
                ",".join(proposer_properties),
                ",".join(receiver_properties),
                str(int(payload.get("proposer_jail_cards", 0))),
                str(int(payload.get("receiver_jail_cards", 0))),
            )
        )

    def _owned_property_names(self, frontend_state: FrontendStateView, owner_name: str) -> list[str]:
        """List tradeable owned property names for one player in the current view."""
        return [space.name for space in frontend_state.board_spaces if space.owner_name == owner_name and space.price is not None]

    def _space_by_name(self, frontend_state: FrontendStateView, property_name: str | None):
        """Return a board-space view by property name if present."""
        if property_name is None:
            return None
        return next((space for space in frontend_state.board_spaces if space.name == property_name), None)

    def _group_progress(self, frontend_state: FrontendStateView, owner_name: str, color_group: str | None) -> float:
        """Measure how much of a color group a player currently owns."""
        if color_group is None:
            return 0.0
        group_spaces = [space for space in frontend_state.board_spaces if space.color_group == color_group]
        if not group_spaces:
            return 0.0
        owned_count = sum(1 for space in group_spaces if space.owner_name == owner_name)
        return owned_count / float(len(group_spaces))

    def _progress_with_transfer(self, frontend_state: FrontendStateView, owner_name: str, property_name: str, *, gaining: bool) -> float:
        """Estimate post-trade group progress after gaining or losing one property."""
        space = self._space_by_name(frontend_state, property_name)
        if space is None or space.color_group is None:
            return 0.0
        current = self._group_progress(frontend_state, owner_name, space.color_group)
        group_spaces = [candidate for candidate in frontend_state.board_spaces if candidate.color_group == space.color_group]
        if not group_spaces:
            return current
        step = 1.0 / float(len(group_spaces))
        next_progress = current + step if gaining else current - step
        return max(0.0, min(1.0, next_progress))

    def _estimate_property_trade_value(self, frontend_state: FrontendStateView, analysis, actor_name: str, property_name: str) -> float:
        """Score how strategically valuable a property is in a potential trade."""
        space = self._space_by_name(frontend_state, property_name)
        if space is None:
            return 0.0
        strongest = strongest_opponent_name(analysis, actor_name)
        pressure = analysis.estimated_space_pressure_by_index.get(space.index, 0.0)
        actor_progress = self._group_progress(frontend_state, actor_name, space.color_group)
        opponent_progress = 0.0 if strongest is None else self._group_progress(frontend_state, strongest, space.color_group)
        completion_bonus = 140.0 if self._progress_with_transfer(frontend_state, actor_name, property_name, gaining=True) >= 1.0 > actor_progress else 0.0
        denial_bonus = 80.0 if strongest is not None and self._progress_with_transfer(frontend_state, strongest, property_name, gaining=True) >= 1.0 > opponent_progress else 0.0
        mortgage_penalty = 20.0 if space.mortgaged else 0.0
        building_bonus = float(space.house_cost or 0) * float(space.building_count or 0) * 0.35
        return pressure + float(space.price or 0) * 0.28 + actor_progress * 60.0 + opponent_progress * 35.0 + completion_bonus + denial_bonus + building_bonus - mortgage_penalty

    def _lowest_value_tradeable_property(self, frontend_state: FrontendStateView, analysis, actor_name: str, property_names: list[str]) -> str | None:
        """Pick the least costly property for the actor to give away."""
        candidates = [
            property_name
            for property_name in property_names
            if not self._giving_property_breaks_monopoly(frontend_state, actor_name, property_name)
        ]
        if not candidates:
            candidates = list(property_names)
        if not candidates:
            return None
        return min(candidates, key=lambda property_name: self._estimate_property_trade_value(frontend_state, analysis, actor_name, property_name))

    def _best_mortgaged_tradeable_property(self, frontend_state: FrontendStateView, analysis, actor_name: str, property_names: list[str]) -> str | None:
        """Choose the weakest mortgaged property that can be offered in trade."""
        mortgaged = [
            property_name
            for property_name in property_names
            if (space := self._space_by_name(frontend_state, property_name)) is not None and space.mortgaged
        ]
        if not mortgaged:
            return None
        return min(mortgaged, key=lambda property_name: self._estimate_property_trade_value(frontend_state, analysis, actor_name, property_name))

    def _best_isolated_tradeable_property(self, frontend_state: FrontendStateView, analysis, actor_name: str, property_names: list[str]) -> str | None:
        """Choose an expendable property from a weak or isolated color position."""
        isolated = []
        for property_name in property_names:
            if self._giving_property_breaks_monopoly(frontend_state, actor_name, property_name):
                continue
            space = self._space_by_name(frontend_state, property_name)
            if space is None:
                continue
            if self._group_progress(frontend_state, actor_name, space.color_group) <= 0.5:
                isolated.append(property_name)
        if not isolated:
            return None
        return min(isolated, key=lambda property_name: self._estimate_property_trade_value(frontend_state, analysis, actor_name, property_name))

    def _best_completion_target(self, frontend_state: FrontendStateView, analysis, actor_name: str, property_names: list[str]) -> str | None:
        """Choose the property that best completes an actor monopoly."""
        candidates = [
            property_name
            for property_name in property_names
            if self._completion_swing(frontend_state, actor_name, property_name, gaining=True) > 0.0
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda property_name: (self._completion_swing(frontend_state, actor_name, property_name, gaining=True), self._estimate_property_trade_value(frontend_state, analysis, actor_name, property_name)))

    def _best_expansion_target(self, frontend_state: FrontendStateView, analysis, actor_name: str, property_names: list[str]) -> str | None:
        """Choose the most attractive property that grows an existing set without completing it."""
        candidates = []
        for property_name in property_names:
            space = self._space_by_name(frontend_state, property_name)
            if space is None or space.color_group is None:
                continue
            current_progress = self._group_progress(frontend_state, actor_name, space.color_group)
            next_progress = self._progress_with_transfer(frontend_state, actor_name, property_name, gaining=True)
            if current_progress <= 0.0:
                continue
            if next_progress >= 1.0:
                continue
            if next_progress <= current_progress:
                continue
            candidates.append(property_name)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda property_name: (
                self._group_progress(frontend_state, actor_name, self._space_by_name(frontend_state, property_name).color_group),
                self._estimate_property_trade_value(frontend_state, analysis, actor_name, property_name),
            ),
        )

    def _best_denial_target(self, frontend_state: FrontendStateView, analysis, actor_name: str, target_name: str, property_names: list[str]) -> str | None:
        """Choose the property that most disrupts an opponent's likely monopoly."""
        candidates = [
            property_name
            for property_name in property_names
            if self._completion_swing(frontend_state, target_name, property_name, gaining=False) > 0.0
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda property_name: (self._completion_swing(frontend_state, target_name, property_name, gaining=False), self._estimate_property_trade_value(frontend_state, analysis, actor_name, property_name)))

    def _best_fit_target(self, frontend_state: FrontendStateView, analysis, actor_name: str, property_names: list[str]) -> str | None:
        """Choose the highest-value target property among available candidates."""
        if not property_names:
            return None
        return max(property_names, key=lambda property_name: self._estimate_property_trade_value(frontend_state, analysis, actor_name, property_name))

    def _completion_swing(self, frontend_state: FrontendStateView, owner_name: str, property_name: str, *, gaining: bool) -> float:
        """Quantify how much a transfer changes monopoly completion pressure."""
        space = self._space_by_name(frontend_state, property_name)
        if space is None or space.color_group is None:
            return 0.0
        current = self._group_progress(frontend_state, owner_name, space.color_group)
        next_progress = self._progress_with_transfer(frontend_state, owner_name, property_name, gaining=gaining)
        if gaining and current < 1.0 <= next_progress:
            return 2.0
        if gaining and current < 0.75 <= next_progress:
            return 1.0
        if not gaining and current >= 1.0 > next_progress:
            return 2.2
        if not gaining and current >= 0.75 > next_progress:
            return 1.1
        return max(0.0, next_progress - current) if gaining else max(0.0, current - next_progress)

    def _giving_property_breaks_monopoly(self, frontend_state: FrontendStateView, owner_name: str, property_name: str) -> bool:
        """Whether surrendering a property would break a completed monopoly."""
        return self._completion_swing(frontend_state, owner_name, property_name, gaining=False) >= 2.0

    @staticmethod
    def _bounded_cash_amount(available_cash: int, reference_value: float, factor: float) -> int:
        """Convert a relative pricing factor into a spendable rounded cash amount."""
        reserve = max(100, int(round(available_cash * 0.2)))
        spendable = max(0, available_cash - reserve)
        if spendable < 25:
            return 0
        scaled = int(round(reference_value * factor / 25.0) * 25)
        return max(25, min(spendable, scaled))

    def _build_cash_for_property_payload(self, proposer_name: str, receiver_name: str, requested_property: str | None, available_cash: int, factor: float) -> dict[str, Any] | None:
        """Build a cash-for-property trade template."""
        if requested_property is None:
            return None
        reference_value = float(self._property_name_to_price.get(requested_property, 0))
        cash_amount = self._bounded_cash_amount(available_cash, reference_value, factor)
        if cash_amount <= 0:
            return None
        return self._trade_payload(
            proposer_name,
            receiver_name,
            proposer_cash=cash_amount,
            receiver_property_names=[requested_property],
        )

    def _build_property_plus_cash_for_property_payload(
        self,
        proposer_name: str,
        receiver_name: str,
        offered_property: str | None,
        requested_property: str | None,
        available_cash: int,
        factor: float,
    ) -> dict[str, Any] | None:
        """Build a property-plus-cash trade template for a more valuable target."""
        if offered_property is None or requested_property is None or offered_property == requested_property:
            return None
        value_gap = max(0.0, float(self._property_name_to_price.get(requested_property, 0)) - float(self._property_name_to_price.get(offered_property, 0)))
        cash_amount = self._bounded_cash_amount(available_cash, max(25.0, value_gap), factor)
        return self._trade_payload(
            proposer_name,
            receiver_name,
            proposer_cash=cash_amount,
            proposer_property_names=[offered_property],
            receiver_property_names=[requested_property],
        )

    def _build_property_swap_payload(self, proposer_name: str, receiver_name: str, offered_property: str | None, requested_property: str | None) -> dict[str, Any] | None:
        """Build a straight property-for-property swap template."""
        if offered_property is None or requested_property is None or offered_property == requested_property:
            return None
        return self._trade_payload(
            proposer_name,
            receiver_name,
            proposer_property_names=[offered_property],
            receiver_property_names=[requested_property],
        )

    def _build_property_for_cash_payload(self, proposer_name: str, receiver_name: str, offered_property: str | None, available_cash: int, factor: float) -> dict[str, Any] | None:
        """Build a template that sells one property to the counterpart for cash."""
        if offered_property is None:
            return None
        reference_value = float(self._property_name_to_price.get(offered_property, 0))
        cash_amount = self._bounded_cash_amount(available_cash, reference_value, factor)
        if cash_amount <= 0:
            return None
        return self._trade_payload(
            proposer_name,
            receiver_name,
            receiver_cash=cash_amount,
            proposer_property_names=[offered_property],
        )

    @staticmethod
    def _is_two_sided_trade_payload(payload: dict[str, Any]) -> bool:
        """Whether both sides contribute something meaningful to a trade payload."""
        proposer_gives = int(payload.get("proposer_cash", 0)) > 0 or bool(payload.get("proposer_property_names")) or int(payload.get("proposer_jail_cards", 0)) > 0
        receiver_gives = int(payload.get("receiver_cash", 0)) > 0 or bool(payload.get("receiver_property_names")) or int(payload.get("receiver_jail_cards", 0)) > 0
        return proposer_gives and receiver_gives

    def _estimate_trade_payload_score(self, frontend_state: FrontendStateView, analysis, actor_name: str, payload: dict[str, Any]) -> float:
        """Estimate whether a generated trade template is worth keeping as a choice."""
        if not self._is_two_sided_trade_payload(payload):
            return -1.0
        receive_properties = tuple(str(name) for name in payload.get("receiver_property_names", []))
        give_properties = tuple(str(name) for name in payload.get("proposer_property_names", []))
        receive_value = sum(self._estimate_property_trade_value(frontend_state, analysis, actor_name, property_name) for property_name in receive_properties)
        give_value = sum(self._estimate_property_trade_value(frontend_state, analysis, actor_name, property_name) for property_name in give_properties)
        completion_bonus = sum(self._completion_swing(frontend_state, actor_name, property_name, gaining=True) * 85.0 for property_name in receive_properties)
        completion_penalty = sum(self._completion_swing(frontend_state, actor_name, property_name, gaining=False) * 95.0 for property_name in give_properties)
        strongest = strongest_opponent_name(analysis, actor_name)
        opponent_penalty = 0.0 if strongest is None else sum(self._completion_swing(frontend_state, strongest, property_name, gaining=True) * 90.0 for property_name in give_properties)
        cash_net = float(payload.get("receiver_cash", 0)) - float(payload.get("proposer_cash", 0))
        actor = next((player for player in frontend_state.game_view.players if player.name == actor_name), None)
        liquidity_penalty = 0.0
        if actor is not None:
            reserve = max(100.0, float(actor.cash) * 0.2)
            liquidity_penalty = max(0.0, reserve - (float(actor.cash) - float(payload.get("proposer_cash", 0)))) * 0.4
        return cash_net + receive_value - give_value + completion_bonus - completion_penalty - opponent_penalty - liquidity_penalty

    @staticmethod
    def _target_slot(frontend_state: FrontendStateView, actor_name: str, target_name: str) -> int | None:
        """Map a target player into the actor-relative trade slot numbering."""
        players = list(frontend_state.game_view.players)
        actor_index = next((index for index, player in enumerate(players) if player.name == actor_name), None)
        if actor_index is None:
            return None
        ordered_players = players[actor_index + 1:] + players[:actor_index]
        for slot, player in enumerate(ordered_players, start=1):
            if player.name == target_name:
                return slot
        return None

    @staticmethod
    def _auction_color_progress(frontend_state: FrontendStateView, actor_name: str, color_group: str | None) -> tuple[float, float]:
        """Return actor and strongest-opponent ownership shares within a color group."""
        if color_group is None:
            return 0.0, 0.0
        group_spaces = [space for space in frontend_state.board_spaces if space.color_group == color_group]
        if not group_spaces:
            return 0.0, 0.0
        group_size = float(len(group_spaces))
        actor_count = sum(1 for space in group_spaces if space.owner_name == actor_name)
        opponent_max = max(
            (
                sum(1 for space in group_spaces if space.owner_name == player.name)
                for player in frontend_state.game_view.players
                if player.name != actor_name
            ),
            default=0,
        )
        return actor_count / group_size, opponent_max / group_size