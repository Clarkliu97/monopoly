from __future__ import annotations

"""Main Monopoly game state machine.

This module is the core orchestrator for the project. It owns turn order,
interactive phase transitions, pending decisions, auctions, jail flow, trading,
bankruptcy, and serialization. Frontend code, online hosting, and RL training
all rely on this class as the authoritative source of game behaviour.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from monopoly.api import (
    AuctionView,
    BoardSpaceView,
    FrontendStateView,
    GameView,
    GameSetup,
    InteractionResult,
    JailDecisionView,
    LegalActionOption,
    PendingActionView,
    PlayerView,
    PropertyActionView,
    TurnPlanView,
    TradeDecisionView,
)
from monopoly.board import Board, create_standard_board
from monopoly.constants import AI_ROLE, BANK_HOTELS, BANK_HOUSES, BOARD_SIZE, GO_SALARY, HUMAN_ROLE, IN_TURN_PHASE, JAIL_FINE, JAIL_INDEX, MAX_PLAYERS, MIN_PLAYERS, PLAYER_ROLES, POST_ROLL_PHASE, PRE_ROLL_PHASE, STARTING_CASH
from monopoly.dice import Dice, DiceRoll
from monopoly.player import Player
from monopoly.rules import MonopolyRules
from monopoly.spaces import ActionSpace, CardSpace, PropertySpace, RailroadPropertySpace, StreetPropertySpace, TaxSpace, UtilityPropertySpace
from monopoly.trading import TradeOffer


@dataclass(slots=True)
class TurnReport:
    """Messages emitted while resolving one player's turn."""

    player_name: str
    messages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AuctionResult:
    """Resolved outcome of an auction, including winner and narration."""

    winner: Player | None
    winning_bid: int = 0
    messages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class JailTurnResult:
    """Result of a jail attempt and whether it consumed the turn."""

    messages: list[str]
    turn_consumed: bool


@dataclass(slots=True)
class PurchaseDecisionState:
    """Pending state for a landed-but-unowned property purchase choice."""

    player_name: str
    property_index: int
    property_name: str
    price: int


@dataclass(slots=True)
class PendingAuctionState:
    """Pending state for a live auction while players are still bidding."""

    property_index: int
    property_name: str
    eligible_player_names: list[str]
    active_player_names: list[str]
    current_bid: int = 0
    current_winner_name: str | None = None
    current_bidder_index: int = 0


@dataclass(slots=True)
class TurnContinuation:
    """Bookkeeping for extra rolls caused by doubles."""

    player_name: str
    doubles_in_row: int
    rolled_double: bool


@dataclass(slots=True)
class PendingJailDecisionState:
    """Pending state for a jailed player's release decision."""

    player_name: str
    available_actions: list[str]


@dataclass(slots=True)
class PendingPropertyActionState:
    """Pending state for build, sell, mortgage, or unmortgage confirmation."""

    action_type: str
    player_name: str
    property_name: str
    property_index: int


@dataclass(slots=True)
class PendingTradeDecisionState:
    """Pending state for an unanswered trade offer."""

    trade_offer: TradeOffer
    counter_count: int = 0


class Game:
    """Authoritative Monopoly game engine.

    The game advances through explicit phases: ``pre_roll`` for starting a turn,
    ``in_turn`` while resolving interactive decisions, and ``post_roll`` when the
    player is done acting and must explicitly end the turn. Pending decision
    fields encode which sub-flow currently owns control.
    """

    def __init__(
        self,
        player_names: list[str],
        dice: Dice | None = None,
        board: Board | None = None,
        player_roles: dict[str, str] | list[str] | None = None,
        starting_cash: int = STARTING_CASH,
    ) -> None:
        """Create a new game with validated players, board, dice, and starting bank.

        ``player_roles`` may be omitted for all-human games, supplied as a list in
        player order, or supplied as a name-to-role mapping. The constructor also
        initializes all pending interactive state so local play, online hosting,
        and RL environments can serialize the game consistently from turn one.
        """
        self._validate_player_count(player_names)
        if starting_cash <= 0:
            raise ValueError("starting_cash must be greater than zero.")
        normalized_roles = self._normalize_player_roles(player_names, player_roles)
        self.players = [Player(name=name, role=normalized_roles[name], cash=starting_cash) for name in player_names]
        self.board = board or create_standard_board()
        self.dice = dice or Dice()
        self.starting_cash = starting_cash
        self.current_player_index = 0
        self.turn_counter = 0
        self.houses_remaining = BANK_HOUSES
        self.hotels_remaining = BANK_HOTELS
        self.auction_bids_by_space: dict[int, dict[str, int]] = {}
        self.pending_purchase_decision: PurchaseDecisionState | None = None
        self.pending_auction: PendingAuctionState | None = None
        self.pending_jail_decision: PendingJailDecisionState | None = None
        self.pending_property_action: PendingPropertyActionState | None = None
        self.pending_trade_decision: PendingTradeDecisionState | None = None
        self._blocked_property_action_requests: set[tuple[str, str, int]] = set()
        self._blocked_trade_offer_signatures: set[str] = set()
        self._trade_proposals_this_turn: set[tuple[str, str]] = set()
        self._pending_turn_continuation: TurnContinuation | None = None
        self.debug_next_rolls_by_player: dict[str, list[tuple[int, int]]] = {}
        self._interactive_flow_active = False
        self.current_turn_phase = PRE_ROLL_PHASE

    @staticmethod
    def _validate_player_count(player_names: list[str]) -> None:
        """Reject unsupported player counts and duplicate player names."""
        if not MIN_PLAYERS <= len(player_names) <= MAX_PLAYERS:
            raise ValueError(f"Monopoly supports between {MIN_PLAYERS} and {MAX_PLAYERS} players in this model.")
        if len(set(player_names)) != len(player_names):
            raise ValueError("Player names must be unique.")

    @staticmethod
    def _normalize_player_roles(
        player_names: list[str],
        player_roles: dict[str, str] | list[str] | None,
    ) -> dict[str, str]:
        """Normalize optional role input into a validated name-to-role mapping."""
        if player_roles is None:
            return {name: HUMAN_ROLE for name in player_names}

        if isinstance(player_roles, list):
            if len(player_roles) != len(player_names):
                raise ValueError("player_roles list must match the number of players.")
            role_map = {name: role for name, role in zip(player_names, player_roles, strict=True)}
        else:
            missing_names = [name for name in player_names if name not in player_roles]
            if missing_names:
                raise ValueError(f"Missing roles for players: {', '.join(missing_names)}")
            role_map = {name: player_roles[name] for name in player_names}

        invalid_roles = sorted({role for role in role_map.values() if role not in PLAYER_ROLES})
        if invalid_roles:
            raise ValueError(f"Unsupported player roles: {', '.join(invalid_roles)}")
        return role_map

    @property
    def active_players(self) -> list[Player]:
        """Return players who have not gone bankrupt."""
        return [player for player in self.players if not player.is_bankrupt]

    @property
    def current_player(self) -> Player:
        """Return the active player whose turn index is currently selected."""
        active_players = self.active_players
        if not active_players:
            raise RuntimeError("No active players remain.")
        if self.current_player_index >= len(active_players):
            self.current_player_index = 0
        return active_players[self.current_player_index]

    def next_player(self) -> None:
        """Advance turn order to the next non-bankrupt player and reset the phase."""
        active_players = self.active_players
        if not active_players:
            return
        self.current_player_index = (self.current_player_index + 1) % len(active_players)
        self.current_turn_phase = PRE_ROLL_PHASE

    def move_player_by(self, player: Player, steps: int) -> list[str]:
        """Move a player forward, handling board wraparound and salary collection."""
        old_position = player.position
        new_position = (player.position + steps) % BOARD_SIZE
        passed_go = old_position + steps >= BOARD_SIZE
        player.move_to(new_position)
        messages = [f"{player.name} moves from {old_position} to {new_position} ({self.board.get_space(new_position).name})."]
        if passed_go:
            player.receive(GO_SALARY)
            messages.append(f"{player.name} passes Go and collects ${GO_SALARY}.")
        return messages

    def move_player_to(self, player: Player, position: int, collect_go_salary: bool) -> None:
        """Move a player directly to a board position, optionally paying for passing Go."""
        if collect_go_salary:
            player.receive(GO_SALARY)
        player.move_to(position)

    def _roll_for_player(self, player_name: str) -> DiceRoll:
        """Consume a scripted debug roll first, then fall back to the dice object."""
        scripted_rolls = self.debug_next_rolls_by_player.get(player_name)
        if scripted_rolls:
            die_one, die_two = scripted_rolls.pop(0)
            if not scripted_rolls:
                self.debug_next_rolls_by_player.pop(player_name, None)
            return DiceRoll(die_one=int(die_one), die_two=int(die_two))
        return self.dice.roll()

    def _recalculate_building_supply_from_board(self) -> None:
        """Rebuild bank house and hotel inventory from the current street states."""
        houses_used = 0
        hotels_used = 0
        for space in self.board.spaces:
            if not isinstance(space, StreetPropertySpace):
                continue
            if space.building_count >= 5:
                hotels_used += 1
            else:
                houses_used += space.building_count
        self.houses_remaining = BANK_HOUSES - houses_used
        self.hotels_remaining = BANK_HOTELS - hotels_used

    def send_player_to_jail(self, player: Player) -> None:
        """Move a player to jail and apply the standard jail-side effects."""
        player.move_to(JAIL_INDEX)
        player.in_jail = True
        player.jail_turns = 0

    def take_turn(self, auto_buy_unowned: bool = True) -> TurnReport:
        if self.has_pending_interaction():
            raise ValueError("Cannot auto-play a turn while an interactive decision is pending.")
        if self.current_turn_phase != PRE_ROLL_PHASE:
            raise ValueError("Cannot auto-play a turn while the move is still being resolved.")
        player = self.current_player
        report = TurnReport(player_name=player.name)

        if player.is_bankrupt:
            report.messages.append(f"{player.name} is bankrupt and cannot take a turn.")
            self.next_player()
            return report

        report.messages.append(f"Turn {self.turn_counter + 1}: {player.name} begins.")

        doubles_in_row = 0
        while True:
            if player.in_jail:
                jail_result = self._handle_jail_turn(player)
                report.messages.extend(jail_result.messages)
                if jail_result.turn_consumed:
                    break
                if player.in_jail:
                    break

            roll = self._roll_for_player(player.name)
            report.messages.append(
                f"{player.name} rolls {roll.die_one} and {roll.die_two} (total {roll.total})."
            )

            if roll.is_double:
                doubles_in_row += 1
            else:
                doubles_in_row = 0

            if doubles_in_row == 3:
                self.send_player_to_jail(player)
                report.messages.append(f"{player.name} rolled doubles three times and goes to jail.")
                break

            report.messages.extend(self.move_player_by(player, roll.total))
            report.messages.extend(
                self.resolve_current_space(
                    player,
                    allow_property_purchase=auto_buy_unowned,
                    dice_total=roll.total,
                )
            )

            if player.in_jail or player.is_bankrupt or not roll.is_double:
                break

            report.messages.append(f"{player.name} rolled doubles and takes another turn.")

        self.turn_counter += 1
        if len(self.active_players) > 1:
            self.next_player()
        return report

    def _handle_jail_turn(self, player: Player) -> JailTurnResult:
        messages = [f"{player.name} starts the turn in jail."]

        if player.get_out_of_jail_cards > 0:
            player.get_out_of_jail_cards -= 1
            player.in_jail = False
            player.jail_turns = 0
            messages.append(f"{player.name} uses a Get Out of Jail Free card.")
            return JailTurnResult(messages=messages, turn_consumed=False)

        if player.jail_turns >= 2:
            messages.extend(self.charge_player(player, None, JAIL_FINE, "Jail Fine"))
            player.in_jail = False
            player.jail_turns = 0
            messages.append(f"{player.name} pays the jail fine and leaves jail.")
            return JailTurnResult(messages=messages, turn_consumed=False)

        roll = self._roll_for_player(player.name)
        messages.append(f"{player.name} attempts to roll doubles: {roll.die_one} and {roll.die_two}.")
        if roll.is_double:
            player.in_jail = False
            player.jail_turns = 0
            messages.append(f"{player.name} rolls doubles and leaves jail.")
            messages.extend(self.move_player_by(player, roll.total))
            messages.extend(self.resolve_current_space(player, allow_property_purchase=True, dice_total=roll.total))
            return JailTurnResult(messages=messages, turn_consumed=True)

        player.jail_turns += 1
        messages.append(f"{player.name} does not roll doubles and remains in jail.")
        return JailTurnResult(messages=messages, turn_consumed=True)

    def resolve_current_space(
        self,
        player: Player,
        allow_property_purchase: bool,
        dice_total: int = 0,
        forced_railroad_multiplier: int = 1,
        forced_utility_multiplier: int | None = None,
        interactive: bool | None = None,
    ) -> list[str]:
        """Resolve the board space under the player's token.

        The same landing rules are shared by auto-play, interactive desktop play,
        online hosting, and training. The `interactive` flag selects whether the
        engine should open pending decisions or resolve outcomes immediately.
        """
        if interactive is None:
            interactive = self._interactive_flow_active
        if interactive:
            return self._resolve_current_space_interactive(
                player,
                dice_total=dice_total,
                forced_railroad_multiplier=forced_railroad_multiplier,
                forced_utility_multiplier=forced_utility_multiplier,
            )

        return self._resolve_current_space_auto(
            player,
            allow_property_purchase=allow_property_purchase,
            dice_total=dice_total,
            forced_railroad_multiplier=forced_railroad_multiplier,
            forced_utility_multiplier=forced_utility_multiplier,
        )

    def _resolve_current_space_auto(
        self,
        player: Player,
        allow_property_purchase: bool,
        dice_total: int = 0,
        forced_railroad_multiplier: int = 1,
        forced_utility_multiplier: int | None = None,
    ) -> list[str]:
        """Resolve a landing immediately without creating pending UI decisions."""
        space = self.board.get_space(player.position)
        messages = [f"{player.name} lands on {space.name}."]

        if isinstance(space, PropertySpace):
            messages.extend(
                self._resolve_property_landing(
                    player,
                    space,
                    dice_total,
                    allow_property_purchase,
                    forced_railroad_multiplier,
                    forced_utility_multiplier,
                )
            )
        elif isinstance(space, TaxSpace):
            messages.extend(self.charge_player(player, None, space.tax_amount, space.name))
        elif isinstance(space, CardSpace):
            card = self.board.draw_card(space.deck_name)
            messages.append(f"{player.name} draws {card.name}: {card.description}")
            messages.extend(card.apply(self, player))
        elif isinstance(space, ActionSpace) and space.space_type == "go_to_jail":
            self.send_player_to_jail(player)
            messages.append(f"{player.name} goes directly to jail.")
        else:
            messages.append(f"{space.name} does not trigger a payment or purchase.")

        return messages

    def _resolve_current_space_interactive(
        self,
        player: Player,
        dice_total: int = 0,
        forced_railroad_multiplier: int = 1,
        forced_utility_multiplier: int | None = None,
    ) -> list[str]:
        """Resolve a landing for the interactive flow, opening pending prompts when needed."""
        space = self.board.get_space(player.position)
        messages = [f"{player.name} lands on {space.name}."]

        if isinstance(space, PropertySpace):
            if space.owner is None:
                self.pending_purchase_decision = PurchaseDecisionState(
                    player_name=player.name,
                    property_index=space.index,
                    property_name=space.name,
                    price=space.price,
                )
                messages.append(f"{player.name} may buy {space.name} for ${space.price} or send it to auction.")
                return messages
            messages.extend(
                self._resolve_property_landing(
                    player,
                    space,
                    dice_total,
                    allow_property_purchase=False,
                    forced_railroad_multiplier=forced_railroad_multiplier,
                    forced_utility_multiplier=forced_utility_multiplier,
                )
            )
            return messages

        if isinstance(space, TaxSpace):
            messages.extend(self.charge_player(player, None, space.tax_amount, space.name))
        elif isinstance(space, CardSpace):
            card = self.board.draw_card(space.deck_name)
            messages.append(f"{player.name} draws {card.name}: {card.description}")
            messages.extend(card.apply(self, player))
        elif isinstance(space, ActionSpace) and space.space_type == "go_to_jail":
            self.send_player_to_jail(player)
            messages.append(f"{player.name} goes directly to jail.")
        else:
            messages.append(f"{space.name} does not trigger a payment or purchase.")

        return messages

    def _resolve_property_landing(
        self,
        player: Player,
        property_space: PropertySpace,
        dice_total: int,
        allow_property_purchase: bool,
        forced_railroad_multiplier: int,
        forced_utility_multiplier: int | None,
    ) -> list[str]:
        """Handle buy, rent, auction, or self-owned outcomes for a property landing."""
        if property_space.owner is None:
            if allow_property_purchase and player.cash >= property_space.price:
                player.pay(property_space.price)
                property_space.assign_owner(player)
                return [f"{player.name} buys {property_space.name} for ${property_space.price}."]
            messages = [f"{property_space.name} is unowned."]
            bids = self.auction_bids_by_space.pop(property_space.index, None)
            auction_result = self.auction_property(property_space, bids)
            messages.extend(auction_result.messages)
            return messages

        if property_space.owner is player:
            return [f"{player.name} already owns {property_space.name}."]

        rent = MonopolyRules.calculate_rent(
            self.board,
            property_space,
            dice_total,
            railroad_multiplier=forced_railroad_multiplier,
            utility_multiplier=forced_utility_multiplier,
        )
        if rent == 0:
            return [f"{property_space.name} is mortgaged, so no rent is due."]
        return self.charge_player(player, property_space.owner, rent, f"Rent on {property_space.name}")

    def charge_player(self, payer: Player, payee: Player | None, amount: int, reason: str) -> list[str]:
        """Collect a payment, attempting liquidation first and bankrupting if necessary."""
        if amount <= 0:
            return [f"No payment is required for {reason}."]

        messages = self._attempt_auto_liquidation(payer, amount)
        if payer.cash < amount:
            messages.extend(self._bankrupt_player(payer, payee, reason))
            return messages

        payer.pay(amount)
        if payee is not None:
            payee.receive(amount)
            messages.append(f"{payer.name} pays ${amount} to {payee.name} for {reason}.")
        else:
            messages.append(f"{payer.name} pays ${amount} to the bank for {reason}.")
        return messages

    def _bankrupt_player(self, player: Player, creditor: Player | None, reason: str) -> list[str]:
        """Resolve bankruptcy by transferring or releasing all remaining assets."""
        messages = [f"{player.name} cannot cover the debt for {reason} and goes bankrupt."]
        player.is_bankrupt = True
        player.in_jail = False

        if creditor is not None:
            if player.cash > 0:
                creditor.receive(player.cash)
                messages.append(f"{creditor.name} receives ${player.cash} from {player.name}'s remaining cash.")
                player.cash = 0
            for property_space in list(player.properties):
                player.remove_property(property_space)
                if isinstance(property_space, StreetPropertySpace):
                    self._reclaim_building_supply(property_space)
                creditor.add_property(property_space)
                property_space.owner = creditor
                messages.append(f"{property_space.name} transfers to {creditor.name}.")
            creditor.get_out_of_jail_cards += player.get_out_of_jail_cards
            player.get_out_of_jail_cards = 0
        else:
            for property_space in list(player.properties):
                if isinstance(property_space, StreetPropertySpace):
                    self._reclaim_building_supply(property_space)
                property_space.release_to_bank()
                messages.append(f"{property_space.name} returns to the bank.")

        return messages

    def build_on_property(self, player: Player, property_name: str) -> list[str]:
        """Apply one legal build step and update bank house or hotel inventory."""
        street = self._find_street(player, property_name)
        transaction = MonopolyRules.build_house(
            self.board,
            player,
            street,
            houses_remaining=self.houses_remaining,
            hotels_remaining=self.hotels_remaining,
        )
        self.houses_remaining += transaction.house_delta
        self.hotels_remaining += transaction.hotel_delta
        return transaction.messages

    def sell_building(self, player: Player, property_name: str) -> list[str]:
        """Sell one legal building step and restore the bank supply counters."""
        street = self._find_street(player, property_name)
        transaction = MonopolyRules.sell_building(
            self.board,
            player,
            street,
            houses_remaining=self.houses_remaining,
        )
        self.houses_remaining += transaction.house_delta
        self.hotels_remaining += transaction.hotel_delta
        return transaction.messages

    def mortgage_property(self, player: Player, property_name: str) -> list[str]:
        """Mortgage one owned property through the shared rules helper."""
        property_space = self._find_property(player, property_name)
        return MonopolyRules.mortgage_property(self.board, player, property_space)

    def unmortgage_property(self, player: Player, property_name: str) -> list[str]:
        """Remove a mortgage from one owned property through the shared rules helper."""
        property_space = self._find_property(player, property_name)
        return MonopolyRules.unmortgage_property(player, property_space)

    def execute_trade(self, trade_offer: TradeOffer) -> list[str]:
        """Execute a validated trade offer and return the narration it emits."""
        return trade_offer.execute()

    def request_property_action(self, player_name: str, action_type: str, property_name: str) -> InteractionResult:
        """Open a confirm-or-cancel prompt for a build, sell, mortgage, or unmortgage action."""
        player = self._get_player_by_name(player_name)
        self._ensure_free_action_window(player)
        property_space = self._find_property(player, property_name)
        if self._is_property_action_request_blocked(player.name, action_type, property_space.index):
            raise ValueError(f"{player.name} already cancelled the {action_type} action for {property_space.name} this turn.")

        if action_type == "build":
            if not isinstance(property_space, StreetPropertySpace):
                raise ValueError(f"{property_name} is not a street property.")
            can_build, reason = MonopolyRules.can_build_house(self.board, player, property_space)
            if not can_build:
                raise ValueError(reason)
        elif action_type == "sell_building":
            if not isinstance(property_space, StreetPropertySpace):
                raise ValueError(f"{property_name} is not a street property.")
            can_sell, reason = MonopolyRules.can_sell_building(self.board, player, property_space)
            if not can_sell:
                raise ValueError(reason)
        elif action_type == "mortgage":
            can_mortgage, reason = MonopolyRules.can_mortgage(self.board, player, property_space)
            if not can_mortgage:
                raise ValueError(reason)
        elif action_type == "unmortgage":
            if not property_space.mortgaged:
                raise ValueError(f"{property_space.name} is not mortgaged.")
            if player.cash < MonopolyRules.unmortgage_cost(property_space):
                raise ValueError(f"{player.name} cannot afford to unmortgage {property_space.name}.")
        else:
            raise ValueError(f"Unsupported property action: {action_type}")

        self.pending_property_action = PendingPropertyActionState(
            action_type=action_type,
            player_name=player.name,
            property_name=property_space.name,
            property_index=property_space.index,
        )
        return self._build_interaction_result([f"{player.name} is deciding whether to {self._property_action_label(action_type)} {property_space.name}."])

    def resolve_property_action(self, confirm: bool) -> InteractionResult:
        """Apply or cancel the currently pending property action request."""
        if self.pending_property_action is None:
            raise ValueError("There is no property action to resolve.")

        action = self.pending_property_action
        self.pending_property_action = None
        player = self._get_player_by_name(action.player_name)
        messages: list[str] = []
        if not confirm:
            self._block_property_action_request(player.name, action.action_type, action.property_index)
            messages.append(f"{player.name} cancels the {action.action_type} action for {action.property_name}.")
            return self._build_interaction_result(messages)

        if action.action_type == "build":
            messages.extend(self.build_on_property(player, action.property_name))
        elif action.action_type == "sell_building":
            messages.extend(self.sell_building(player, action.property_name))
        elif action.action_type == "mortgage":
            messages.extend(self.mortgage_property(player, action.property_name))
        elif action.action_type == "unmortgage":
            messages.extend(self.unmortgage_property(player, action.property_name))
        else:
            raise ValueError(f"Unsupported property action: {action.action_type}")
        return self._build_interaction_result(messages)

    def propose_trade_interactive(self, trade_offer: TradeOffer) -> InteractionResult:
        """Open a pending trade decision after validating phase limits and offer contents."""
        self._ensure_free_action_window(trade_offer.proposer)

        errors = trade_offer.validate()
        if errors:
            raise ValueError(" ".join(errors))
        if not self._can_propose_trade_this_phase(trade_offer.proposer.name):
            raise ValueError(f"{trade_offer.proposer.name} has already made a trade offer during this phase.")
        if self.is_trade_offer_blocked(trade_offer):
            raise ValueError("That trade offer was already rejected this turn.")

        self._record_trade_proposal(trade_offer.proposer.name)
        self.pending_trade_decision = PendingTradeDecisionState(trade_offer=trade_offer, counter_count=0)
        return self._build_interaction_result(
            [f"{trade_offer.proposer.name} offers a trade to {trade_offer.receiver.name}."]
        )

    def counter_trade_interactive(self, trade_offer: TradeOffer) -> InteractionResult:
        """Replace the active trade decision with one permitted counter-offer."""
        if self.pending_trade_decision is None:
            raise ValueError("There is no trade decision to counter.")

        pending = self.pending_trade_decision
        original_offer = pending.trade_offer
        if pending.counter_count >= 1:
            raise ValueError("This trade has already been countered once.")
        if trade_offer.proposer is not original_offer.receiver:
            raise ValueError(f"Counter trade proposer must be {original_offer.receiver.name}.")
        if trade_offer.receiver is not original_offer.proposer:
            raise ValueError(f"Counter trade receiver must be {original_offer.proposer.name}.")

        errors = trade_offer.validate()
        if errors:
            raise ValueError(" ".join(errors))
        if not self._can_propose_trade_this_phase(trade_offer.proposer.name):
            raise ValueError(f"{trade_offer.proposer.name} has already made a trade offer during this phase.")
        if self.is_trade_offer_blocked(trade_offer):
            raise ValueError("That trade offer was already rejected this turn.")

        self._block_trade_offer(original_offer)
        self._record_trade_proposal(trade_offer.proposer.name)
        self.pending_trade_decision = PendingTradeDecisionState(trade_offer=trade_offer, counter_count=pending.counter_count + 1)
        return self._build_interaction_result(
            [
                f"{original_offer.receiver.name} rejects the trade from {original_offer.proposer.name} and makes a counter-offer.",
                f"{trade_offer.proposer.name} offers a counter-trade to {trade_offer.receiver.name}.",
            ]
        )

    def resolve_trade_decision(self, accept: bool) -> InteractionResult:
        """Accept or reject the currently pending trade and resume the turn flow."""
        if self.pending_trade_decision is None:
            raise ValueError("There is no trade decision to resolve.")

        pending = self.pending_trade_decision
        self.pending_trade_decision = None
        trade_offer = pending.trade_offer
        if not accept:
            self._block_trade_offer(trade_offer)
            return self._build_interaction_result(
                [f"{trade_offer.receiver.name} rejects the trade from {trade_offer.proposer.name}."]
            )

        return self._build_interaction_result(trade_offer.execute())

    def start_turn_interactive(self) -> InteractionResult:
        """Enter the interactive turn loop from the pre-roll phase."""
        if self.has_pending_interaction():
            raise ValueError("Finish the current choice before rolling.")
        if self.current_turn_phase != PRE_ROLL_PHASE:
            raise ValueError("You can only roll at the start of a turn.")

        player = self.current_player
        messages = [f"Turn {self.turn_counter + 1}: {player.name} begins."]
        return self._play_interactive_turn(player, doubles_in_row=0, opening_messages=messages)

    def end_turn_interactive(self) -> InteractionResult:
        """Finish a resolved turn, advance order, and clear per-turn interaction blockers."""
        if self.has_pending_interaction():
            raise ValueError("Finish the current choice before ending the turn.")
        if self.current_turn_phase != POST_ROLL_PHASE:
            raise ValueError("You can only end the turn after the move is resolved.")

        player = self.current_player
        messages = [f"{player.name} ends the turn."]
        return self._complete_interactive_turn(messages)

    def resolve_property_decision(self, buy_property: bool) -> InteractionResult:
        """Resolve an unowned-property prompt by buying immediately or opening an auction."""
        if self.pending_purchase_decision is None:
            raise ValueError("There is no property purchase decision to resolve.")

        decision = self.pending_purchase_decision
        self.pending_purchase_decision = None
        player = self._get_player_by_name(decision.player_name)
        property_space = self.board.get_space(decision.property_index)
        messages: list[str] = []

        if buy_property and player.cash >= property_space.price:
            player.pay(property_space.price)
            property_space.assign_owner(player)
            messages.append(f"{player.name} buys {property_space.name} for ${property_space.price}.")
            return self._continue_after_pending_interaction(messages)

        messages.append(f"{player.name} declines to buy {property_space.name}.")
        messages.extend(self._open_interactive_auction(property_space))
        if self.pending_auction is None:
            return self._continue_after_pending_interaction(messages)
        return self._build_interaction_result(messages)

    def resolve_jail_decision(self, action: str) -> InteractionResult:
        """Resolve a jailed player's chosen release method and continue the turn if possible."""
        if self.pending_jail_decision is None:
            raise ValueError("There is no jail decision to resolve.")

        decision = self.pending_jail_decision
        self.pending_jail_decision = None
        player = self._get_player_by_name(decision.player_name)
        if action not in decision.available_actions:
            raise ValueError(f"{action} is not an allowed jail action for {player.name}.")

        messages: list[str] = []
        if action == "use_card":
            player.get_out_of_jail_cards -= 1
            player.in_jail = False
            player.jail_turns = 0
            messages.append(f"{player.name} uses a Get Out of Jail Free card.")
            return self._play_interactive_turn(player, doubles_in_row=0, opening_messages=messages)

        if action == "pay_fine":
            messages.extend(self.charge_player(player, None, JAIL_FINE, "Jail Fine"))
            player.in_jail = False
            player.jail_turns = 0
            messages.append(f"{player.name} pays the jail fine and leaves jail.")
            return self._play_interactive_turn(player, doubles_in_row=0, opening_messages=messages)

        roll = self._roll_for_player(player.name)
        messages.append(f"{player.name} attempts to roll doubles: {roll.die_one} and {roll.die_two}.")
        if roll.is_double:
            player.in_jail = False
            player.jail_turns = 0
            messages.append(f"{player.name} rolls doubles and leaves jail.")
            self._pending_turn_continuation = TurnContinuation(player_name=player.name, doubles_in_row=0, rolled_double=False)
            messages.extend(self.move_player_by(player, roll.total))
            messages.extend(self.resolve_current_space(player, allow_property_purchase=False, dice_total=roll.total, interactive=True))
            if self.has_pending_interaction():
                return self._build_interaction_result(messages)
            return self._conclude_roll_sequence(player, messages)

        if player.jail_turns >= 2:
            messages.append(f"{player.name} fails the third jail roll and must pay the fine.")
            messages.extend(self.charge_player(player, None, JAIL_FINE, "Jail Fine"))
            player.in_jail = False
            player.jail_turns = 0
            messages.extend(self.move_player_by(player, roll.total))
            self._pending_turn_continuation = TurnContinuation(player_name=player.name, doubles_in_row=0, rolled_double=False)
            messages.extend(self.resolve_current_space(player, allow_property_purchase=False, dice_total=roll.total, interactive=True))
            if self.has_pending_interaction():
                return self._build_interaction_result(messages)
            return self._conclude_roll_sequence(player, messages)

        player.jail_turns += 1
        messages.append(f"{player.name} does not roll doubles and remains in jail.")
        return self._conclude_roll_sequence(player, messages)

    def submit_auction_bid(self, player_name: str, bid_amount: int | None) -> InteractionResult:
        """Record one bid or pass in the live interactive auction flow."""
        if self.pending_auction is None:
            raise ValueError("There is no auction in progress.")

        auction = self.pending_auction
        current_bidder = auction.active_player_names[auction.current_bidder_index]
        if current_bidder != player_name:
            raise ValueError(f"It is {current_bidder}'s turn to act in the auction.")

        player = self._get_player_by_name(player_name)
        messages: list[str] = []
        if bid_amount is None:
            auction.active_player_names.pop(auction.current_bidder_index)
            messages.append(f"{player_name} passes in the auction for {auction.property_name}.")
        else:
            minimum_bid = self._get_pending_auction_minimum_bid(auction)
            if bid_amount < minimum_bid:
                raise ValueError(f"The minimum valid bid is ${minimum_bid}.")
            if bid_amount > player.cash:
                raise ValueError(f"{player.name} cannot afford a bid of ${bid_amount}.")
            auction.current_bid = bid_amount
            auction.current_winner_name = player.name
            messages.append(f"{player.name} bids ${bid_amount} for {auction.property_name}.")
            auction.current_bidder_index = (auction.current_bidder_index + 1) % len(auction.active_player_names)

        if self._auction_is_ready_to_resolve(auction):
            messages.extend(self._resolve_interactive_auction())
            return self._continue_after_pending_interaction(messages)

        if auction.active_player_names:
            auction.current_bidder_index %= len(auction.active_player_names)
            next_bidder = auction.active_player_names[auction.current_bidder_index]
            messages.append(f"{next_bidder} is now up in the auction.")
        return self._build_interaction_result(messages)

    def auction_property(self, property_space: PropertySpace, bids: dict[str, int] | None = None) -> AuctionResult:
        """Run a complete non-interactive auction and return the resolved outcome."""
        if property_space.owner is not None:
            raise ValueError(f"{property_space.name} is already owned.")

        messages = [f"Auction begins for {property_space.name}."]
        valid_bids: list[tuple[int, int, Player]] = []

        if bids is None:
            bids = self._default_auction_bids(property_space)

        for turn_order, player in enumerate(self.active_players):
            bid_amount = bids.get(player.name, 0)
            if bid_amount <= 0:
                continue
            if bid_amount > player.cash:
                messages.append(f"{player.name}'s bid of ${bid_amount} is ignored because they cannot afford it.")
                continue
            valid_bids.append((bid_amount, -turn_order, player))

        if not valid_bids:
            messages.append(f"No valid bids were placed for {property_space.name}. The bank keeps the property.")
            return AuctionResult(winner=None, messages=messages)

        winning_bid, _, winner = max(valid_bids)
        winner.pay(winning_bid)
        property_space.assign_owner(winner)
        messages.append(f"{winner.name} wins the auction for {property_space.name} at ${winning_bid}.")
        return AuctionResult(winner=winner, winning_bid=winning_bid, messages=messages)

    def calculate_repair_fee(self, player: Player, price_per_house: int, price_per_hotel: int) -> int:
        """Compute one card-driven repair charge across all developed streets the player owns."""
        total = 0
        for property_space in player.properties:
            if isinstance(property_space, StreetPropertySpace):
                if property_space.has_hotel:
                    total += price_per_hotel
                else:
                    total += property_space.building_count * price_per_house
        return total

    def winner(self) -> Player | None:
        """Return the sole surviving player once only one non-bankrupt seat remains."""
        active_players = self.active_players
        if len(active_players) == 1:
            return active_players[0]
        return None

    def game_state_summary(self) -> str:
        """Return a compact human-readable snapshot of the current authoritative state."""
        lines = [
            f"Turn counter: {self.turn_counter}",
            f"Bank supply: houses={self.houses_remaining}, hotels={self.hotels_remaining}",
        ]
        for player in self.players:
            lines.append(player.summary())
        return "\n".join(lines)

    def _attempt_auto_liquidation(self, player: Player, amount_needed: int) -> list[str]:
        """Sell buildings and mortgage assets until the player can pay or has no legal options."""
        messages: list[str] = []

        while player.cash < amount_needed:
            sold_something = False

            for property_space in [space for space in player.properties if isinstance(space, StreetPropertySpace)]:
                while property_space.building_count > 0 and player.cash < amount_needed:
                    try:
                        messages.extend(self.sell_building(player, property_space.name))
                        sold_something = True
                    except ValueError:
                        break

            if player.cash >= amount_needed:
                break

            mortgage_candidates = sorted(
                [space for space in player.properties if not space.mortgaged],
                key=lambda item: item.mortgage_value,
                reverse=True,
            )
            for property_space in mortgage_candidates:
                try:
                    messages.extend(self.mortgage_property(player, property_space.name))
                    sold_something = True
                    if player.cash >= amount_needed:
                        break
                except ValueError:
                    continue

            if not sold_something:
                break

        return messages

    def _reclaim_building_supply(self, street: StreetPropertySpace) -> None:
        """Return houses or hotels from a released street back to the bank counters."""
        if street.building_count == 0:
            return
        if street.has_hotel:
            self.hotels_remaining += 1
        else:
            self.houses_remaining += street.building_count

    def _default_auction_bids(self, property_space: PropertySpace) -> dict[str, int]:
        """Generate a simple fallback auction bid map for deterministic auto-play flows."""
        bids: dict[str, int] = {}
        reserve = max(1, property_space.price // 2)
        for player in self.active_players:
            bid = min(player.cash, property_space.price)
            if bid >= reserve:
                bids[player.name] = bid
        return bids

    def get_game_view(self) -> GameView:
        """Build the high-level public game snapshot consumed by UI and replay layers."""
        current_player_name = self.current_player.name if self.active_players else None
        current_player_role = self.current_player.role if self.active_players else None
        players = tuple(
            PlayerView(
                name=player.name,
                role=player.role,
                cash=player.cash,
                position=player.position,
                in_jail=player.in_jail,
                jail_turns=player.jail_turns,
                get_out_of_jail_cards=player.get_out_of_jail_cards,
                is_bankrupt=player.is_bankrupt,
                properties=tuple(property_space.name for property_space in player.properties),
            )
            for player in self.players
        )
        return GameView(
            turn_counter=self.turn_counter,
            current_player_name=current_player_name,
            current_player_role=current_player_role,
            current_turn_phase=self.current_turn_phase if self.active_players else None,
            starting_cash=self.starting_cash,
            houses_remaining=self.houses_remaining,
            hotels_remaining=self.hotels_remaining,
            players=players,
            pending_action=self.get_pending_action(),
            blocked_trade_offer_signatures=tuple(sorted(self._blocked_trade_offer_signatures)),
        )

    def get_game_setup(self) -> GameSetup:
        """Expose the current seat configuration as a reusable setup payload."""
        return GameSetup(
            player_names=tuple(player.name for player in self.players),
            starting_cash=self.starting_cash,
            player_roles=tuple(player.role for player in self.players),
        )

    def get_active_turn_plan(self) -> TurnPlanView:
        """Return the turn plan for the actor who must respond right now."""
        pending_action = self.get_pending_action()
        if pending_action is not None:
            return self.get_turn_plan(pending_action.player_name)
        return self.get_turn_plan(self.current_player.name)

    def get_frontend_state(self) -> FrontendStateView:
        """Build the complete frontend snapshot: game summary, active plan, and board view."""
        return FrontendStateView(
            game_view=self.get_game_view(),
            active_turn_plan=self.get_active_turn_plan(),
            board_spaces=self.get_board_space_views(),
        )

    def get_board_space_views(self) -> tuple[BoardSpaceView, ...]:
        """Project all board spaces into serializable UI-friendly view objects."""
        occupants_by_space: dict[int, list[str]] = {}
        for player in self.players:
            occupants_by_space.setdefault(player.position, []).append(player.name)

        spaces: list[BoardSpaceView] = []
        for space in self.board.spaces:
            owner_name = None
            color_group = None
            mortgaged = False
            building_count = None
            price = None
            house_cost = None
            notes = None

            if isinstance(space, PropertySpace):
                owner_name = None if space.owner is None else space.owner.name
                mortgaged = space.mortgaged
                price = space.price
            if isinstance(space, StreetPropertySpace):
                color_group = space.color_group
                building_count = space.building_count
                house_cost = space.house_cost
            if isinstance(space, ActionSpace):
                notes = space.notes
            elif isinstance(space, TaxSpace):
                notes = f"Tax: ${space.tax_amount}"
            elif isinstance(space, CardSpace):
                notes = f"Deck: {space.deck_name}"

            spaces.append(
                BoardSpaceView(
                    index=space.index,
                    name=space.name,
                    space_type=space.space_type,
                    occupant_names=tuple(occupants_by_space.get(space.index, [])),
                    owner_name=owner_name,
                    color_group=color_group,
                    mortgaged=mortgaged,
                    building_count=building_count,
                    price=price,
                    house_cost=house_cost,
                    notes=notes,
                )
            )
        return tuple(spaces)

    def get_pending_action(self) -> PendingActionView | None:
        """Translate internal pending state into the public action contract."""
        if self.pending_purchase_decision is not None:
            decision_player = self._get_player_by_name(self.pending_purchase_decision.player_name)
            if decision_player.is_bankrupt:
                self.pending_purchase_decision = None
                self._pending_turn_continuation = None
                if self.current_turn_phase == IN_TURN_PHASE:
                    self.turn_counter += 1
                    self.current_turn_phase = PRE_ROLL_PHASE

        if self.pending_jail_decision is not None:
            decision = self.pending_jail_decision
            player = self._get_player_by_name(decision.player_name)
            return PendingActionView(
                action_type="jail_decision",
                player_name=decision.player_name,
                player_role=player.role,
                turn_phase=IN_TURN_PHASE,
                prompt=f"{decision.player_name} must choose how to leave jail.",
                available_actions=tuple(decision.available_actions),
                jail=JailDecisionView(
                    player_name=decision.player_name,
                    player_role=player.role,
                    available_actions=tuple(decision.available_actions),
                    fine_amount=JAIL_FINE,
                    has_get_out_of_jail_card=player.get_out_of_jail_cards > 0,
                    jail_turns=player.jail_turns,
                ),
            )
        if self.pending_purchase_decision is not None:
            decision = self.pending_purchase_decision
            player = self._get_player_by_name(decision.player_name)
            return PendingActionView(
                action_type="property_purchase",
                player_name=decision.player_name,
                player_role=player.role,
                turn_phase=IN_TURN_PHASE,
                prompt=f"{decision.player_name} can buy {decision.property_name} for ${decision.price}, or send it to auction.",
                available_actions=("buy", "decline"),
                property_name=decision.property_name,
                property_index=decision.property_index,
                price=decision.price,
            )
        if self.pending_auction is not None:
            auction = self.pending_auction
            current_bidder_name = auction.active_player_names[auction.current_bidder_index] if auction.active_player_names else None
            return PendingActionView(
                action_type="auction",
                player_name=current_bidder_name or "",
                player_role=self._get_player_by_name(current_bidder_name).role if current_bidder_name else HUMAN_ROLE,
                turn_phase=IN_TURN_PHASE,
                prompt=f"Auction in progress for {auction.property_name}.",
                available_actions=("bid", "pass"),
                property_name=auction.property_name,
                property_index=auction.property_index,
                auction=AuctionView(
                    property_name=auction.property_name,
                    property_index=auction.property_index,
                    current_bid=auction.current_bid,
                    current_winner_name=auction.current_winner_name,
                    current_bidder_name=current_bidder_name,
                    eligible_player_names=tuple(auction.eligible_player_names),
                    active_player_names=tuple(auction.active_player_names),
                    minimum_bid=self._get_pending_auction_minimum_bid(auction),
                ),
            )
        if self.pending_property_action is not None:
            action = self.pending_property_action
            player = self._get_player_by_name(action.player_name)
            property_space = self.board.get_space(action.property_index)
            cash_effect = self._property_action_cash_effect(action.action_type, property_space)
            return PendingActionView(
                action_type="property_action",
                player_name=action.player_name,
                player_role=player.role,
                turn_phase=IN_TURN_PHASE,
                prompt=f"{action.player_name} is deciding whether to {self._property_action_label(action.action_type)} {action.property_name}.",
                available_actions=("confirm", "cancel"),
                property_name=action.property_name,
                property_index=action.property_index,
                property_action=PropertyActionView(
                    action_type=action.action_type,
                    player_name=player.name,
                    player_role=player.role,
                    property_name=action.property_name,
                    property_index=action.property_index,
                    cash_effect=cash_effect,
                    description=self._property_action_description(action.action_type, property_space),
                ),
            )
        if self.pending_trade_decision is not None:
            trade_offer = self.pending_trade_decision.trade_offer
            return PendingActionView(
                action_type="trade_decision",
                player_name=trade_offer.receiver.name,
                player_role=trade_offer.receiver.role,
                turn_phase=IN_TURN_PHASE,
                prompt=(
                    f"{trade_offer.receiver.name} must accept or reject the trade from {trade_offer.proposer.name}."
                    if self.pending_trade_decision.counter_count >= 1
                    else f"{trade_offer.receiver.name} must accept, reject, or counter the trade from {trade_offer.proposer.name}."
                ),
                available_actions=("accept", "reject") if self.pending_trade_decision.counter_count >= 1 else ("accept", "reject", "counter"),
                trade=TradeDecisionView(
                    proposer_name=trade_offer.proposer.name,
                    proposer_role=trade_offer.proposer.role,
                    receiver_name=trade_offer.receiver.name,
                    receiver_role=trade_offer.receiver.role,
                    proposer_cash=trade_offer.proposer_cash,
                    receiver_cash=trade_offer.receiver_cash,
                    proposer_property_names=tuple(space.name for space in trade_offer.proposer_properties),
                    receiver_property_names=tuple(space.name for space in trade_offer.receiver_properties),
                    proposer_jail_cards=trade_offer.proposer_jail_cards,
                    receiver_jail_cards=trade_offer.receiver_jail_cards,
                    note=trade_offer.note,
                ),
            )
        return None

    def get_pending_action_role(self) -> str | None:
        """Return the role of the actor who owns the current pending prompt, if any."""
        pending_action = self.get_pending_action()
        if pending_action is None:
            return None
        return pending_action.player_role

    def is_pending_action_for_ai(self) -> bool:
        """Whether the current pending prompt belongs to an AI-controlled seat."""
        return self.get_pending_action_role() == AI_ROLE

    def is_pending_action_for_human(self) -> bool:
        """Whether the current pending prompt belongs to a human-controlled seat."""
        return self.get_pending_action_role() == HUMAN_ROLE

    def execute_legal_action(
        self,
        action: LegalActionOption,
        *,
        bid_amount: int | None = None,
        trade_offer: TradeOffer | None = None,
    ) -> InteractionResult:
        """Execute one validated legal action from the public turn-plan contract."""
        plan = self.get_turn_plan(action.actor_name)
        if action not in plan.legal_actions:
            raise ValueError(f"{action.action_type} is not currently legal for {action.actor_name}.")

        if action.handler_name == "start_turn_interactive":
            return self.start_turn_interactive()

        if action.handler_name == "end_turn_interactive":
            return self.end_turn_interactive()

        if action.handler_name == "resolve_property_decision":
            if action.fixed_choice not in {"buy", "decline"}:
                raise ValueError("Property decision actions must specify a buy or decline choice.")
            return self.resolve_property_decision(buy_property=action.fixed_choice == "buy")

        if action.handler_name == "submit_auction_bid":
            if action.fixed_choice == "pass":
                if bid_amount is not None:
                    raise ValueError("Pass actions do not accept a bid amount.")
                return self.submit_auction_bid(action.actor_name, None)
            if bid_amount is None:
                raise ValueError("Bid actions require a bid amount.")
            return self.submit_auction_bid(action.actor_name, bid_amount)

        if action.handler_name == "resolve_jail_decision":
            if action.fixed_choice is None:
                raise ValueError("Jail decision actions must specify a fixed choice.")
            return self.resolve_jail_decision(action.fixed_choice)

        if action.handler_name == "request_property_action":
            if action.fixed_choice is None or action.property_name is None:
                raise ValueError("Property request actions must include the property name and action type.")
            return self.request_property_action(action.actor_name, action.fixed_choice, action.property_name)

        if action.handler_name == "resolve_property_action":
            if action.fixed_choice not in {"confirm", "cancel"}:
                raise ValueError("Property action resolution must specify confirm or cancel.")
            return self.resolve_property_action(confirm=action.fixed_choice == "confirm")

        if action.handler_name == "propose_trade_interactive":
            if trade_offer is None:
                raise ValueError("Trade proposal actions require a TradeOffer.")
            if trade_offer.proposer.name != action.actor_name:
                raise ValueError(f"Trade proposer must be {action.actor_name}.")
            if action.target_player_name is not None and trade_offer.receiver.name != action.target_player_name:
                raise ValueError(f"Trade receiver must be {action.target_player_name}.")
            return self.propose_trade_interactive(trade_offer)

        if action.handler_name == "counter_trade_interactive":
            if trade_offer is None:
                raise ValueError("Counter trade actions require a TradeOffer.")
            if trade_offer.proposer.name != action.actor_name:
                raise ValueError(f"Counter trade proposer must be {action.actor_name}.")
            if action.target_player_name is not None and trade_offer.receiver.name != action.target_player_name:
                raise ValueError(f"Counter trade receiver must be {action.target_player_name}.")
            return self.counter_trade_interactive(trade_offer)

        if action.handler_name == "resolve_trade_decision":
            if action.fixed_choice not in {"accept", "reject"}:
                raise ValueError("Trade decision actions must specify accept or reject.")
            return self.resolve_trade_decision(accept=action.fixed_choice == "accept")

        raise ValueError(f"Unsupported legal action handler: {action.handler_name}")

    def get_serialized_game_view(self) -> dict[str, Any]:
        """Return the public game snapshot encoded as plain serialized data."""
        return self.get_game_view().to_dict()

    def get_serialized_turn_plan(self, player_name: str | None = None) -> dict[str, Any]:
        """Return one player's public turn plan as serialized data."""
        return self.get_turn_plan(player_name).to_dict()

    def get_serialized_game_setup(self) -> dict[str, Any]:
        """Return the current seat configuration as serialized data."""
        return self.get_game_setup().to_dict()

    def get_serialized_frontend_state(self) -> dict[str, Any]:
        """Return the full frontend snapshot as serialized data."""
        return self.get_frontend_state().to_dict()

    def serialize_full_state(self) -> dict[str, Any]:
        """Serialize the authoritative runtime state required for save/load and hosting."""
        property_states: list[dict[str, Any]] = []
        for space in self.board.spaces:
            if not isinstance(space, PropertySpace):
                continue
            property_state = {
                "index": space.index,
                "owner_name": None if space.owner is None else space.owner.name,
                "mortgaged": space.mortgaged,
            }
            if isinstance(space, StreetPropertySpace):
                property_state["building_count"] = space.building_count
            property_states.append(property_state)

        current_player_name = None
        if self.active_players:
            current_player_name = self.current_player.name

        return {
            "starting_cash": self.starting_cash,
            "players": [
                {
                    "name": player.name,
                    "role": player.role,
                    "cash": player.cash,
                    "position": player.position,
                    "in_jail": player.in_jail,
                    "jail_turns": player.jail_turns,
                    "get_out_of_jail_cards": player.get_out_of_jail_cards,
                    "is_bankrupt": player.is_bankrupt,
                }
                for player in self.players
            ],
            "board": {
                "properties": property_states,
                "chance_deck": [card.name for card in self.board.chance_deck],
                "community_chest_deck": [card.name for card in self.board.community_chest_deck],
            },
            "runtime": {
                "current_player_name": current_player_name,
                "turn_counter": self.turn_counter,
                "houses_remaining": self.houses_remaining,
                "hotels_remaining": self.hotels_remaining,
                "auction_bids_by_space": {str(index): dict(bids) for index, bids in self.auction_bids_by_space.items()},
                "current_turn_phase": self.current_turn_phase,
                "pending_purchase_decision": None if self.pending_purchase_decision is None else {
                    "player_name": self.pending_purchase_decision.player_name,
                    "property_index": self.pending_purchase_decision.property_index,
                    "property_name": self.pending_purchase_decision.property_name,
                    "price": self.pending_purchase_decision.price,
                },
                "pending_auction": None if self.pending_auction is None else {
                    "property_index": self.pending_auction.property_index,
                    "property_name": self.pending_auction.property_name,
                    "eligible_player_names": list(self.pending_auction.eligible_player_names),
                    "active_player_names": list(self.pending_auction.active_player_names),
                    "current_bid": self.pending_auction.current_bid,
                    "current_winner_name": self.pending_auction.current_winner_name,
                    "current_bidder_index": self.pending_auction.current_bidder_index,
                },
                "pending_jail_decision": None if self.pending_jail_decision is None else {
                    "player_name": self.pending_jail_decision.player_name,
                    "available_actions": list(self.pending_jail_decision.available_actions),
                },
                "pending_property_action": None if self.pending_property_action is None else {
                    "action_type": self.pending_property_action.action_type,
                    "player_name": self.pending_property_action.player_name,
                    "property_name": self.pending_property_action.property_name,
                    "property_index": self.pending_property_action.property_index,
                },
                "blocked_property_action_requests": [
                    {
                        "player_name": player_name,
                        "action_type": action_type,
                        "property_index": property_index,
                    }
                    for player_name, action_type, property_index in sorted(self._blocked_property_action_requests)
                ],
                "blocked_trade_offer_signatures": sorted(self._blocked_trade_offer_signatures),
                "trade_proposals_this_turn": [
                    {
                        "player_name": player_name,
                        "turn_phase": turn_phase,
                    }
                    for player_name, turn_phase in sorted(self._trade_proposals_this_turn)
                ],
                "pending_trade_decision": None if self.pending_trade_decision is None else self.serialize_trade_offer(self.pending_trade_decision.trade_offer),
                "pending_trade_counter_count": 0 if self.pending_trade_decision is None else self.pending_trade_decision.counter_count,
                "pending_turn_continuation": None if self._pending_turn_continuation is None else {
                    "player_name": self._pending_turn_continuation.player_name,
                    "doubles_in_row": self._pending_turn_continuation.doubles_in_row,
                    "rolled_double": self._pending_turn_continuation.rolled_double,
                },
                "debug_next_rolls_by_player": {
                    player_name: [list(roll) for roll in rolls]
                    for player_name, rolls in self.debug_next_rolls_by_player.items()
                },
            },
            "dice": self.dice.to_dict(),
        }

    @classmethod
    def from_serialized_state(cls, payload: Mapping[str, Any]) -> Game:
        """Rebuild a live `Game` instance from `serialize_full_state()` output."""
        player_payloads = payload["players"]
        player_names = [str(player["name"]) for player in player_payloads]
        player_roles = {str(player["name"]): str(player["role"]) for player in player_payloads}
        game = cls(
            player_names=player_names,
            player_roles=player_roles,
            starting_cash=int(payload["starting_cash"]),
            dice=Dice.from_dict(payload["dice"]),
        )

        player_lookup = {player.name: player for player in game.players}
        for player_data in player_payloads:
            player = player_lookup[str(player_data["name"])]
            player.cash = int(player_data["cash"])
            player.position = int(player_data["position"])
            player.in_jail = bool(player_data["in_jail"])
            player.jail_turns = int(player_data["jail_turns"])
            player.get_out_of_jail_cards = int(player_data["get_out_of_jail_cards"])
            player.is_bankrupt = bool(player_data["is_bankrupt"])
            player.properties.clear()

        for space in game.board.spaces:
            if isinstance(space, PropertySpace):
                space.release_to_bank()

        for property_data in payload["board"]["properties"]:
            property_space = game.board.get_space(int(property_data["index"]))
            if not isinstance(property_space, PropertySpace):
                continue
            owner_name = property_data.get("owner_name")
            if owner_name is not None:
                property_space.assign_owner(player_lookup[str(owner_name)])
            property_space.mortgaged = bool(property_data.get("mortgaged", False))
            if isinstance(property_space, StreetPropertySpace):
                property_space.building_count = int(property_data.get("building_count", 0))

        game._restore_deck_state(game.board.chance_deck, payload["board"]["chance_deck"])
        game._restore_deck_state(game.board.community_chest_deck, payload["board"]["community_chest_deck"])

        runtime = payload["runtime"]
        game.turn_counter = int(runtime["turn_counter"])
        game.houses_remaining = int(runtime["houses_remaining"])
        game.hotels_remaining = int(runtime["hotels_remaining"])
        game.auction_bids_by_space = {int(index): {str(name): int(bid) for name, bid in bids.items()} for index, bids in runtime["auction_bids_by_space"].items()}
        game.current_turn_phase = str(runtime["current_turn_phase"])
        game.debug_next_rolls_by_player = {
            str(player_name): [tuple(int(value) for value in roll) for roll in rolls]
            for player_name, rolls in runtime.get("debug_next_rolls_by_player", {}).items()
        }

        current_player_name = runtime.get("current_player_name")
        if current_player_name is None:
            game.current_player_index = 0
        else:
            active_names = [player.name for player in game.active_players]
            game.current_player_index = active_names.index(str(current_player_name)) if active_names and str(current_player_name) in active_names else 0

        purchase_data = runtime.get("pending_purchase_decision")
        if purchase_data is not None:
            game.pending_purchase_decision = PurchaseDecisionState(
                player_name=str(purchase_data["player_name"]),
                property_index=int(purchase_data["property_index"]),
                property_name=str(purchase_data["property_name"]),
                price=int(purchase_data["price"]),
            )

        auction_data = runtime.get("pending_auction")
        if auction_data is not None:
            game.pending_auction = PendingAuctionState(
                property_index=int(auction_data["property_index"]),
                property_name=str(auction_data["property_name"]),
                eligible_player_names=[str(name) for name in auction_data["eligible_player_names"]],
                active_player_names=[str(name) for name in auction_data["active_player_names"]],
                current_bid=int(auction_data["current_bid"]),
                current_winner_name=None if auction_data.get("current_winner_name") is None else str(auction_data["current_winner_name"]),
                current_bidder_index=int(auction_data["current_bidder_index"]),
            )

        jail_data = runtime.get("pending_jail_decision")
        if jail_data is not None:
            game.pending_jail_decision = PendingJailDecisionState(
                player_name=str(jail_data["player_name"]),
                available_actions=[str(action) for action in jail_data["available_actions"]],
            )

        property_action_data = runtime.get("pending_property_action")
        if property_action_data is not None:
            game.pending_property_action = PendingPropertyActionState(
                action_type=str(property_action_data["action_type"]),
                player_name=str(property_action_data["player_name"]),
                property_name=str(property_action_data["property_name"]),
                property_index=int(property_action_data["property_index"]),
            )

        game._blocked_property_action_requests = {
            (
                str(item["player_name"]),
                str(item["action_type"]),
                int(item["property_index"]),
            )
            for item in runtime.get("blocked_property_action_requests", [])
        }
        game._blocked_trade_offer_signatures = {str(item) for item in runtime.get("blocked_trade_offer_signatures", [])}
        game._trade_proposals_this_turn = {
            (str(item["player_name"]), str(item["turn_phase"]))
            for item in runtime.get("trade_proposals_this_turn", [])
        }

        trade_data = runtime.get("pending_trade_decision")
        if trade_data is not None:
            game.pending_trade_decision = PendingTradeDecisionState(
                trade_offer=game.deserialize_trade_offer(trade_data),
                counter_count=int(runtime.get("pending_trade_counter_count", 0)),
            )

        continuation_data = runtime.get("pending_turn_continuation")
        if continuation_data is not None:
            game._pending_turn_continuation = TurnContinuation(
                player_name=str(continuation_data["player_name"]),
                doubles_in_row=int(continuation_data["doubles_in_row"]),
                rolled_double=bool(continuation_data["rolled_double"]),
            )

        game._recalculate_building_supply_from_board()

        return game

    def save_to_file(self, file_path: str) -> None:
        """Persist the authoritative game state to a JSON file."""
        import json

        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(self.serialize_full_state(), handle, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> Game:
        """Load a previously saved game state from disk."""
        import json

        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_serialized_state(payload)

    @staticmethod
    def _restore_deck_state(deck: Any, card_names: list[str]) -> None:
        """Restore deck order by card name while preserving duplicate-name card copies."""
        card_lookup: dict[str, list[Any]] = {}
        for card in list(deck):
            card_lookup.setdefault(card.name, []).append(card)
        deck.clear()
        for name in card_names:
            matching_cards = card_lookup.get(str(name))
            if not matching_cards:
                raise KeyError(f"Card named {name!r} was not found while restoring deck state.")
            deck.append(matching_cards.pop(0))

    def serialize_trade_offer(self, trade_offer: TradeOffer) -> dict[str, Any]:
        """Encode a trade offer into the stable payload used by UI and transport code."""
        return {
            "proposer_name": trade_offer.proposer.name,
            "receiver_name": trade_offer.receiver.name,
            "proposer_cash": trade_offer.proposer_cash,
            "receiver_cash": trade_offer.receiver_cash,
            "proposer_property_names": [space.name for space in trade_offer.proposer_properties],
            "receiver_property_names": [space.name for space in trade_offer.receiver_properties],
            "proposer_jail_cards": trade_offer.proposer_jail_cards,
            "receiver_jail_cards": trade_offer.receiver_jail_cards,
            "note": trade_offer.note,
        }

    def deserialize_trade_offer(self, payload: Mapping[str, Any]) -> TradeOffer:
        """Decode a serialized trade offer back into live player and property references."""
        proposer = self._get_player_by_name(str(payload["proposer_name"]))
        receiver = self._get_player_by_name(str(payload["receiver_name"]))
        return TradeOffer(
            proposer=proposer,
            receiver=receiver,
            proposer_cash=int(payload.get("proposer_cash", 0)),
            receiver_cash=int(payload.get("receiver_cash", 0)),
            proposer_properties=[self._find_property(proposer, str(name)) for name in payload.get("proposer_property_names", [])],
            receiver_properties=[self._find_property(receiver, str(name)) for name in payload.get("receiver_property_names", [])],
            proposer_jail_cards=int(payload.get("proposer_jail_cards", 0)),
            receiver_jail_cards=int(payload.get("receiver_jail_cards", 0)),
            note=str(payload.get("note", "")),
        )

    @staticmethod
    def _trade_offer_signature_from_payload(payload: Mapping[str, Any]) -> str:
        """Canonicalize a serialized trade payload into a stable deduplication key."""
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

    def _trade_offer_signature(self, trade_offer: TradeOffer) -> str:
        """Build the deduplication key for one live trade offer instance."""
        return self._trade_offer_signature_from_payload(self.serialize_trade_offer(trade_offer))

    def is_trade_offer_blocked(self, trade_offer: TradeOffer) -> bool:
        """Whether an equivalent trade payload has already been rejected this turn."""
        return self._trade_offer_signature(trade_offer) in self._blocked_trade_offer_signatures

    @staticmethod
    def _trade_phase_key(turn_phase: str | None) -> str | None:
        """Map runtime turn phases to the subset where free trade offers are allowed."""
        if turn_phase in {PRE_ROLL_PHASE, POST_ROLL_PHASE}:
            return str(turn_phase)
        return None

    def _can_propose_trade_this_phase(self, player_name: str) -> bool:
        """Whether the player still has an unused trade proposal slot this phase."""
        phase_key = self._trade_phase_key(self.current_turn_phase)
        if phase_key is None:
            return False
        return (player_name, phase_key) not in self._trade_proposals_this_turn

    def _record_trade_proposal(self, player_name: str) -> None:
        """Consume the caller's one-trade-per-phase allowance."""
        phase_key = self._trade_phase_key(self.current_turn_phase)
        if phase_key is None:
            raise ValueError("Trade proposals are only available before the roll or after the move is resolved.")
        self._trade_proposals_this_turn.add((player_name, phase_key))

    def _block_trade_offer(self, trade_offer: TradeOffer) -> None:
        """Remember a rejected offer signature so the same proposal cannot be re-sent this turn."""
        self._blocked_trade_offer_signatures.add(self._trade_offer_signature(trade_offer))

    def _clear_trade_turn_state(self) -> None:
        """Clear per-turn trade limits and rejection memory when the turn fully ends."""
        self._blocked_trade_offer_signatures.clear()
        self._trade_proposals_this_turn.clear()

    def _is_property_action_request_blocked(self, player_name: str, action_type: str, property_index: int) -> bool:
        """Whether a confirmable property action was already cancelled earlier this turn."""
        return (player_name, action_type, property_index) in self._blocked_property_action_requests

    def _block_property_action_request(self, player_name: str, action_type: str, property_index: int) -> None:
        """Remember a cancelled property action so the same prompt is not reopened immediately."""
        self._blocked_property_action_requests.add((player_name, action_type, property_index))

    def _clear_blocked_property_action_requests(self) -> None:
        """Clear all per-turn property-action cancellation blockers."""
        self._blocked_property_action_requests.clear()

    def execute_serialized_action(
        self,
        action_payload: Mapping[str, Any],
        *,
        bid_amount: int | None = None,
        trade_offer_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one legal action from serialized API payloads and return a serialized result."""
        trade_offer = None if trade_offer_payload is None else self.deserialize_trade_offer(trade_offer_payload)
        requested_action = LegalActionOption.from_dict(action_payload)
        result = self.execute_legal_action(
            self._resolve_serialized_legal_action(requested_action),
            bid_amount=bid_amount,
            trade_offer=trade_offer,
        )
        return result.to_dict()

    def _resolve_serialized_legal_action(self, requested_action: LegalActionOption) -> LegalActionOption:
        """Map a deserialized action payload back to the live action instance for this state."""
        plan = self.get_turn_plan(requested_action.actor_name)
        if requested_action in plan.legal_actions:
            return requested_action

        matches = [
            legal_action
            for legal_action in plan.legal_actions
            if legal_action.action_type == requested_action.action_type
            and legal_action.actor_name == requested_action.actor_name
            and legal_action.handler_name == requested_action.handler_name
            and legal_action.property_name == requested_action.property_name
            and legal_action.target_player_name == requested_action.target_player_name
            and legal_action.fixed_choice == requested_action.fixed_choice
            and legal_action.min_bid == requested_action.min_bid
            and legal_action.max_bid == requested_action.max_bid
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"{requested_action.action_type} is not currently legal for {requested_action.actor_name}.")
        raise ValueError(
            f"Serialized action for {requested_action.actor_name} matched multiple legal actions for {requested_action.action_type}."
        )

    def get_turn_plan(self, player_name: str | None = None) -> TurnPlanView:
        """Describe what one player may legally do in the current authoritative state."""
        player = self.current_player if player_name is None else self._get_player_by_name(player_name)
        pending_action = self.get_pending_action()

        if player.is_bankrupt:
            return TurnPlanView(
                player_name=player.name,
                player_role=player.role,
                turn_phase=self.current_turn_phase,
                is_current_player=player is self.current_player,
                has_pending_action=False,
                pending_action=None,
                legal_actions=(),
                reason=f"{player.name} is bankrupt and cannot act.",
            )

        if pending_action is not None and pending_action.player_name != player.name:
            return TurnPlanView(
                player_name=player.name,
                player_role=player.role,
                turn_phase=IN_TURN_PHASE,
                is_current_player=player is self.current_player,
                has_pending_action=False,
                pending_action=pending_action,
                legal_actions=(),
                reason=f"Pending action belongs to {pending_action.player_name}.",
            )

        if pending_action is not None:
            return TurnPlanView(
                player_name=player.name,
                player_role=player.role,
                turn_phase=IN_TURN_PHASE,
                is_current_player=player is self.current_player,
                has_pending_action=True,
                pending_action=pending_action,
                legal_actions=self._legal_actions_for_pending_action(player, pending_action),
            )

        if player is not self.current_player:
            return TurnPlanView(
                player_name=player.name,
                player_role=player.role,
                turn_phase=self.current_turn_phase,
                is_current_player=False,
                has_pending_action=False,
                pending_action=None,
                legal_actions=(),
                reason=f"It is currently {self.current_player.name}'s turn.",
            )

        if self.current_turn_phase == PRE_ROLL_PHASE:
            legal_actions = self._legal_actions_for_pre_roll(player)
        elif self.current_turn_phase == POST_ROLL_PHASE:
            legal_actions = self._legal_actions_for_post_roll(player)
        else:
            legal_actions = ()

        return TurnPlanView(
            player_name=player.name,
            player_role=player.role,
            turn_phase=self.current_turn_phase,
            is_current_player=True,
            has_pending_action=False,
            pending_action=None,
            legal_actions=legal_actions,
            reason=None if legal_actions else "The turn is currently resolving an in-turn action.",
        )

    def _property_action_cash_effect(self, action_type: str, property_space: PropertySpace) -> int:
        """Return the signed cash delta shown to the player for a property action prompt."""
        if action_type == "build":
            return -property_space.house_cost
        if action_type == "sell_building":
            return property_space.house_cost // 2
        if action_type == "mortgage":
            return property_space.mortgage_value
        if action_type == "unmortgage":
            return -(property_space.mortgage_value + int(property_space.mortgage_value * 0.10 + 0.9999))
        raise ValueError(f"Unsupported property action: {action_type}")

    def _property_action_description(self, action_type: str, property_space: PropertySpace) -> str:
        """Return human-readable copy describing one pending property action."""
        if action_type == "build":
            return f"Build one house or upgrade to a hotel on {property_space.name}."
        if action_type == "sell_building":
            return f"Sell one building from {property_space.name}."
        if action_type == "mortgage":
            return f"Mortgage {property_space.name} for cash from the bank."
        if action_type == "unmortgage":
            return f"Pay to remove the mortgage from {property_space.name}."
        raise ValueError(f"Unsupported property action: {action_type}")

    @staticmethod
    def _property_action_label(action_type: str) -> str:
        """Map internal property action types to short UI verb phrases."""
        labels = {
            "build": "build on",
            "sell_building": "sell a building from",
            "mortgage": "mortgage",
            "unmortgage": "remove the mortgage from",
        }
        return labels.get(action_type, action_type.replace("_", " "))

    @staticmethod
    def _jail_action_description(choice: str) -> str:
        """Map jail decision identifiers to player-facing action text."""
        if choice == "roll":
            return "Roll for doubles to leave jail."
        if choice == "pay_fine":
            return f"Pay ${JAIL_FINE} to leave jail."
        if choice == "use_card":
            return "Use a Get Out of Jail Free card."
        return f"Choose {choice}."

    def has_pending_interaction(self) -> bool:
        """Whether the interactive engine is waiting on any explicit player decision."""
        return any(
            state is not None
            for state in (
                self.pending_jail_decision,
                self.pending_purchase_decision,
                self.pending_auction,
                self.pending_property_action,
                self.pending_trade_decision,
            )
        )

    def _legal_actions_for_pending_action(
        self,
        player: Player,
        pending_action: PendingActionView,
    ) -> tuple[LegalActionOption, ...]:
        """Build the legal action list for the currently pending prompt."""
        if pending_action.action_type == "property_purchase":
            return (
                LegalActionOption(
                    action_type="buy_property",
                    actor_name=player.name,
                    actor_role=player.role,
                    handler_name="resolve_property_decision",
                    description=f"Buy {pending_action.property_name} for ${pending_action.price}.",
                    property_name=pending_action.property_name,
                    fixed_choice="buy",
                ),
                LegalActionOption(
                    action_type="decline_property",
                    actor_name=player.name,
                    actor_role=player.role,
                    handler_name="resolve_property_decision",
                    description=f"Pass on {pending_action.property_name} and send it to auction.",
                    property_name=pending_action.property_name,
                    fixed_choice="decline",
                ),
            )

        if pending_action.action_type == "auction" and pending_action.auction is not None:
            return (
                LegalActionOption(
                    action_type="place_auction_bid",
                    actor_name=player.name,
                    actor_role=player.role,
                    handler_name="submit_auction_bid",
                    description=f"Bid at least ${pending_action.auction.minimum_bid} for {pending_action.property_name}.",
                    property_name=pending_action.property_name,
                    min_bid=pending_action.auction.minimum_bid,
                    max_bid=player.cash,
                ),
                LegalActionOption(
                    action_type="pass_auction",
                    actor_name=player.name,
                    actor_role=player.role,
                    handler_name="submit_auction_bid",
                    description=f"Pass on {pending_action.property_name} in the auction.",
                    property_name=pending_action.property_name,
                    fixed_choice="pass",
                ),
            )

        if pending_action.action_type == "jail_decision":
            return tuple(
                LegalActionOption(
                    action_type=f"jail_{choice}",
                    actor_name=player.name,
                    actor_role=player.role,
                    handler_name="resolve_jail_decision",
                    description=self._jail_action_description(choice),
                    fixed_choice=choice,
                )
                for choice in pending_action.available_actions
            )

        if pending_action.action_type == "property_action":
            return (
                LegalActionOption(
                    action_type="confirm_property_action",
                    actor_name=player.name,
                    actor_role=player.role,
                    handler_name="resolve_property_action",
                    description=f"Go through with {self._property_action_label(pending_action.property_action.action_type)} {pending_action.property_name}.",
                    property_name=pending_action.property_name,
                    fixed_choice="confirm",
                ),
                LegalActionOption(
                    action_type="cancel_property_action",
                    actor_name=player.name,
                    actor_role=player.role,
                    handler_name="resolve_property_action",
                    description=f"Keep {pending_action.property_name} as it is.",
                    property_name=pending_action.property_name,
                    fixed_choice="cancel",
                ),
            )

        if pending_action.action_type == "trade_decision":
            actions = [
                LegalActionOption(
                    action_type="accept_trade",
                    actor_name=player.name,
                    actor_role=player.role,
                    handler_name="resolve_trade_decision",
                    description=f"Accept the trade from {pending_action.trade.proposer_name}.",
                    target_player_name=pending_action.trade.proposer_name,
                    fixed_choice="accept",
                ),
                LegalActionOption(
                    action_type="reject_trade",
                    actor_name=player.name,
                    actor_role=player.role,
                    handler_name="resolve_trade_decision",
                    description=f"Decline the trade from {pending_action.trade.proposer_name}.",
                    target_player_name=pending_action.trade.proposer_name,
                    fixed_choice="reject",
                ),
            ]
            if self.pending_trade_decision is not None and self.pending_trade_decision.counter_count < 1:
                actions.append(
                    LegalActionOption(
                        action_type="counter_trade",
                        actor_name=player.name,
                        actor_role=player.role,
                        handler_name="counter_trade_interactive",
                        description=f"Counter the trade from {pending_action.trade.proposer_name}.",
                        target_player_name=pending_action.trade.proposer_name,
                    )
                )
            return tuple(actions)

        return ()

    def _legal_actions_for_pre_roll(self, player: Player) -> tuple[LegalActionOption, ...]:
        """Return legal actions available before the player rolls."""
        actions: list[LegalActionOption] = [
            LegalActionOption(
                action_type="start_turn",
                actor_name=player.name,
                actor_role=player.role,
                handler_name="start_turn_interactive",
                description=f"Roll the dice for {player.name}.",
            )
        ]

        actions.extend(self._legal_actions_for_between_roll_windows(player))
        return tuple(actions)

    def _legal_actions_for_post_roll(self, player: Player) -> tuple[LegalActionOption, ...]:
        """Return legal actions available after movement has fully resolved."""
        actions = list(self._legal_actions_for_between_roll_windows(player))
        actions.append(
            LegalActionOption(
                action_type="end_turn",
                actor_name=player.name,
                actor_role=player.role,
                handler_name="end_turn_interactive",
                description=f"Finish {player.name}'s turn.",
            )
        )
        return tuple(actions)

    def _legal_actions_for_between_roll_windows(self, player: Player) -> tuple[LegalActionOption, ...]:
        """Return free-action options shared by pre-roll and post-roll phases."""
        actions: list[LegalActionOption] = []

        for property_space in player.properties:
            if isinstance(property_space, StreetPropertySpace):
                can_build, _ = MonopolyRules.can_build_house(self.board, player, property_space)
                if can_build and not self._is_property_action_request_blocked(player.name, "build", property_space.index):
                    actions.append(
                        LegalActionOption(
                            action_type="request_build",
                            actor_name=player.name,
                            actor_role=player.role,
                            handler_name="request_property_action",
                            description=f"Build on {property_space.name}.",
                            property_name=property_space.name,
                            fixed_choice="build",
                        )
                    )

                can_sell, _ = MonopolyRules.can_sell_building(self.board, player, property_space)
                if can_sell and not self._is_property_action_request_blocked(player.name, "sell_building", property_space.index):
                    actions.append(
                        LegalActionOption(
                            action_type="request_sell_building",
                            actor_name=player.name,
                            actor_role=player.role,
                            handler_name="request_property_action",
                            description=f"Sell a building on {property_space.name}.",
                            property_name=property_space.name,
                            fixed_choice="sell_building",
                        )
                    )

            can_mortgage, _ = MonopolyRules.can_mortgage(self.board, player, property_space)
            if can_mortgage and not self._is_property_action_request_blocked(player.name, "mortgage", property_space.index):
                actions.append(
                    LegalActionOption(
                        action_type="request_mortgage",
                        actor_name=player.name,
                        actor_role=player.role,
                        handler_name="request_property_action",
                        description=f"Mortgage {property_space.name}.",
                        property_name=property_space.name,
                        fixed_choice="mortgage",
                    )
                )

            if (
                property_space.mortgaged
                and player.cash >= MonopolyRules.unmortgage_cost(property_space)
                and not self._is_property_action_request_blocked(player.name, "unmortgage", property_space.index)
            ):
                actions.append(
                    LegalActionOption(
                        action_type="request_unmortgage",
                        actor_name=player.name,
                        actor_role=player.role,
                        handler_name="request_property_action",
                        description=f"Remove the mortgage from {property_space.name}.",
                        property_name=property_space.name,
                        fixed_choice="unmortgage",
                    )
                )

        if self._can_propose_trade_this_phase(player.name):
            for other_player in self.active_players:
                if other_player is player:
                    continue
                actions.append(
                    LegalActionOption(
                        action_type="propose_trade",
                        actor_name=player.name,
                        actor_role=player.role,
                        handler_name="propose_trade_interactive",
                        description=f"Offer a trade to {other_player.name}.",
                        target_player_name=other_player.name,
                    )
                )

        return tuple(actions)

    def _play_interactive_turn(self, player: Player, doubles_in_row: int, opening_messages: list[str]) -> InteractionResult:
        """Drive the interactive turn loop until another prompt or phase boundary is reached."""
        messages = list(opening_messages)
        self._interactive_flow_active = True
        self._clear_blocked_property_action_requests()
        self.current_turn_phase = IN_TURN_PHASE
        try:
            if player.is_bankrupt:
                messages.append(f"{player.name} is bankrupt and cannot take a turn.")
                return self._complete_interactive_turn(messages)

            running_doubles = doubles_in_row
            while True:
                if player.in_jail:
                    jail_result = self._handle_jail_turn_interactive(player)
                    messages.extend(jail_result.messages)
                    if self.has_pending_interaction():
                        return self._build_interaction_result(messages)
                    if jail_result.turn_consumed:
                        return self._conclude_roll_sequence(player, messages)

                roll = self._roll_for_player(player.name)
                messages.append(f"{player.name} rolls {roll.die_one} and {roll.die_two} (total {roll.total}).")

                running_doubles = running_doubles + 1 if roll.is_double else 0
                if running_doubles == 3:
                    self.send_player_to_jail(player)
                    messages.append(f"{player.name} rolled doubles three times and goes to jail.")
                    return self._conclude_roll_sequence(player, messages)

                self._pending_turn_continuation = TurnContinuation(
                    player_name=player.name,
                    doubles_in_row=running_doubles,
                    rolled_double=roll.is_double,
                )
                messages.extend(self.move_player_by(player, roll.total))
                messages.extend(self.resolve_current_space(player, allow_property_purchase=False, dice_total=roll.total, interactive=True))

                if self.has_pending_interaction():
                    return self._build_interaction_result(messages)
                if player.in_jail or player.is_bankrupt or not roll.is_double:
                    return self._conclude_roll_sequence(player, messages)

                messages.append(f"{player.name} rolled doubles and takes another turn.")
        finally:
            self._interactive_flow_active = False

    def _handle_jail_turn_interactive(self, player: Player) -> JailTurnResult:
        """Open the jail-choice prompt used by human, online, and AI interactive clients."""
        available_actions = ["roll"]
        if player.can_afford(JAIL_FINE):
            available_actions.append("pay_fine")
        if player.get_out_of_jail_cards > 0:
            available_actions.append("use_card")

        self.pending_jail_decision = PendingJailDecisionState(
            player_name=player.name,
            available_actions=available_actions,
        )
        messages = [
            f"{player.name} starts the turn in jail.",
            f"{player.name} must choose how to leave jail: {', '.join(available_actions)}.",
        ]
        return JailTurnResult(messages=messages, turn_consumed=False)

    def _build_interaction_result(self, messages: list[str]) -> InteractionResult:
        """Package narration with the current public game snapshot and pending action."""
        pending_action = self.get_pending_action()
        return InteractionResult(messages=tuple(messages), game_view=self.get_game_view(), pending_action=pending_action)

    def _conclude_roll_sequence(self, player: Player, messages: list[str]) -> InteractionResult:
        """Finish one roll sequence and transition into the post-roll action window."""
        self._pending_turn_continuation = None
        if player.is_bankrupt:
            return self._complete_interactive_turn(messages)

        self.current_turn_phase = POST_ROLL_PHASE
        messages.append(f"{player.name} has reached the post-roll phase.")
        return InteractionResult(messages=tuple(messages), game_view=self.get_game_view(), pending_action=None)

    def _complete_interactive_turn(self, messages: list[str]) -> InteractionResult:
        """Advance turn order, clear per-turn interaction state, and expose the next snapshot."""
        self.turn_counter += 1
        self._clear_blocked_property_action_requests()
        self._clear_trade_turn_state()
        self._pending_turn_continuation = None
        if len(self.active_players) > 1:
            self.next_player()
        else:
            self.current_turn_phase = PRE_ROLL_PHASE
        return InteractionResult(messages=tuple(messages), game_view=self.get_game_view(), pending_action=None)

    def _continue_after_pending_interaction(self, messages: list[str]) -> InteractionResult:
        """Resume the interrupted turn flow after a pending prompt has been resolved."""
        continuation = self._pending_turn_continuation
        self._pending_turn_continuation = None
        if continuation is None:
            return self._build_interaction_result(messages)

        player = self._get_player_by_name(continuation.player_name)
        if player.in_jail or player.is_bankrupt or not continuation.rolled_double:
            return self._conclude_roll_sequence(player, messages)

        messages.append(f"{player.name} rolled doubles and takes another turn.")
        return self._play_interactive_turn(player, doubles_in_row=continuation.doubles_in_row, opening_messages=messages)

    def _open_interactive_auction(self, property_space: PropertySpace) -> list[str]:
        """Create a pending auction prompt for a declined property purchase."""
        eligible_players = [player.name for player in self.active_players if player.cash > 0]
        messages = [f"Auction begins for {property_space.name}."]
        if not eligible_players:
            messages.append(f"No players can bid on {property_space.name}. The bank keeps the property.")
            return messages

        self.pending_auction = PendingAuctionState(
            property_index=property_space.index,
            property_name=property_space.name,
            eligible_player_names=list(eligible_players),
            active_player_names=list(eligible_players),
        )
        messages.append(f"{self.pending_auction.active_player_names[0]} is first to act in the auction.")
        return messages

    def _auction_is_ready_to_resolve(self, auction: PendingAuctionState) -> bool:
        """Whether only the current winner remains able or willing to keep bidding."""
        if not auction.active_player_names:
            return True
        if auction.current_winner_name is None:
            return False
        challengers = [name for name in auction.active_player_names if name != auction.current_winner_name]
        return not challengers

    def _get_pending_auction_minimum_bid(self, auction: PendingAuctionState) -> int:
        """Return the next legal bid amount for the live interactive auction."""
        if auction.current_bid <= 0:
            return 1
        property_space = self.board.get_space(auction.property_index)
        return auction.current_bid + self._get_auction_minimum_raise(property_space.price, auction.current_bid)

    def _get_auction_minimum_raise(self, property_price: int, current_bid: int) -> int:
        """Compute the dynamic minimum raise used by all interactive auctions."""
        reference_value = max(property_price, current_bid)
        raw_raise = reference_value * 0.05
        rounded_raise = int(math.ceil(raw_raise / 5.0) * 5)
        return max(5, min(50, rounded_raise))

    def _resolve_interactive_auction(self) -> list[str]:
        """Close the live auction, transfer ownership if needed, and clear pending state."""
        if self.pending_auction is None:
            return []

        auction = self.pending_auction
        property_space = self.board.get_space(auction.property_index)
        messages: list[str] = []
        if auction.current_winner_name is None:
            messages.append(f"No bids were placed for {auction.property_name}. The bank keeps the property.")
        else:
            winner = self._get_player_by_name(auction.current_winner_name)
            winner.pay(auction.current_bid)
            property_space.assign_owner(winner)
            messages.append(f"{winner.name} wins the auction for {auction.property_name} at ${auction.current_bid}.")
        self.pending_auction = None
        return messages

    def _get_player_by_name(self, player_name: str) -> Player:
        """Resolve a player name into the live player object for this game."""
        for player in self.players:
            if player.name == player_name:
                return player
        raise ValueError(f"Unknown player: {player_name}")

    def _ensure_free_action_window(self, player: Player) -> None:
        """Require that free actions happen only on the active player's open action window."""
        if self.has_pending_interaction():
            raise ValueError("Finish the pending decision before starting another interactive action.")
        if player is not self.current_player:
            raise ValueError(f"It is currently {self.current_player.name}'s turn.")
        if self.current_turn_phase not in {PRE_ROLL_PHASE, POST_ROLL_PHASE}:
            raise ValueError("These actions are only available before the roll or after the move is resolved.")

    def _find_property(self, player: Player, property_name: str) -> PropertySpace:
        """Look up one property currently owned by the given player."""
        for property_space in player.properties:
            if property_space.name == property_name:
                return property_space
        raise ValueError(f"{player.name} does not own {property_name}.")

    def _find_street(self, player: Player, property_name: str) -> StreetPropertySpace:
        """Look up one owned street property, rejecting railroads and utilities."""
        property_space = self._find_property(player, property_name)
        if not isinstance(property_space, StreetPropertySpace):
            raise ValueError(f"{property_name} is not a street property.")
        return property_space
