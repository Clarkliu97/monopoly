from __future__ import annotations

"""Trade offer validation and execution helpers.

Interactive trade prompts in the game engine eventually funnel through this
object so both human and AI trade flows share one validation and settlement path.
"""

from dataclasses import dataclass, field

from monopoly.constants import UNMORTGAGE_INTEREST_RATE
from monopoly.spaces import PropertySpace


@dataclass(slots=True)
class TradeOffer:
    """Proposed asset exchange between two players."""

    proposer: object
    receiver: object
    proposer_cash: int = 0
    receiver_cash: int = 0
    proposer_properties: list[PropertySpace] = field(default_factory=list)
    receiver_properties: list[PropertySpace] = field(default_factory=list)
    proposer_jail_cards: int = 0
    receiver_jail_cards: int = 0
    note: str = ""

    def validate(self) -> list[str]:
        """Return validation errors for illegal or unaffordable trade contents."""
        errors: list[str] = []

        if self.proposer is self.receiver:
            errors.append("A player cannot trade with themselves.")
        if self.proposer_cash < 0 or self.receiver_cash < 0:
            errors.append("Trade cash values must be zero or positive.")
        if self.proposer_jail_cards < 0 or self.receiver_jail_cards < 0:
            errors.append("Trade jail card counts must be zero or positive.")
        if getattr(self.proposer, "cash", 0) < self.proposer_cash:
            errors.append(f"{self.proposer.name} does not have enough cash for this trade.")
        if getattr(self.receiver, "cash", 0) < self.receiver_cash:
            errors.append(f"{self.receiver.name} does not have enough cash for this trade.")
        if getattr(self.proposer, "get_out_of_jail_cards", 0) < self.proposer_jail_cards:
            errors.append(f"{self.proposer.name} does not have enough Get Out of Jail cards.")
        if getattr(self.receiver, "get_out_of_jail_cards", 0) < self.receiver_jail_cards:
            errors.append(f"{self.receiver.name} does not have enough Get Out of Jail cards.")

        for property_space in self.proposer_properties:
            if property_space.owner is not self.proposer:
                errors.append(f"{self.proposer.name} does not own {property_space.name}.")
        for property_space in self.receiver_properties:
            if property_space.owner is not self.receiver:
                errors.append(f"{self.receiver.name} does not own {property_space.name}.")

        return errors

    def execute(self) -> list[str]:
        """Apply a validated trade and return the narration describing each transfer."""
        errors = self.validate()
        if errors:
            raise ValueError(" ".join(errors))

        messages = [f"Trade accepted between {self.proposer.name} and {self.receiver.name}."]

        self.proposer.pay(self.proposer_cash)
        self.receiver.receive(self.proposer_cash)
        if self.proposer_cash:
            messages.append(f"{self.proposer.name} pays ${self.proposer_cash} to {self.receiver.name}.")

        self.receiver.pay(self.receiver_cash)
        self.proposer.receive(self.receiver_cash)
        if self.receiver_cash:
            messages.append(f"{self.receiver.name} pays ${self.receiver_cash} to {self.proposer.name}.")

        self.proposer.get_out_of_jail_cards -= self.proposer_jail_cards
        self.receiver.get_out_of_jail_cards += self.proposer_jail_cards
        self.receiver.get_out_of_jail_cards -= self.receiver_jail_cards
        self.proposer.get_out_of_jail_cards += self.receiver_jail_cards

        for property_space in self.proposer_properties:
            self._transfer_property(property_space, self.proposer, self.receiver, messages)
        for property_space in self.receiver_properties:
            self._transfer_property(property_space, self.receiver, self.proposer, messages)

        return messages

    @staticmethod
    def _transfer_property(property_space: PropertySpace, current_owner: object, new_owner: object, messages: list[str]) -> None:
        """Move one property between players, including mortgage-interest handling."""
        current_owner.remove_property(property_space)
        new_owner.add_property(property_space)
        property_space.owner = new_owner
        messages.append(f"{property_space.name} moves from {current_owner.name} to {new_owner.name}.")
        if property_space.mortgaged:
            interest_due = int(property_space.mortgage_value * UNMORTGAGE_INTEREST_RATE)
            new_owner.pay(interest_due)
            messages.append(
                f"{new_owner.name} pays ${interest_due} interest because {property_space.name} is mortgaged."
            )
