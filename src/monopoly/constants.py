from __future__ import annotations

"""Shared constants for Monopoly rules, board layout, and UI rendering.

The values in this module define the canonical game configuration used across the
engine, frontend, online runtime, and agent stack.
"""

# Player and turn-flow constraints.
MIN_PLAYERS = 2
MAX_PLAYERS = 6
HUMAN_ROLE = "human"
AI_ROLE = "ai"
PLAYER_ROLES = {HUMAN_ROLE, AI_ROLE}
PRE_ROLL_PHASE = "pre_roll"
IN_TURN_PHASE = "in_turn"
POST_ROLL_PHASE = "post_roll"
TURN_PHASES = {PRE_ROLL_PHASE, IN_TURN_PHASE, POST_ROLL_PHASE}

# Core economy and board positions.
STARTING_CASH = 1500
GO_SALARY = 200
BOARD_SIZE = 40
JAIL_INDEX = 10
GO_TO_JAIL_INDEX = 30
INCOME_TAX_AMOUNT = 200
LUXURY_TAX_AMOUNT = 100
JAIL_FINE = 50
UNMORTGAGE_INTEREST_RATE = 0.10
HOUSE_LIMIT = 4
HOTEL_BUILDING_COUNT = 5
BANK_HOUSES = 32
BANK_HOTELS = 12

# Deck identifiers.
CHANCE = "chance"
COMMUNITY_CHEST = "community_chest"

# Property-group structure and frontend colors.
COLOR_GROUP_SIZES = {
    "Brown": 2,
    "Light Blue": 3,
    "Pink": 3,
    "Orange": 3,
    "Red": 3,
    "Yellow": 3,
    "Green": 3,
    "Dark Blue": 2,
}

BOARD_COLOR_GROUPS = {
    "Brown": "#7D5330",
    "Light Blue": "#9CD8F7",
    "Pink": "#D96AA7",
    "Orange": "#F39A34",
    "Red": "#D94A43",
    "Yellow": "#E8D14B",
    "Green": "#2E8B57",
    "Dark Blue": "#24539B",
}
