from __future__ import annotations

from dataclasses import dataclass

import pygame

from monopoly.api import BoardSpaceView, FrontendStateView
from monopoly.constants import BOARD_COLOR_GROUPS
from monopoly.gui.pygame_frontend.generated import build_panel_surface, build_stamp_surface, build_token_surface
from monopoly.gui.pygame_frontend import theme


COMMONWEALTH_LABELS: dict[str, list[str]] = {
    "Go": ["GO", "COLLECT", "$200"],
    "Old Kent Road": ["OLD KENT", "ROAD"],
    "Community Chest": ["COMMUNITY", "CHEST"],
    "Whitechapel Road": ["WHITECHAPEL", "ROAD"],
    "Income Tax": ["INCOME", "TAX"],
    "King's Cross Station": ["KING'S CROSS", "STATION"],
    "The Angel Islington": ["THE ANGEL", "ISLINGTON"],
    "Chance": ["CHANCE"],
    "Euston Road": ["EUSTON", "ROAD"],
    "Pentonville Road": ["PENTONVILLE", "ROAD"],
    "Jail / Just Visiting": ["IN JAIL", "JUST VISITING"],
    "Pall Mall": ["PALL", "MALL"],
    "Electric Company": ["ELECTRIC", "COMPANY"],
    "Whitehall": ["WHITEHALL"],
    "Northumberland Avenue": ["NORTHUMBERLAND", "AVENUE"],
    "Marylebone Station": ["MARYLEBONE", "STATION"],
    "Bow Street": ["BOW", "STREET"],
    "Marlborough Street": ["MARLBOROUGH", "STREET"],
    "Vine Street": ["VINE", "STREET"],
    "Free Parking": ["FREE", "PARKING"],
    "Strand": ["STRAND"],
    "Fleet Street": ["FLEET", "STREET"],
    "Trafalgar Square": ["TRAFALGAR", "SQUARE"],
    "Fenchurch St Station": ["FENCHURCH ST", "STATION"],
    "Leicester Square": ["LEICESTER", "SQUARE"],
    "Coventry Street": ["COVENTRY", "STREET"],
    "Water Works": ["WATER", "WORKS"],
    "Piccadilly": ["PICCADILLY"],
    "Go To Jail": ["GO TO", "JAIL"],
    "Regent Street": ["REGENT", "STREET"],
    "Oxford Street": ["OXFORD", "STREET"],
    "Bond Street": ["BOND", "STREET"],
    "Liverpool St Station": ["LIVERPOOL ST", "STATION"],
    "Park Lane": ["PARK", "LANE"],
    "Super Tax": ["SUPER", "TAX"],
    "Mayfair": ["MAYFAIR"],
}


@dataclass(frozen=True, slots=True)
class BoardLayout:
    board_rect: pygame.Rect
    corner_size: int = 0
    edge_depth: int = 0
    edge_step: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "corner_size", max(100, int(min(self.board_rect.width, self.board_rect.height) * 0.12)))
        object.__setattr__(self, "edge_depth", self.corner_size)
        edge_length = self.board_rect.width - (self.corner_size * 2)
        object.__setattr__(self, "edge_step", edge_length / 9.0)

    def space_rect(self, index: int) -> pygame.Rect:
        x = self.board_rect.left
        y = self.board_rect.top
        size = self.board_rect.width
        outer_right = x + size
        outer_bottom = y + size
        corner = self.corner_size
        step = self.edge_step

        if index == 0:
            return pygame.Rect(outer_right - corner, outer_bottom - corner, corner, corner)
        if 1 <= index <= 9:
            left = int(outer_right - corner - step * index)
            return pygame.Rect(left, outer_bottom - corner, int(step), corner)
        if index == 10:
            return pygame.Rect(x, outer_bottom - corner, corner, corner)
        if 11 <= index <= 19:
            offset = index - 11
            top = int(outer_bottom - corner - step * (offset + 1))
            return pygame.Rect(x, top, corner, int(step))
        if index == 20:
            return pygame.Rect(x, y, corner, corner)
        if 21 <= index <= 29:
            offset = index - 21
            left = int(x + corner + step * offset)
            return pygame.Rect(left, y, int(step), corner)
        if index == 30:
            return pygame.Rect(outer_right - corner, y, corner, corner)
        offset = index - 31
        top = int(y + corner + step * offset)
        return pygame.Rect(outer_right - corner, top, corner, int(step))

    def hit_test(self, position: tuple[int, int]) -> int | None:
        if not self.board_rect.collidepoint(position):
            return None
        for index in range(40):
            if self.space_rect(index).collidepoint(position):
                return index
        return None

    def token_anchor(self, index: int, token_slot: int) -> tuple[int, int]:
        rect = self.space_rect(index)
        columns = 2
        row = token_slot // columns
        column = token_slot % columns
        x = rect.left + 20 + column * 26
        y = rect.top + 20 + row * 26
        return x, y


class BoardRenderer:
    def __init__(self, board_rect: pygame.Rect) -> None:
        self.layout = BoardLayout(board_rect)
        self._cached_size: tuple[int, int] | None = None
        self._static_surface: pygame.Surface | None = None
        self._label_font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None
        self._title_font: pygame.font.Font | None = None

    def update_board_rect(self, board_rect: pygame.Rect) -> None:
        self.layout = BoardLayout(board_rect)
        self._cached_size = None
        self._static_surface = None

    def render(
        self,
        target: pygame.Surface,
        state: FrontendStateView,
        *,
        selected_space_index: int,
        hovered_space_index: int | None,
        hidden_player_names: set[str] | None = None,
        token_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._ensure_fonts()
        static_surface = self._build_static_surface(state)
        target.blit(static_surface, self.layout.board_rect.topleft)
        self._draw_dynamic_overlays(
            target,
            state,
            selected_space_index,
            hovered_space_index,
            hidden_player_names or set(),
            token_overrides or {},
        )

    def _ensure_fonts(self) -> None:
        if self._label_font is not None:
            return
        self._label_font = pygame.font.SysFont("segoeui", 14, bold=True)
        self._small_font = pygame.font.SysFont("segoeui", 11)
        self._title_font = pygame.font.SysFont("georgia", 34, bold=True)

    def _build_static_surface(self, state: FrontendStateView) -> pygame.Surface:
        size = self.layout.board_rect.size
        if self._static_surface is not None and self._cached_size == size:
            return self._static_surface

        static = pygame.Surface(size, pygame.SRCALPHA)
        static.blit(build_panel_surface(size, theme.BOARD_BACKGROUND, theme.BOARD_INNER), (0, 0))
        pygame.draw.rect(static, pygame.Color(theme.BOARD_LINE), static.get_rect(), width=4, border_radius=18)

        inner_margin = self.layout.corner_size + 8
        center_rect = pygame.Rect(
            inner_margin,
            inner_margin,
            size[0] - inner_margin * 2,
            size[1] - inner_margin * 2,
        )
        pygame.draw.rect(static, pygame.Color(theme.CENTER_FILL), center_rect, border_radius=18)
        pygame.draw.rect(static, pygame.Color("#B79F75"), center_rect, width=2, border_radius=18)

        title_surface = self._title_font.render("MONOPOLY", True, pygame.Color("#A44A3F"))
        rotated = pygame.transform.rotate(title_surface, 28)
        static.blit(rotated, rotated.get_rect(center=center_rect.center))
        subtitle_surface = self._label_font.render("LONDON EDITION", True, pygame.Color("#4A3827"))
        subtitle_rotated = pygame.transform.rotate(subtitle_surface, 28)
        static.blit(subtitle_rotated, subtitle_rotated.get_rect(center=(center_rect.centerx, center_rect.centery + 46)))

        for space in state.board_spaces:
            self._draw_space_base(static, space)

        self._cached_size = size
        self._static_surface = static
        return static

    def _draw_space_base(self, surface: pygame.Surface, space: BoardSpaceView) -> None:
        local_rect = self.layout.space_rect(space.index).move(-self.layout.board_rect.left, -self.layout.board_rect.top)
        fill = self._space_fill(space)
        pygame.draw.rect(surface, pygame.Color(fill), local_rect, border_radius=6)
        pygame.draw.rect(surface, pygame.Color(theme.BOARD_LINE), local_rect, width=2, border_radius=6)

        self._draw_special_accent(surface, local_rect, space)

        if space.color_group is not None:
            self._draw_color_band(surface, local_rect, space)

        label_lines = self._space_label(space)
        self._draw_label(surface, local_rect, label_lines)

    def _draw_color_band(self, surface: pygame.Surface, rect: pygame.Rect, space: BoardSpaceView) -> None:
        color = BOARD_COLOR_GROUPS.get(space.color_group or "", "#DDDDDD")
        band_color = pygame.Color(color)
        band_thickness = 14
        if space.index <= 10:
            band_rect = pygame.Rect(rect.left, rect.bottom - band_thickness, rect.width, band_thickness)
        elif 10 < space.index <= 20:
            band_rect = pygame.Rect(rect.left, rect.top, band_thickness, rect.height)
        elif 20 < space.index <= 30:
            band_rect = pygame.Rect(rect.left, rect.top, rect.width, band_thickness)
        else:
            band_rect = pygame.Rect(rect.right - band_thickness, rect.top, band_thickness, rect.height)
        pygame.draw.rect(surface, band_color, band_rect)
        pygame.draw.rect(surface, pygame.Color("#3A2E22"), band_rect, width=1)

    def _draw_special_accent(self, surface: pygame.Surface, rect: pygame.Rect, space: BoardSpaceView) -> None:
        accent_color = self._special_accent_color(space)
        if accent_color is None or space.color_group is not None:
            return
        accent = pygame.Color(accent_color)
        if space.index <= 10:
            accent_rect = pygame.Rect(rect.left + 6, rect.top + 6, rect.width - 12, 8)
        elif 10 < space.index <= 20:
            accent_rect = pygame.Rect(rect.right - 14, rect.top + 6, 8, rect.height - 12)
        elif 20 < space.index <= 30:
            accent_rect = pygame.Rect(rect.left + 6, rect.bottom - 14, rect.width - 12, 8)
        else:
            accent_rect = pygame.Rect(rect.left + 6, rect.top + 6, 8, rect.height - 12)
        pygame.draw.rect(surface, accent, accent_rect, border_radius=4)

    def _draw_label(self, surface: pygame.Surface, rect: pygame.Rect, lines: list[str]) -> None:
        total_height = len(lines) * 13
        y = rect.centery - total_height // 2
        for line in lines:
            rendered = self._small_font.render(line, True, pygame.Color(theme.TEXT_PRIMARY))
            surface.blit(rendered, rendered.get_rect(center=(rect.centerx, y + rendered.get_height() // 2)))
            y += 13

    def _draw_dynamic_overlays(
        self,
        surface: pygame.Surface,
        state: FrontendStateView,
        selected_space_index: int,
        hovered_space_index: int | None,
        hidden_player_names: set[str],
        token_overrides: dict[str, tuple[float, float]],
    ) -> None:
        for space in state.board_spaces:
            rect = self.layout.space_rect(space.index)
            if space.owner_name is not None:
                self._draw_owner_strip(surface, state, rect, space.owner_name, space.index)
            if space.building_count:
                self._draw_buildings(surface, rect, space.building_count)
            if space.mortgaged:
                stamp = build_stamp_surface((72, 24), "MORTGAGE", "#9F2D2D")
                surface.blit(stamp, stamp.get_rect(center=rect.center))

            outline_color = None
            outline_width = 0
            if space.index == selected_space_index:
                outline_color = pygame.Color(theme.ACTIVE_HIGHLIGHT)
                outline_width = 4
            elif hovered_space_index == space.index:
                outline_color = pygame.Color(theme.HOVER_HIGHLIGHT)
                outline_width = 3
            if outline_color is not None:
                pygame.draw.rect(surface, outline_color, rect, width=outline_width, border_radius=6)

            for slot, occupant in enumerate(space.occupant_names):
                if occupant in hidden_player_names:
                    continue
                token_color = self._player_color(state, occupant)
                token = build_token_surface(token_color, 22)
                surface.blit(token, self.layout.token_anchor(space.index, slot))

        for player_name, position in token_overrides.items():
            token_color = self._player_color(state, player_name)
            token = build_token_surface(token_color, 22)
            token_rect = token.get_rect(center=(int(position[0]), int(position[1])))
            surface.blit(token, token_rect)

    def _draw_owner_strip(
        self,
        surface: pygame.Surface,
        state: FrontendStateView,
        rect: pygame.Rect,
        owner_name: str,
        index: int,
    ) -> None:
        color = pygame.Color(self._player_color(state, owner_name))
        if index <= 10:
            strip = pygame.Rect(rect.left + 6, rect.top + 6, rect.width - 12, 8)
        elif 10 < index <= 20:
            strip = pygame.Rect(rect.right - 14, rect.top + 6, 8, rect.height - 12)
        elif 20 < index <= 30:
            strip = pygame.Rect(rect.left + 6, rect.bottom - 14, rect.width - 12, 8)
        else:
            strip = pygame.Rect(rect.left + 6, rect.top + 6, 8, rect.height - 12)
        pygame.draw.rect(surface, color, strip, border_radius=4)

    def _draw_buildings(self, surface: pygame.Surface, rect: pygame.Rect, count: int) -> None:
        for offset in range(min(count, 5)):
            building_rect = pygame.Rect(rect.left + 8 + offset * 14, rect.top + 8, 10, 10)
            fill = "#0F9D58" if offset < 4 else "#D14343"
            pygame.draw.rect(surface, pygame.Color(fill), building_rect, border_radius=2)
            pygame.draw.rect(surface, pygame.Color("#1F1A17"), building_rect, width=1, border_radius=2)

    def _space_fill(self, space: BoardSpaceView) -> str:
        if space.color_group is not None:
            return "#F7F3EA"
        if space.space_type == "card":
            return self._card_fill(space)
        return theme.SPECIAL_SPACE_COLORS.get(space.space_type, "#EFE8DA")

    def _space_label(self, space: BoardSpaceView) -> list[str]:
        lines = list(COMMONWEALTH_LABELS.get(space.name, self._fallback_label_lines(space.name)))
        if space.price is not None:
            lines.append(f"${space.price}")
        return lines[:3]

    def _card_fill(self, space: BoardSpaceView) -> str:
        notes = (space.notes or "").lower()
        if "community_chest" in notes:
            return "#DCE7C8"
        if "chance" in notes:
            return "#F2D7A8"
        return theme.SPECIAL_SPACE_COLORS.get(space.space_type, "#EFE8DA")

    def _special_accent_color(self, space: BoardSpaceView) -> str | None:
        if space.space_type == "railroad":
            return "#5A4E45"
        if space.space_type == "utility":
            return "#4C8D79"
        if space.space_type == "tax":
            return "#B65B4D"
        if space.space_type == "card":
            notes = (space.notes or "").lower()
            if "community_chest" in notes:
                return "#6D8A45"
            if "chance" in notes:
                return "#B6782A"
        if space.space_type == "go_to_jail":
            return "#A34638"
        if space.space_type == "free_parking":
            return "#B28E25"
        return None

    @staticmethod
    def _fallback_label_lines(name: str) -> list[str]:
        words = name.upper().split()
        if len(words) <= 2:
            return [" ".join(words)]
        midpoint = max(1, len(words) // 2)
        return [" ".join(words[:midpoint]), " ".join(words[midpoint:])]

    def _player_color(self, state: FrontendStateView | None, player_name: str) -> str:
        if state is not None:
            for index, player in enumerate(state.game_view.players):
                if player.name == player_name:
                    return theme.PLAYER_COLORS[index % len(theme.PLAYER_COLORS)]
        return theme.PLAYER_COLORS[abs(hash(player_name)) % len(theme.PLAYER_COLORS)]