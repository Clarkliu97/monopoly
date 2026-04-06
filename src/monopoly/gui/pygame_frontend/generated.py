from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter
import pygame


def _hex_to_rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4)) + (alpha,)


def _pil_to_surface(image: Image.Image) -> pygame.Surface:
    rgba = image.convert("RGBA")
    surface = pygame.image.fromstring(rgba.tobytes(), rgba.size, "RGBA")
    if pygame.display.get_surface() is not None:
        return surface.convert_alpha()
    return surface


@lru_cache(maxsize=32)
def build_panel_surface(size: tuple[int, int], top_color: str, bottom_color: str) -> pygame.Surface:
    width, height = size
    image = Image.new("RGBA", size, _hex_to_rgba(bottom_color))
    draw = ImageDraw.Draw(image)
    top_rgba = _hex_to_rgba(top_color)
    bottom_rgba = _hex_to_rgba(bottom_color)
    for y in range(height):
        factor = y / max(height - 1, 1)
        blended = tuple(
            int(top_rgba[channel] + (bottom_rgba[channel] - top_rgba[channel]) * factor)
            for channel in range(3)
        ) + (255,)
        draw.line((0, y, width, y), fill=blended)
    draw.rounded_rectangle((1, 1, width - 2, height - 2), radius=18, outline=(84, 67, 51, 70), width=2)
    return _pil_to_surface(image)


@lru_cache(maxsize=64)
def build_token_surface(color: str, diameter: int) -> pygame.Surface:
    image = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    shadow = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.ellipse((4, 6, diameter - 2, diameter), fill=(0, 0, 0, 90))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))
    image.alpha_composite(shadow)

    draw.ellipse((2, 2, diameter - 4, diameter - 4), fill=_hex_to_rgba(color), outline=(20, 20, 20, 220), width=2)
    highlight = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    highlight_draw = ImageDraw.Draw(highlight)
    highlight_draw.ellipse((6, 4, diameter - 10, diameter // 2), fill=(255, 255, 255, 70))
    image.alpha_composite(highlight)
    return _pil_to_surface(image)


@lru_cache(maxsize=16)
def build_stamp_surface(size: tuple[int, int], text: str, color: str) -> pygame.Surface:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    rgba = _hex_to_rgba(color, 170)
    draw.rounded_rectangle((2, 2, size[0] - 2, size[1] - 2), radius=10, outline=rgba, width=3)
    draw.line((8, size[1] - 8, size[0] - 8, 8), fill=rgba, width=3)
    draw.text((10, size[1] // 2 - 7), text, fill=rgba)
    return _pil_to_surface(image)