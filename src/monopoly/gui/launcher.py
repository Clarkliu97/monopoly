from __future__ import annotations

import multiprocessing as mp

from monopoly.gui.pygame_frontend import run_gui_process as run_pygame_gui_process
from monopoly.gui.rendezvous import DEFAULT_RENDEZVOUS_HOST, DEFAULT_RENDEZVOUS_PORT, run_rendezvous_process


def launch_monopoly_game(frontend: str | None = None, debug_mode: bool = False) -> None:
    mp.freeze_support()
    selected_frontend = (frontend or "pygame").strip().lower()
    if selected_frontend != "pygame":
        raise ValueError("frontend must be 'pygame'.")

    rendezvous_process = mp.Process(
        target=run_rendezvous_process,
        args=(DEFAULT_RENDEZVOUS_HOST, DEFAULT_RENDEZVOUS_PORT),
        name="monopoly-rendezvous",
    )

    gui_process = mp.Process(
        target=run_pygame_gui_process,
        args=(None, None, DEFAULT_RENDEZVOUS_HOST, DEFAULT_RENDEZVOUS_PORT, debug_mode),
        name="monopoly-pygame-gui",
    )

    rendezvous_process.start()
    gui_process.start()

    gui_process.join()
    if rendezvous_process.is_alive():
        rendezvous_process.terminate()
    rendezvous_process.join()
