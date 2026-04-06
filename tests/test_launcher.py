from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from monopoly.gui import launcher
from monopoly.gui.rendezvous import DEFAULT_RENDEZVOUS_HOST, DEFAULT_RENDEZVOUS_PORT, run_rendezvous_process


class LaunchMonopolyGameTests(unittest.TestCase):
    @patch("monopoly.gui.launcher.mp.freeze_support")
    @patch("monopoly.gui.launcher.mp.Process")
    def test_launcher_defaults_to_pygame_frontend(self, process_cls, freeze_support) -> None:
        rendezvous_process = MagicMock()
        gui_process = MagicMock()
        process_cls.side_effect = [rendezvous_process, gui_process]

        launcher.launch_monopoly_game(debug_mode=True)

        freeze_support.assert_called_once_with()
        self.assertEqual(2, process_cls.call_count)
        rendezvous_call = process_cls.call_args_list[0]
        gui_call = process_cls.call_args_list[1]
        self.assertEqual("monopoly-rendezvous", rendezvous_call.kwargs["name"])
        self.assertEqual((DEFAULT_RENDEZVOUS_HOST, DEFAULT_RENDEZVOUS_PORT), rendezvous_call.kwargs["args"])
        self.assertIs(run_rendezvous_process, rendezvous_call.kwargs["target"])
        self.assertEqual("monopoly-pygame-gui", gui_call.kwargs["name"])
        self.assertEqual((None, None, DEFAULT_RENDEZVOUS_HOST, DEFAULT_RENDEZVOUS_PORT, True), gui_call.kwargs["args"])
        self.assertIs(launcher.run_pygame_gui_process, gui_call.kwargs["target"])
        rendezvous_process.start.assert_called_once_with()
        gui_process.start.assert_called_once_with()
        gui_process.join.assert_called_once_with()
        rendezvous_process.join.assert_called_once_with()

    def test_launcher_rejects_non_pygame_frontends(self) -> None:
        with self.assertRaises(ValueError):
            launcher.launch_monopoly_game("tkinter")


if __name__ == "__main__":
    unittest.main()