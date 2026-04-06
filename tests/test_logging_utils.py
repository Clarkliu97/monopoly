from __future__ import annotations

import logging
import re
import tempfile
import unittest

from monopoly.logging_utils import configure_process_logging


class LoggingUtilsTests(unittest.TestCase):
    def test_configure_process_logging_writes_separate_component_files_with_consistent_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frontend_log = configure_process_logging("frontend", log_directory=temp_dir, level="DEBUG")
            test_logger = logging.getLogger("monopoly.tests.logging")
            test_logger.info("Frontend ready")
            self._flush_monopoly_handlers()

            backend_log = configure_process_logging("backend", log_directory=temp_dir, level="INFO")
            test_logger.warning("Backend ready")
            self._flush_monopoly_handlers()

            self.assertEqual("frontend.log", frontend_log.name)
            self.assertEqual("backend.log", backend_log.name)

            frontend_contents = frontend_log.read_text(encoding="utf-8")
            backend_contents = backend_log.read_text(encoding="utf-8")

            self.assertRegex(
                frontend_contents,
                re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO\s+\| monopoly\.tests\.logging \| Frontend ready$", re.MULTILINE),
            )
            self.assertRegex(
                backend_contents,
                re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| WARNING\s+\| monopoly\.tests\.logging \| Backend ready$", re.MULTILINE),
            )
            self.assertNotIn("Backend ready", frontend_contents)
            self._close_monopoly_handlers()

    @staticmethod
    def _flush_monopoly_handlers() -> None:
        monopoly_logger = logging.getLogger("monopoly")
        for handler in monopoly_logger.handlers:
            handler.flush()

    @staticmethod
    def _close_monopoly_handlers() -> None:
        monopoly_logger = logging.getLogger("monopoly")
        for handler in list(monopoly_logger.handlers):
            monopoly_logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    unittest.main()