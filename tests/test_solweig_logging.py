"""Tests for the unified `solweig.solweig_logging` adapter."""

from __future__ import annotations

import logging

from solweig.solweig_logging import LogLevel, SolweigLogger, get_logger, set_global_level


def test_get_logger_returns_same_instance_for_same_name():
    a = get_logger("solweig.test_logging_unique_a")
    b = get_logger("solweig.test_logging_unique_a")
    assert a is b


def test_log_level_filtering_skips_below_minimum(caplog):
    log = get_logger("solweig.test_logging_filter", level=LogLevel.WARNING)
    with caplog.at_level(logging.DEBUG, logger="solweig.test_logging_filter"):
        log.debug("debug-msg")
        log.info("info-msg")
        log.warning("warn-msg")
        log.error("error-msg")

    messages = {r.message for r in caplog.records}
    assert "debug-msg" not in messages
    assert "info-msg" not in messages
    assert "warn-msg" in messages
    assert "error-msg" in messages


def test_set_global_level_propagates_to_all_loggers():
    a = get_logger("solweig.test_logging_global_a", level=LogLevel.INFO)
    b = get_logger("solweig.test_logging_global_b", level=LogLevel.INFO)
    set_global_level(LogLevel.ERROR)
    assert a.level == LogLevel.ERROR
    assert b.level == LogLevel.ERROR
    # Restore so other tests aren't affected
    set_global_level(LogLevel.INFO)


def test_qgis_backend_falls_back_to_python_logging_without_feedback(caplog):
    """If the QGIS backend is detected but no feedback object is set, logs must
    still appear via the standard Python logging module."""
    log = SolweigLogger("solweig.test_qgis_fallback", level=LogLevel.INFO)
    # Force the QGIS branch even on a non-QGIS host — the fallback path
    # explicitly handles this case.
    log._backend = "qgis"
    log._feedback = None
    with caplog.at_level(logging.INFO, logger="solweig.test_qgis_fallback"):
        log.info("falling-back")
    assert any(r.message == "falling-back" for r in caplog.records)


def test_qgis_feedback_receives_push_info_when_set():
    """When a feedback object is attached, info routes through pushInfo and
    error routes through reportError."""

    class FakeFeedback:
        def __init__(self):
            self.info_calls = []
            self.error_calls = []
            self.debug_calls = []

        def pushInfo(self, msg):  # noqa: N802 — matches QGIS API
            self.info_calls.append(msg)

        def reportError(self, msg):  # noqa: N802
            self.error_calls.append(msg)

        def pushDebugInfo(self, msg):  # noqa: N802
            self.debug_calls.append(msg)

    fb = FakeFeedback()
    log = SolweigLogger("solweig.test_qgis_feedback", level=LogLevel.DEBUG)
    log._backend = "qgis"
    log.set_feedback(fb)

    log.debug("d")
    log.info("i")
    log.warning("w")
    log.error("e")

    assert fb.debug_calls == ["d"]
    # WARNING is pushed as info with a "WARNING:" prefix per the adapter contract.
    assert fb.info_calls == ["i", "WARNING: w"]
    assert fb.error_calls == ["e"]
