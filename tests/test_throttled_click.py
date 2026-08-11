"""Tests for ThrottledClick."""

from unittest.mock import patch

from hat.interactive.widgets import ThrottledClick


class TestThrottledClick:
    def test_first_call_returns_true(self):
        tc = ThrottledClick(delay=1.0)
        assert tc.should_process() is True

    def test_immediate_second_call_returns_false(self):
        with patch("time.time", side_effect=[100.0, 100.0]):
            tc = ThrottledClick(delay=1.0)
            tc.should_process()  # first call at t=100
        with patch("time.time", return_value=100.1):
            assert tc.should_process() is False

    def test_after_delay_returns_true(self):
        tc = ThrottledClick(delay=0.5)
        with patch("time.time", return_value=1000.0):
            assert tc.should_process() is True
        with patch("time.time", return_value=1000.3):
            assert tc.should_process() is False
        with patch("time.time", return_value=1000.6):
            assert tc.should_process() is True

    def test_custom_delay(self):
        tc = ThrottledClick(delay=5.0)
        with patch("time.time", return_value=100.0):
            assert tc.should_process() is True
        with patch("time.time", return_value=104.9):
            assert tc.should_process() is False
        with patch("time.time", return_value=105.1):
            assert tc.should_process() is True
