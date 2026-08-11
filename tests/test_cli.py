"""Tests for hat.cli.commandlineify decorator."""

import pytest
import yaml

from hat.cli import commandlineify


class TestCommandlineify:
    def test_calls_function_with_config(self, tmp_path):
        config_data = {"grid": {"source": "test"}, "station": {"file": "x.csv"}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        received = {}

        @commandlineify
        def my_func(config):
            received.update(config)

        my_func([str(config_file)])
        assert received == config_data

    def test_missing_file_raises(self):
        @commandlineify
        def my_func(config):
            pass

        with pytest.raises((SystemExit, FileNotFoundError)):
            my_func(["/nonexistent/path.yaml"])

    def test_no_args_raises_system_exit(self):
        @commandlineify
        def my_func(config):
            pass

        with pytest.raises(SystemExit):
            my_func([])
