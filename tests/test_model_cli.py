import argparse
import json

import colab_t4.cli as cli


def test_model_current_parser():
    args = cli.build_parser().parse_args(["model", "current", "--json"])
    assert args.command == "model"
    assert args.model_command == "current"
    assert args.json is True


def test_model_switch_parser_defaults():
    args = cli.build_parser().parse_args(["model", "switch", "example/model-GGUF"])
    assert args.command == "model"
    assert args.model_command == "switch"
    assert args.repository == "example/model-GGUF"
    assert args.quant == "Q4_K_M"
    assert args.timeout == 3600


def test_model_current_json_output(monkeypatch, capsys):
    expected = {
        "session": "colab-t4",
        "runtime_state": "ready",
        "model_repo": "example/current-GGUF",
        "quant": "Q4_K_M",
    }
    monkeypatch.setattr(cli, "current_model", lambda: expected)
    args = argparse.Namespace(model_command="current", json=True)
    assert cli.cmd_model(args) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_model_switch_delegates_and_prints_json(monkeypatch, capsys):
    calls = []
    result = {
        "runtime_state": "ready",
        "model_repo": "example/new-GGUF",
        "quant": "Q5_K_M",
    }
    monkeypatch.setattr(
        cli,
        "switch_model",
        lambda repository, quant, timeout: calls.append((repository, quant, timeout)) or result,
    )
    args = argparse.Namespace(
        model_command="switch",
        repository="example/new-GGUF",
        quant="Q5_K_M",
        timeout=123.0,
        json=True,
    )
    assert cli.cmd_model(args) == 0
    assert calls == [("example/new-GGUF", "Q5_K_M", 123.0)]
    assert json.loads(capsys.readouterr().out) == result
