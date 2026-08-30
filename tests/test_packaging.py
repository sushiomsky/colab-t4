from pathlib import Path

import colab_t4


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_release_version_is_0_4_0():
    assert colab_t4.__version__ == "0.4.0"
    assert 'version = "0.4.0"' in PYPROJECT


def test_mcp_extra_is_conditional_and_current_major():
    assert 'mcp = ["mcp>=2,<3; python_version >= \'3.10\'"]' in PYPROJECT


def test_mcp_console_entrypoint_is_registered():
    assert 'colab-t4-mcp = "colab_t4.mcp_server:main"' in PYPROJECT


def test_base_package_still_supports_python_3_9():
    assert 'requires-python = ">=3.9"' in PYPROJECT
