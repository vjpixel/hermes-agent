"""Regression test for #6673: the gateway ``/model`` text-list fallback
(Telegram/Discord platforms without a native picker) honors
``model_catalog.picker_scope: routing`` end to end, and ``--all`` bypasses
it for a single invocation.
"""

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}  # no picker adapter -> forces the text-list fallback
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _row(slug, models):
    return {
        "slug": slug,
        "name": slug,
        "models": models,
        "total_models": len(models),
        "is_current": False,
        "is_user_defined": False,
        "source": "built-in",
    }


def _setup_isolated_home(tmp_path, monkeypatch, model_catalog_cfg, smart_routing_cfg):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "model": {"default": "current-model", "provider": "openrouter"},
                "providers": {},
                "model_catalog": model_catalog_cfg,
                "smart_model_routing": smart_routing_cfg,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    return cfg_path


@pytest.mark.asyncio
async def test_text_list_fallback_narrows_to_routing_scope(tmp_path, monkeypatch):
    _setup_isolated_home(
        tmp_path,
        monkeypatch,
        model_catalog_cfg={"picker_scope": "routing"},
        smart_routing_cfg={
            "enabled": True,
            "profiles": {
                "simple": {
                    "primary_model": {"provider": "openrouter", "model": "kept/model"},
                },
            },
        },
    )

    rows = [_row("openrouter", ["kept/model", "dropped/model", "also-dropped"])]
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kw: rows,
    )

    result = await _make_runner()._handle_model_command(_make_event("/model"))

    assert result is not None
    assert "kept/model" in result
    assert "dropped/model" not in result
    assert "also-dropped" not in result


@pytest.mark.asyncio
async def test_all_flag_bypasses_routing_scope(tmp_path, monkeypatch):
    _setup_isolated_home(
        tmp_path,
        monkeypatch,
        model_catalog_cfg={"picker_scope": "routing"},
        smart_routing_cfg={
            "enabled": True,
            "profiles": {
                "simple": {
                    "primary_model": {"provider": "openrouter", "model": "kept/model"},
                },
            },
        },
    )

    rows = [_row("openrouter", ["kept/model", "also-shown"])]
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kw: rows,
    )

    result = await _make_runner()._handle_model_command(_make_event("/model --all"))

    assert result is not None
    assert "kept/model" in result
    assert "also-shown" in result


@pytest.mark.asyncio
async def test_default_scope_all_is_unrestricted(tmp_path, monkeypatch):
    """Sanity: no ``model_catalog.picker_scope`` set at all -> today's
    unrestricted listing, unchanged by this feature."""
    _setup_isolated_home(
        tmp_path,
        monkeypatch,
        model_catalog_cfg={},
        smart_routing_cfg={},
    )

    rows = [_row("openrouter", ["m1", "m2", "m3"])]
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kw: rows,
    )

    result = await _make_runner()._handle_model_command(_make_event("/model"))

    assert result is not None
    for expected in ("m1", "m2", "m3"):
        assert expected in result
