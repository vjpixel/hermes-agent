"""Regression test for #6673: every gateway ``/model`` listing path —
the native inline-keyboard picker (Telegram/Discord) AND the text-list
fallback for platforms without one — honors
``model_catalog.picker_scope: routing`` end to end, and ``--all`` bypasses
it for a single invocation.
"""

import types

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _FakePickerAdapter:
    """Captures the ``providers`` payload handed to the native picker."""

    def __init__(self):
        self.captured_providers = None

    async def send_model_picker(self, *, providers, **kwargs):
        self.captured_providers = providers
        return types.SimpleNamespace(success=True)


def _make_runner(adapter=None):
    runner = object.__new__(GatewayRunner)
    # No adapter -> forces the text-list fallback. With an adapter ->
    # exercises the native inline-keyboard picker branch instead.
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
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


# ─── native inline-keyboard picker branch (has_picker=True) ────────────
#
# The text-list fallback tests above deliberately force runner.adapters={}
# so the picker adapter is never consulted. Telegram/Discord normally DO
# have a native picker (send_model_picker), which is the primary UX this
# feature exists for — a bug here (like the max_models=50-before-filter
# truncation this test locks in) would ship invisibly if only the fallback
# path were tested.
#
# list_picker_providers() unconditionally replaces the openrouter row's
# models with a LIVE fetch_openrouter_models() result (never the raw
# config-mocked list_authenticated_providers() rows) — every test in this
# section must stub that live fetch too, or it silently hits the real
# network / a stale disk cache instead of the fixture.


def _stub_openrouter_live(monkeypatch, models):
    monkeypatch.setattr(
        "hermes_cli.models.fetch_openrouter_models",
        lambda *a, **kw: [(m, {}) for m in models],
    )


@pytest.mark.asyncio
async def test_native_picker_narrows_to_routing_scope(tmp_path, monkeypatch):
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
    _stub_openrouter_live(monkeypatch, ["kept/model", "dropped/model", "also-dropped"])
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kw: rows,
    )

    adapter = _FakePickerAdapter()
    result = await _make_runner(adapter)._handle_model_command(_make_event("/model"))

    assert result is None  # picker sent — adapter handled the response
    assert adapter.captured_providers is not None
    openrouter_row = next(
        r for r in adapter.captured_providers if r["slug"] == "openrouter"
    )
    assert openrouter_row["models"] == ["kept/model"]


@pytest.mark.asyncio
async def test_native_picker_all_flag_bypasses_routing_scope(tmp_path, monkeypatch):
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
    _stub_openrouter_live(monkeypatch, ["kept/model", "also-shown"])
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kw: rows,
    )

    adapter = _FakePickerAdapter()
    await _make_runner(adapter)._handle_model_command(_make_event("/model --all"))

    openrouter_row = next(
        r for r in adapter.captured_providers if r["slug"] == "openrouter"
    )
    assert openrouter_row["models"] == ["kept/model", "also-shown"]


@pytest.mark.asyncio
async def test_native_picker_does_not_truncate_routing_model_past_position_50(
    tmp_path, monkeypatch
):
    """Regression: the interactive-picker branch used to pass a hardcoded
    ``max_models=50`` into list_picker_providers() UNCONDITIONALLY, which
    truncated each provider's curated model list to 50 entries BEFORE
    apply_routing_scope() ran — silently dropping a routing-allowlisted
    model that happened to sit past position 50 in a large curated catalog
    (e.g. an aggregator with 65+ entries). The scoped fetch must be
    unclipped so the routing filter sees the full list first."""
    routed_model = "vendor/model-past-position-50"
    _setup_isolated_home(
        tmp_path,
        monkeypatch,
        model_catalog_cfg={"picker_scope": "routing"},
        smart_routing_cfg={
            "enabled": True,
            "profiles": {
                "simple": {
                    "primary_model": {"provider": "openrouter", "model": routed_model},
                },
            },
        },
    )

    # 60 unrelated models ahead of the routed one in the curated list —
    # position 60, well past the old hardcoded max_models=50 cap.
    big_catalog = [f"other/model-{i}" for i in range(60)] + [routed_model]
    rows = [_row("openrouter", big_catalog)]
    _stub_openrouter_live(monkeypatch, big_catalog)
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kw: rows,
    )

    adapter = _FakePickerAdapter()
    await _make_runner(adapter)._handle_model_command(_make_event("/model"))

    openrouter_row = next(
        r for r in adapter.captured_providers if r["slug"] == "openrouter"
    )
    assert openrouter_row["models"] == [routed_model]


@pytest.mark.asyncio
async def test_text_list_fallback_keeps_more_than_five_routing_matches(
    tmp_path, monkeypatch
):
    """The post-scope display cap (``[:5]``) must slice the ROUTING-SCOPED
    list, not silently lose matches beyond the first 5 of the raw curated
    catalog — and the "+N more" suffix must count the scoped remainder."""
    kept = [f"kept/model-{i}" for i in range(7)]
    _setup_isolated_home(
        tmp_path,
        monkeypatch,
        model_catalog_cfg={"picker_scope": "routing"},
        smart_routing_cfg={
            "enabled": True,
            "profiles": {
                "simple": {"fallback_chain": [{"provider": "openrouter", "model": m} for m in kept]},
            },
        },
    )

    rows = [_row("openrouter", kept + ["dropped/model"])]
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kw: rows,
    )

    result = await _make_runner()._handle_model_command(_make_event("/model"))

    assert result is not None
    assert "dropped/model" not in result
    # First 5 of the 7 routing-scoped matches are displayed inline...
    for shown in kept[:5]:
        assert shown in result
    # ...and the "+N more" count reflects the routing-scoped remainder (2:
    # 7 scoped matches minus the 5 displayed), not what it would be against
    # the raw 8-entry catalog (3).
    assert "+2 more" in result
