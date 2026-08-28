"""Tests for `model_catalog.picker_scope: routing` (#6673).

The picker allowlist must be DERIVED from `smart_model_routing` — never a
literal list of today's model names — so the next fallback-chain rotation
(like #6663) is followed automatically instead of breaking a hardcoded
test. Every "routing" test below builds its own `smart_model_routing`
fixture and asserts the union derived FROM that fixture, not a copy/paste
of the real config.yaml's current chain.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.inventory import (
    apply_routing_scope,
    resolve_routing_scope,
)


# ─── resolve_routing_scope ──────────────────────────────────────────────


def test_default_scope_is_all_and_derives_nothing():
    scope, pairs = resolve_routing_scope({})
    assert scope == "all"
    assert pairs == frozenset()


def test_unknown_scope_value_fails_open_to_all():
    cfg = {"model_catalog": {"picker_scope": "bogus"}}
    scope, pairs = resolve_routing_scope(cfg)
    assert scope == "all"
    assert pairs == frozenset()


def _routing_cfg(profiles: dict, extras: list | None = None) -> dict:
    cfg = {
        "model_catalog": {"picker_scope": "routing"},
        "smart_model_routing": {"enabled": True, "profiles": profiles},
    }
    if extras is not None:
        cfg["model_catalog"]["picker_extras"] = extras
    return cfg


def test_routing_scope_derives_union_of_primary_and_fallback_chain():
    """A schema shaped like the issue's example — profiles keyed by name,
    each with primary_model + fallback_chain — must union into exactly the
    (provider, model) pairs present, nothing hardcoded."""
    cfg = _routing_cfg(
        {
            "coding": {
                "primary_model": {"provider": "custom", "model": "local-coder:latest"},
                "fallback_chain": [
                    {"provider": "openrouter", "model": "vendor-x/model-a:free"},
                    {"provider": "openai-codex", "model": "gpt-fast"},
                ],
            },
            "general": {
                "primary_model": {"provider": "openrouter", "model": "vendor-y/model-b:free"},
                "fallback_chain": [
                    {"provider": "openrouter", "model": "vendor-x/model-a:free"},
                ],
            },
        }
    )
    scope, pairs = resolve_routing_scope(cfg)
    assert scope == "routing"
    assert pairs == frozenset(
        {
            ("custom", "local-coder:latest"),
            ("openrouter", "vendor-x/model-a:free"),
            ("openai-codex", "gpt-fast"),
            ("openrouter", "vendor-y/model-b:free"),
        }
    )


def test_routing_scope_survives_a_chain_rotation_without_a_test_edit():
    """Regression for the exact failure mode #6663 documents: swap the
    fixture's chain contents and the derived union follows automatically —
    no hardcoded list in this test needs to change."""
    rotated = _routing_cfg(
        {
            "simple": {
                "primary_model": {"provider": "openrouter", "model": "brand-new/model-2026"},
                "fallback_chain": [
                    {"provider": "openrouter", "model": "yet-another/model"},
                ],
            },
        }
    )
    scope, pairs = resolve_routing_scope(rotated)
    assert scope == "routing"
    assert pairs == frozenset(
        {
            ("openrouter", "brand-new/model-2026"),
            ("openrouter", "yet-another/model"),
        }
    )


def test_routing_scope_includes_picker_extras():
    cfg = _routing_cfg(
        {"simple": {"primary_model": {"provider": "openrouter", "model": "a/b"}}},
        extras=[{"provider": "anthropic", "model": "claude-extra"}],
    )
    _scope, pairs = resolve_routing_scope(cfg)
    assert ("anthropic", "claude-extra") in pairs
    assert ("openrouter", "a/b") in pairs


def test_routing_scope_ignores_malformed_entries():
    cfg = _routing_cfg(
        {
            "simple": {
                "primary_model": {"provider": "openrouter"},  # missing model
                "fallback_chain": [
                    {"model": "no-provider"},
                    "not-a-dict",
                    {"provider": "", "model": "empty-provider"},
                    {"provider": "ok-provider", "model": "ok-model"},
                ],
            }
        }
    )
    _scope, pairs = resolve_routing_scope(cfg)
    assert pairs == frozenset({("ok-provider", "ok-model")})


def test_routing_scope_empty_smart_routing_yields_empty_pairs():
    cfg = {
        "model_catalog": {"picker_scope": "routing"},
        "smart_model_routing": {"enabled": False},
    }
    scope, pairs = resolve_routing_scope(cfg)
    assert scope == "routing"
    assert pairs == frozenset()


# ─── apply_routing_scope ────────────────────────────────────────────────


def _row(slug, models, **extra) -> dict:
    row = {
        "slug": slug,
        "name": slug,
        "models": models,
        "total_models": len(models),
        "is_current": False,
        "is_user_defined": False,
        "source": "built-in",
    }
    row.update(extra)
    return row


def test_apply_routing_scope_noop_when_scope_is_all():
    rows = [_row("openrouter", ["a/b", "c/d"])]
    out = apply_routing_scope(rows, scope="all", routing_pairs=frozenset({("openrouter", "a/b")}))
    assert out == rows
    assert out is rows


def test_apply_routing_scope_fails_open_on_empty_pairs():
    """An empty allowlist under routing scope must not produce an empty
    picker — better to show everything than nothing."""
    rows = [_row("openrouter", ["a/b", "c/d"])]
    out = apply_routing_scope(rows, scope="routing", routing_pairs=frozenset())
    assert out == rows


def test_apply_routing_scope_filters_models_and_drops_empty_rows():
    rows = [
        _row("openrouter", ["a/b", "c/d", "e/f"]),
        _row("anthropic", ["claude-1"]),
    ]
    pairs = frozenset({("openrouter", "a/b"), ("openrouter", "e/f")})
    out = apply_routing_scope(rows, scope="routing", routing_pairs=pairs)

    slugs = {r["slug"] for r in out}
    assert slugs == {"openrouter"}
    openrouter_row = next(r for r in out if r["slug"] == "openrouter")
    assert openrouter_row["models"] == ["a/b", "e/f"]
    assert openrouter_row["total_models"] == 2


def test_apply_routing_scope_matches_custom_provider_via_alias():
    """smart_model_routing may spell a custom endpoint's provider as the
    bare overlay name (e.g. "custom") while the row's slug carries the
    canonical custom:<key> identity — aliases must bridge the two."""
    row = _row(
        "custom:local-ollama",
        ["local-coder:latest", "other-model"],
        is_user_defined=True,
        aliases=["custom:local-ollama", "custom", "local-ollama"],
    )
    pairs = frozenset({("custom", "local-coder:latest")})
    out = apply_routing_scope([row], scope="routing", routing_pairs=pairs)
    assert len(out) == 1
    assert out[0]["models"] == ["local-coder:latest"]


def test_apply_routing_scope_never_mutates_input_rows():
    rows = [_row("openrouter", ["a/b", "c/d"])]
    pairs = frozenset({("openrouter", "a/b")})
    apply_routing_scope(rows, scope="routing", routing_pairs=pairs)
    assert rows[0]["models"] == ["a/b", "c/d"]


# ─── build_models_payload integration ───────────────────────────────────


def _list_auth_returning(rows: list[dict]):
    return patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=rows,
    )


def _no_real_config():
    """build_models_payload's MoA row reads the real on-disk config.yaml
    unless load_config() is patched — isolate every payload test from
    whatever `moa.presets` the machine running the suite happens to have."""
    return patch("hermes_cli.config.load_config", return_value={})


def _find_row(payload: dict, slug: str) -> dict:
    """Locate a row by slug — payloads always carry the virtual `moa` row
    too (a built-in default preset applies even against an empty config),
    so index [0] is not stable."""
    return next(r for r in payload["providers"] if r["slug"] == slug)


def test_build_models_payload_applies_scope_only_when_requested():
    from hermes_cli.inventory import ConfigContext, build_models_payload

    rows = [_row("openrouter", ["a/b", "c/d"])]
    ctx = ConfigContext(
        current_provider="openrouter",
        current_model="a/b",
        current_base_url="",
        user_providers={},
        custom_providers=[],
        picker_scope="routing",
        routing_pairs=frozenset({("openrouter", "a/b")}),
    )

    with _no_real_config(), _list_auth_returning(rows):
        # Aux-picker style call: scope_to_routing defaults False, so the
        # full inventory still comes back even though ctx says "routing".
        unscoped = build_models_payload(ctx)
        assert _find_row(unscoped, "openrouter")["models"] == ["a/b", "c/d"]

    with _no_real_config(), _list_auth_returning(rows):
        scoped = build_models_payload(ctx, scope_to_routing=True)
        assert _find_row(scoped, "openrouter")["models"] == ["a/b"]

    with _no_real_config(), _list_auth_returning(rows):
        # --all escape hatch bypasses scope for this one call.
        all_scoped = build_models_payload(ctx, scope_to_routing=True, show_all=True)
        assert _find_row(all_scoped, "openrouter")["models"] == ["a/b", "c/d"]


def test_build_model_options_payload_always_scopes_and_respects_show_all():
    from hermes_cli.inventory import ConfigContext, build_model_options_payload

    rows = [_row("openrouter", ["a/b", "c/d"])]
    ctx = ConfigContext(
        current_provider="openrouter",
        current_model="a/b",
        current_base_url="",
        user_providers={},
        custom_providers=[],
        picker_scope="routing",
        routing_pairs=frozenset({("openrouter", "a/b")}),
    )

    with _no_real_config(), _list_auth_returning(rows):
        payload = build_model_options_payload(ctx)
    assert _find_row(payload, "openrouter")["models"] == ["a/b"]

    with _no_real_config(), _list_auth_returning(rows):
        payload_all = build_model_options_payload(ctx, show_all=True)
    assert _find_row(payload_all, "openrouter")["models"] == ["a/b", "c/d"]


# ─── list_picker_providers (gateway Telegram/Discord) ──────────────────


def test_list_picker_providers_applies_routing_scope():
    from hermes_cli.model_switch import list_picker_providers

    rows = [_row("anthropic", ["claude-1", "claude-2"])]
    with patch("hermes_cli.model_switch.list_authenticated_providers", return_value=rows):
        scoped = list_picker_providers(
            picker_scope="routing",
            routing_pairs=frozenset({("anthropic", "claude-2")}),
        )
    assert len(scoped) == 1
    assert scoped[0]["models"] == ["claude-2"]


def test_list_picker_providers_default_scope_is_unrestricted():
    from hermes_cli.model_switch import list_picker_providers

    rows = [_row("anthropic", ["claude-1", "claude-2"])]
    with patch("hermes_cli.model_switch.list_authenticated_providers", return_value=rows):
        out = list_picker_providers()
    assert out[0]["models"] == ["claude-1", "claude-2"]


# ─── /model --all flag parsing ──────────────────────────────────────────


def test_parse_model_flags_detailed_recognizes_all_flag():
    from hermes_cli.model_switch import parse_model_flags_detailed

    parsed = parse_model_flags_detailed("--all")
    assert parsed.show_all is True
    assert parsed.model_input == ""

    parsed_with_target = parse_model_flags_detailed("sonnet --all")
    # --all is a listing-only flag; combined with a target it still parses
    # cleanly (the switch path itself ignores show_all).
    assert parsed_with_target.show_all is True
    assert parsed_with_target.model_input == "sonnet"


def test_parse_model_switch_args_propagates_show_all():
    from hermes_cli.model_switch import parse_model_switch_args

    request = parse_model_switch_args("--all")
    assert request.show_all is True
    assert not request.errors
