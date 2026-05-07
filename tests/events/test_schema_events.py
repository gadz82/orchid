"""Schema validation: extra=forbid, identity union, schedule/trigger
cross-checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchid_ai.config.schema_events import (
    ActAsUserIdentity,
    OrchidEventsConfig,
    OrchidScheduleConfig,
    OrchidTriggerConfig,
    OrchidTriggerEmitConfig,
    OrchidTriggerMatchConfig,
    ServiceAccountIdentity,
)


# ── extra=forbid ─────────────────────────────────────────────


def test_trigger_match_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        OrchidTriggerMatchConfig.model_validate({"signal": "demo.event", "unknown": "x"})


def test_events_config_rejects_unknown_top_level_keys() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        OrchidEventsConfig.model_validate({"enabled": False, "weird": True})


# ── Identity discriminated union ────────────────────────────


def test_identity_union_dispatches_on_mode() -> None:
    sa = OrchidTriggerEmitConfig.model_validate(
        {
            "agent": "x",
            "prompt_template": "p",
            "identity": {"mode": "service_account", "name": "bot"},
        }
    )
    assert isinstance(sa.identity, ServiceAccountIdentity)

    aau = OrchidTriggerEmitConfig.model_validate(
        {
            "agent": "x",
            "prompt_template": "p",
            "identity": {"mode": "act_as_user", "user_id_from": "signal.user_id"},
        }
    )
    assert isinstance(aau.identity, ActAsUserIdentity)


def test_identity_union_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        OrchidTriggerEmitConfig.model_validate({"agent": "x", "prompt_template": "p", "identity": {"mode": "??"}})


# ── Schedule timing exclusivity ─────────────────────────────


def test_schedule_must_specify_one_timing() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        OrchidScheduleConfig.model_validate(
            {
                "id": "s1",
                "trigger_id": "t1",
                "identity": {"mode": "service_account", "name": "bot"},
                # neither cron nor interval_seconds
            }
        )


def test_schedule_rejects_both_cron_and_interval() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        OrchidScheduleConfig.model_validate(
            {
                "id": "s1",
                "trigger_id": "t1",
                "cron": "* * * * *",
                "interval_seconds": 30,
                "identity": {"mode": "service_account", "name": "bot"},
            }
        )


# ── Trigger 'on.cron' is only valid with signal == 'cron' ───


def test_trigger_match_cron_requires_cron_signal() -> None:
    with pytest.raises(ValidationError, match="only valid"):
        OrchidTriggerMatchConfig.model_validate({"signal": "ticket.created", "cron": "* * * * *"})


def test_trigger_match_cron_signal_requires_cron_field() -> None:
    with pytest.raises(ValidationError, match="required"):
        OrchidTriggerMatchConfig.model_validate({"signal": "cron"})


# ── enabled=true cross-field validation ─────────────────────


def _trigger(id: str = "t1", signal_type: str = "demo.event") -> OrchidTriggerConfig:
    return OrchidTriggerConfig(
        id=id,
        on=OrchidTriggerMatchConfig(signal=signal_type, cron="* * * * *" if signal_type == "cron" else None),
        emits=OrchidTriggerEmitConfig(
            agent="notifications",
            prompt_template="hi",
            identity=ServiceAccountIdentity(name="bot"),
        ),
    )


def test_enabled_requires_store_queue_processors() -> None:
    with pytest.raises(ValidationError) as excinfo:
        OrchidEventsConfig.model_validate({"enabled": True})
    msg = str(excinfo.value)
    assert "events.store" in msg
    assert "events.queue" in msg
    assert "events.processors" in msg


def test_disabled_does_not_require_store_queue_processors() -> None:
    cfg = OrchidEventsConfig.model_validate({"enabled": False})
    assert cfg.enabled is False
    assert cfg.store is None
    assert cfg.queue is None


def test_schedule_must_reference_existing_trigger() -> None:
    with pytest.raises(ValidationError, match="unknown trigger"):
        OrchidEventsConfig.model_validate(
            {
                "enabled": True,
                "store": {"class": "x.Y"},
                "queue": {"class": "x.Q"},
                "processors": [{"class": "x.P"}],
                "schedules": [
                    {
                        "id": "s1",
                        "cron": "* * * * *",
                        "trigger_id": "missing",
                        "identity": {"mode": "service_account", "name": "bot"},
                    }
                ],
                "triggers": [_trigger("t1").model_dump(by_alias=True)],
            }
        )


def test_schedule_target_trigger_must_be_cron_signal() -> None:
    triggers = [_trigger("t1", signal_type="ticket.created").model_dump(by_alias=True)]
    with pytest.raises(ValidationError, match="not 'cron'"):
        OrchidEventsConfig.model_validate(
            {
                "enabled": True,
                "store": {"class": "x.Y"},
                "queue": {"class": "x.Q"},
                "processors": [{"class": "x.P"}],
                "schedules": [
                    {
                        "id": "s1",
                        "cron": "* * * * *",
                        "trigger_id": "t1",
                        "identity": {"mode": "service_account", "name": "bot"},
                    }
                ],
                "triggers": triggers,
            }
        )


def test_full_valid_events_block() -> None:
    triggers = [_trigger("cron-t", signal_type="cron").model_dump(by_alias=True)]
    cfg = OrchidEventsConfig.model_validate(
        {
            "enabled": True,
            "store": {"class": "x.Y"},
            "queue": {"class": "x.Q", "lease_seconds": 60},
            "processors": [{"class": "x.P", "concurrency": 8}],
            "scheduler": {"class": "x.S"},
            "schedules": [
                {
                    "id": "s1",
                    "cron": "0 * * * *",
                    "trigger_id": "cron-t",
                    "identity": {"mode": "service_account", "name": "bot"},
                }
            ],
            "triggers": triggers,
        }
    )
    assert cfg.enabled is True
    assert cfg.queue is not None and cfg.queue.lease_seconds == 60
    assert cfg.processors[0].concurrency == 8
