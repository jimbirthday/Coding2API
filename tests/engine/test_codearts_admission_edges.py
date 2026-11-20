"""Admission races and providers without a chat throttle."""

from __future__ import annotations

import pytest

from src.engine.executor import NoHealthyCredential
from src.engine.model_resolver import resolve
from src.provider.base import ErrKind, Event, EventKind
from tests.engine.test_codearts_admission import (
    _good,
    _released,
    _request,
    _run,
)
from tests.engine.test_codearts_admission import build as build

pytestmark = pytest.mark.parametrize("mode", ["stream", "complete"])


async def test_selection_excludes_all_tried_accounts_without_acquiring_probe(build, mode):
    harness = build()
    target = resolve(_request(mode).model, _request(mode).model)
    assert harness.executor._select(target, set(harness.ids)) is None
    assert not harness.client.calls
    _released(harness)


async def test_reservation_race_releases_probe_and_reselects(build, mode, monkeypatch):
    harness = build(("a",))
    harness.db.connect().execute("UPDATE credentials SET cooling_until = 1")
    reserve = harness.pacer.try_reserve
    declined = False

    def reserve_after_update(key):
        nonlocal declined
        if not declined:
            declined = True
            return None
        return reserve(key)

    monkeypatch.setattr(harness.pacer, "try_reserve", reserve_after_update)
    assert await _run(harness, mode) == "ok"
    assert declined
    assert len(harness.client.calls) == 1
    _released(harness)


async def test_missing_credential_in_scan_does_not_block_other_accounts(build, mode, monkeypatch):
    harness = build()
    original = harness.executor._safe_credential_data

    def credential_data(credential_id):
        return None if credential_id == harness.ids[0] else original(credential_id)

    monkeypatch.setattr(harness.executor, "_safe_credential_data", credential_data)
    assert await _run(harness, mode) == "ok"
    assert harness.client.calls[0].uid == "b"
    _released(harness)


@pytest.mark.parametrize("full", [False, True])
async def test_provider_without_pacer_still_obeys_global_cap(build, mode, full):
    harness = build(("a",), global_cap=1)
    harness.provider.pacer = None
    if full:
        harness.guard.acquire(harness.ids[0])
        if mode == "complete":
            with pytest.raises(NoHealthyCredential):
                await _run(harness, mode)
        else:
            wire = b"".join([frame async for frame in
                             harness.executor.stream_guarded(_request(mode))])
            assert b"no_healthy_credential" in wire
        assert not harness.client.calls
        harness.guard.release(harness.ids[0])
    else:
        assert await _run(harness, mode) == "ok"
    _released(harness)


async def test_provider_without_pacer_or_global_guard_can_serve(build, mode):
    harness = build(("a",))
    harness.provider.pacer = None
    harness.deps.inflight = None
    assert await _run(harness, mode) == "ok"
    _released(harness)


async def test_pinned_retry_that_loses_eligibility_returns_to_pool(build, mode, monkeypatch):
    queued = Event(kind=EventKind.ERROR, error_kind=ErrKind.QUEUE,
                   error_message="queued", retry_after_seconds=0)
    harness = build(scripts=[[queued], _good()])
    pick = harness.executor._pick

    async def pick_after_update(target, tried, affinity_id=None, **options):
        if options.get("only"):
            return None
        return await pick(target, tried, affinity_id, **options)

    monkeypatch.setattr(harness.executor, "_pick", pick_after_update)
    assert await _run(harness, mode) == "ok"
    assert len(harness.client.calls) == 2
    assert harness.client.calls[0].uid != harness.client.calls[1].uid
    _released(harness)
