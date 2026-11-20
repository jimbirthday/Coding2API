"""Admission checks, reservations, and permit lifecycle for the chat pacer."""

from __future__ import annotations

import asyncio
import math
import random
from unittest.mock import Mock

import pytest

from src.tasks.pacer import JITTER_RATIO, Pacer, PacerPermit, paced, stable_key


@pytest.fixture
def clock():
    return {"now": 100.0}


def test_window_delay_is_finite_after_committed_release(clock):
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=1,
                  window_seconds=60, now=lambda: clock["now"])
    assert pacer.ready_in("k") == 0
    assert pacer._starts == pacer._inflight == {}
    permit = pacer.try_reserve("k")
    assert isinstance(permit, PacerPermit)
    assert (permit.pacer, permit.key, permit.started) == (pacer, "k", 100)
    assert not permit.committed and not permit.closed
    permit.commit()
    permit.close()

    clock["now"] = 115
    assert pacer.ready_in("k") == 45
    assert pacer.try_reserve("k") is None
    assert pacer._starts == {"k": [100]}
    assert pacer._inflight == {}
    clock["now"] = 160
    assert pacer.ready_in("k") == 0
    assert pacer._starts == {}
    replacement = pacer.try_reserve("k")
    assert replacement is not None
    replacement.close()


@pytest.mark.parametrize("window", [0, 60])
def test_active_full_is_infinite_even_after_window_expires(clock, window):
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=1,
                  window_seconds=window, now=lambda: clock["now"])
    permit = pacer.try_reserve("k")
    assert permit is not None
    clock["now"] += 120
    assert pacer.ready_in("k") == math.inf
    assert pacer.try_reserve("k") is None
    assert pacer.ready_in("other") == 0
    permit.close()
    assert pacer.ready_in("k") == 0


@pytest.mark.parametrize(("low", "high", "interval"), [
    (4, 4, 4 * (1 + JITTER_RATIO)),
    (2, 7, 7),
    (0, 0, 0),
])
def test_min_interval_uses_jitter_ceiling_or_variable_upper_bound(clock, low, high, interval):
    pacer = Pacer(low, high, allow_concurrent=True, now=lambda: clock["now"])
    first = pacer.try_reserve("k")
    assert first is not None
    assert first.started is None
    assert pacer.ready_in("k") == 0
    second = pacer.try_reserve("k")
    assert second is not None
    first.commit()
    second.commit()
    first.close()
    second.close()
    assert pacer.ready_in("k") == interval
    if interval:
        assert pacer.try_reserve("k") is None
        clock["now"] += interval - 0.25
        assert pacer.ready_in("k") == pytest.approx(0.25)
        clock["now"] += 0.25
    assert pacer.ready_in("k") == 0
    permit = pacer.try_reserve("k")
    assert permit is not None
    permit.commit()
    assert pacer._bucket_started["k"] == clock["now"]
    permit.close()


def test_live_bounds_can_disable_and_reenable_interval(clock):
    bounds = {"low": 4, "high": 4}
    pacer = Pacer(lambda: bounds["low"], lambda: bounds["high"],
                  allow_concurrent=True, now=lambda: clock["now"])
    permit = pacer.try_reserve("k")
    assert permit is not None
    permit.commit()
    permit.close()
    assert pacer.ready_in("k") == 5
    bounds.update(low=1, high=8)
    assert pacer.ready_in("k") == 8
    bounds.update(low=0, high=0)
    assert pacer.disabled
    assert pacer.ready_in("k") == 0
    permit = pacer.try_reserve("k")
    assert permit is not None
    permit.commit()
    permit.close()
    bounds.update(low=2, high=2)
    assert not pacer.disabled
    assert pacer.ready_in("k") == 2.5


@pytest.mark.parametrize(("low", "high", "message"), [
    (-1, 1, "non-negative"),
    (1, -1, "non-negative"),
    (2, 1, "min must not exceed max"),
])
def test_admission_revalidates_live_bounds(clock, low, high, message):
    bounds = {"low": 0, "high": 0}
    pacer = Pacer(lambda: bounds["low"], lambda: bounds["high"],
                  allow_concurrent=True, now=lambda: clock["now"])
    permit = pacer.try_reserve("k")
    assert permit is not None
    permit.commit()
    permit.close()
    bounds.update(low=low, high=high)
    with pytest.raises(ValueError, match=message):
        pacer.ready_in("k")
    with pytest.raises(ValueError, match=message):
        pacer.try_reserve("k")
    assert pacer._inflight == {}


@pytest.mark.parametrize("window", [0, 60])
@pytest.mark.parametrize("previous_start", [False, True])
def test_uncommitted_rollback_creates_no_phantom_spacing(clock, window, previous_start):
    pacer = Pacer(4, 4, allow_concurrent=True, max_concurrency=2,
                  window_seconds=window, now=lambda: clock["now"])
    if previous_start:
        previous = pacer.try_reserve("k")
        assert previous is not None
        previous.commit()
        previous.close()
        clock["now"] += 10
    expected_starts = {"k": 100} if previous_start else {}
    permit = pacer.try_reserve("k")
    assert permit is not None
    assert pacer._bucket_started == expected_starts
    clock["now"] += 1
    permit.close()
    assert pacer._bucket_started == expected_starts
    assert pacer._inflight == {}
    assert pacer.ready_in("k") == 0
    replacement = pacer.try_reserve("k")
    assert replacement is not None
    replacement.close()


@pytest.mark.parametrize("window", [0, 60])
@pytest.mark.parametrize("rollback_first", [False, True])
def test_rollback_preserves_another_concurrent_committed_start(clock, window, rollback_first):
    pacer = Pacer(4, 4, allow_concurrent=True, max_concurrency=2,
                  window_seconds=window, now=lambda: clock["now"])
    abandoned = pacer.try_reserve("k")
    assert abandoned is not None
    clock["now"] = 101
    started = pacer.try_reserve("k")
    assert started is not None
    clock["now"] = 102
    started.commit()
    assert pacer._bucket_started == {"k": 102}
    clock["now"] = 103
    if rollback_first:
        abandoned.close()
        started.close()
    else:
        started.close()
        abandoned.close()
    assert pacer._inflight == {}
    assert pacer._bucket_started == {"k": 102}
    if window:
        assert pacer._starts == {"k": [101]}
    assert pacer.ready_in("k") == 4
    assert pacer.try_reserve("k") is None
    clock["now"] = 107
    assert pacer.ready_in("k") == 0


def test_live_limit_reduction_waits_for_enough_window_entries(clock):
    settings = {"limit": 3, "window": 60}
    pacer = Pacer(0, 0, allow_concurrent=True,
                  max_concurrency=lambda: settings["limit"],
                  window_seconds=lambda: settings["window"], now=lambda: clock["now"])
    for started in (100, 110, 120):
        clock["now"] = started
        permit = pacer.try_reserve("k")
        assert permit is not None
        permit.commit()
        permit.close()
    clock["now"] = 130
    settings["limit"] = 2
    assert pacer.ready_in("k") == 40
    clock["now"] = 160
    assert pacer.ready_in("k") == 10
    settings["window"] = 50
    assert pacer.ready_in("k") == 0
    assert pacer._starts == {"k": [120]}


@pytest.mark.parametrize("disabled", [0, -1])
def test_live_zero_or_negative_limits_and_windows_disable_gates(clock, disabled):
    settings = {"limit": 1, "window": 60}
    pacer = Pacer(0, 0, allow_concurrent=True,
                  max_concurrency=lambda: settings["limit"],
                  window_seconds=lambda: settings["window"], now=lambda: clock["now"])
    first = pacer.try_reserve("k")
    assert first is not None
    assert pacer.ready_in("k") == math.inf
    settings["limit"] = disabled
    assert pacer.ready_in("k") == 0
    second = pacer.try_reserve("k")
    assert second is not None and second.started is None
    settings["limit"] = 1
    assert pacer.ready_in("k") == math.inf
    second.close()
    first.commit()
    first.close()
    assert pacer.ready_in("k") == 60
    settings["window"] = disabled
    assert pacer.ready_in("k") == 0
    assert pacer._starts == {}
    third = pacer.try_reserve("k")
    assert third is not None and third.started is None
    third.close()
    settings["window"] = 60
    fourth = pacer.try_reserve("k")
    assert fourth is not None and fourth.started == 100
    fourth.close()


def test_duplicate_bucket_keys_share_capacity_and_close_is_idempotent(clock):
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=2,
                  window_seconds=60, now=lambda: clock["now"])
    key = stable_key("provider", "same-identity")
    duplicate = stable_key("provider", "same-identity")
    first = pacer.try_reserve(key)
    second = pacer.try_reserve(duplicate)
    assert first is not None and second is not None
    assert pacer._starts[key] == [100, 100]
    assert pacer.try_reserve(duplicate) is None
    first.close()
    assert pacer._starts[key] == [100]
    assert pacer._inflight[key] == 1
    pacer._wake.clear()
    first.close()
    assert pacer._inflight[key] == 1
    assert pacer._starts[key] == [100]
    assert not pacer._wake.is_set()
    second.commit()
    second.close()
    assert second.committed and second.closed
    assert pacer._starts[key] == [100]
    assert pacer._inflight == {}


@pytest.mark.parametrize("replace", [False, True])
def test_uncommitted_rollback_after_window_prune_preserves_new_starts(clock, replace):
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=2,
                  window_seconds=10, now=lambda: clock["now"])
    old = pacer.try_reserve("k")
    assert old is not None
    clock["now"] += 10
    assert pacer.ready_in("k") == 0
    assert "k" not in pacer._starts
    replacement = pacer.try_reserve("k") if replace else None
    old.close()
    assert old.closed and not old.committed
    if replacement is not None:
        assert pacer._starts == {"k": [110]}
        assert pacer._inflight == {"k": 1}
        replacement.close()
    else:
        assert pacer._starts == pacer._inflight == {}
    assert pacer._inflight == {}


async def test_release_wakes_all_waiters_and_clears_stale_notification():
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=1)
    permit = pacer.try_reserve("k")
    assert permit is not None
    pacer._wake.set()
    async with asyncio.TaskGroup() as group:
        waiters = [group.create_task(pacer.wait_for_change(10)) for _ in range(2)]
        await asyncio.sleep(0)
        assert all(not waiter.done() for waiter in waiters)
        assert not pacer._wake.is_set()
        permit.close()
    assert all(waiter.result() is None for waiter in waiters)
    assert pacer.ready_in("k") == 0


@pytest.mark.parametrize("timeout", [0, 0.001])
async def test_wait_for_change_timeout_returns_without_changing_capacity(timeout):
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=1)
    permit = pacer.try_reserve("k")
    assert permit is not None
    try:
        assert await asyncio.wait_for(pacer.wait_for_change(timeout), 1) is None
        assert pacer.ready_in("k") == math.inf
        assert not permit.closed
    finally:
        permit.close()


async def test_cancelled_waiter_propagates_and_next_waiter_can_wake():
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=1)
    permit = pacer.try_reserve("k")
    assert permit is not None
    waiter = asyncio.create_task(pacer.wait_for_change(10))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not permit.closed
    assert pacer.ready_in("k") == math.inf
    async with asyncio.TaskGroup() as group:
        next_waiter = group.create_task(pacer.wait_for_change(10))
        await asyncio.sleep(0)
        permit.close()
    assert next_waiter.result() is None


async def test_cancellation_before_commit_rolls_back_reserved_window(clock):
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=1,
                  window_seconds=60, now=lambda: clock["now"])
    entered = asyncio.Event()

    async def prepare():
        permit = pacer.try_reserve("k")
        assert permit is not None
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            permit.close()

    task = asyncio.create_task(prepare())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pacer._inflight == {}
    assert pacer._starts == {"k": []}
    assert pacer.ready_in("k") == 0


@pytest.mark.parametrize("with_pacer", [False, True])
async def test_paced_without_permit_still_acquires_and_releases(with_pacer):
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=1)
    async with paced(pacer if with_pacer else None, "k"):
        assert pacer._inflight == ({"k": 1} if with_pacer else {})
    assert pacer._inflight == {}


@pytest.mark.parametrize("exit_kind", ["normal", "error", "cancel"])
@pytest.mark.parametrize("with_pacer", [False, True])
async def test_paced_permit_commits_once_and_keeps_window_on_every_exit(
    clock, monkeypatch, exit_kind, with_pacer,
):
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=1,
                  window_seconds=60, now=lambda: clock["now"])
    permit = pacer.try_reserve("k")
    assert permit is not None
    wait_turn = Mock(side_effect=AssertionError("permit must bypass wait_turn"))
    monkeypatch.setattr(pacer, "wait_turn", wait_turn)
    entered = asyncio.Event()

    async def consume():
        async with paced(pacer if with_pacer else None, "unused-key", permit=permit):
            assert permit.committed and not permit.closed
            assert pacer._inflight == {"k": 1}
            entered.set()
            if exit_kind == "error":
                raise RuntimeError("upstream failed")
            if exit_kind == "cancel":
                await asyncio.Event().wait()

    if exit_kind == "cancel":
        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif exit_kind == "error":
        with pytest.raises(RuntimeError, match="upstream failed"):
            await consume()
    else:
        await consume()
    assert permit.closed and permit.committed
    assert pacer._inflight == {}
    assert pacer._starts == {"k": [100]}
    assert pacer.ready_in("k") == 60
    wait_turn.assert_not_called()


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_random_windows_match_committed_start_history(clock, seed):
    rng = random.Random(seed)
    limit = rng.randint(1, 5)
    window = rng.uniform(1, 30)
    pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=limit,
                  window_seconds=window, now=lambda: clock["now"])
    history = {"a": [], "b": []}
    for _ in range(200):
        clock["now"] += rng.choice([0, window / 10, window / 2, window])
        key = rng.choice(["a", "b"])
        recent = [start for start in history[key] if clock["now"] - start < window]
        expected = 0
        if len(recent) >= limit:
            expected = max(0, recent[-limit] + window - clock["now"])
        assert pacer.ready_in(key) == pytest.approx(expected)
        permit = pacer.try_reserve(key)
        if expected > 0:
            assert permit is None
            continue
        assert permit is not None
        if rng.choice([True, False]):
            permit.commit()
            history[key].append(clock["now"])
        permit.close()
        assert pacer._inflight == {}
        assert pacer.ready_in(key) >= 0
