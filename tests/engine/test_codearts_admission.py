"""CodeArts admission through the real executor, provider, pacer and SQLite repo."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.compat.openai.errors import UpstreamStreamError
from src.compat.openai.request import parse_chat_request
from src.engine import executor as executor_module
from src.engine.affinity import ConversationAffinity
from src.engine.executor import Executor, ExecutorDeps, NoHealthyCredential
from src.engine.half_open import HalfOpenGuard
from src.engine.inflight import InflightGuard
from src.engine.scheduler import Scheduler
from src.provider.base import ErrKind, Event, EventKind
from src.provider.codearts import CodeArtsProvider
from src.tasks.pacer import Pacer

pytestmark = pytest.mark.parametrize("mode", ["stream", "complete"])


def _good(text="ok", finish="stop"):
    return [Event(kind=EventKind.CONTENT, content=text),
            Event(kind=EventKind.FINISH, finish_reason=finish)]


class _Client:
    def __init__(self, pacer, scripts):
        self.pacer = pacer
        self.scripts = scripts
        self.calls = []
        self.closed = []
        self.entered = asyncio.Queue()

    async def stream_chat(self, credential, payload, model):
        index = len(self.calls)
        self.calls.append(SimpleNamespace(
            uid=credential.uid, ak=credential.access_key_id,
            payload=deepcopy(payload), model=model,
            inflight=dict(self.pacer._inflight), starts=deepcopy(self.pacer._starts)))
        self.entered.put_nowait(index)
        try:
            script = self.scripts[index] if index < len(self.scripts) else _good()
            for item in script:
                if isinstance(item, asyncio.Event):
                    await item.wait()
                elif isinstance(item, BaseException):
                    raise item
                else:
                    yield item
        finally:
            self.closed.append(index)


def _request(mode, conversation="admission"):
    return parse_chat_request({
        "model": "deepseek-v4-flash-0731", "stream": mode == "stream",
        "conversation_id": conversation, "reasoning_effort": "high",
        "messages": [{"role": "user", "content": "hello"}],
    })


async def _run(h, mode, request=None, *, finish_reason=None):
    request = request or _request(mode)
    if mode == "complete":
        result = await h.executor.complete(request, username="alice")
        if finish_reason is not None:
            assert result["choices"][0]["finish_reason"] == finish_reason
        return result["choices"][0]["message"]["content"]
    frames = [frame async for frame in h.executor.stream_guarded(request, username="alice")]
    wire = b"".join(frames).decode()
    assert "[DONE]" in wire
    chunks = [json.loads(line[6:]) for line in wire.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert not any("error" in chunk for chunk in chunks), wire
    if finish_reason is not None:
        assert [choice["finish_reason"] for chunk in chunks for choice in chunk.get("choices", [])
                if choice.get("finish_reason") is not None] == [finish_reason]
    return "".join(chunk["choices"][0]["delta"].get("content", "")
                   for chunk in chunks if chunk.get("choices"))


@pytest.fixture()
async def build(repo, monkeypatch):
    credentials, db = repo
    harnesses = []

    def make(uids=("a", "b"), *, window=60, maximum=2, global_cap=0, scripts=()):
        clock = [100.0]

        async def advance(seconds):
            clock[0] += seconds
            await asyncio.sleep(0)

        pacer = Pacer(0, 0, allow_concurrent=True, max_concurrency=maximum,
                      window_seconds=window, now=lambda: clock[0], sleep=advance)
        client = _Client(pacer, scripts)
        provider = CodeArtsProvider(client=client, pacer=pacer)
        ids, keys = [], []
        for index, uid in enumerate(uids):
            data = {"uid": uid, "access_key_id": f"AK-{index}", "secret_access_key": "SK"}
            ids.append(credentials.add(provider="codearts", credential_data=data))
            keys.append(provider.chat_bucket(data)[1])
        affinity = ConversationAffinity(ttl_seconds=600)
        guard = InflightGuard(lambda: global_cap)
        half_open = HalfOpenGuard()
        deps = ExecutorDeps(providers={"codearts": provider}, credentials=credentials,
                            scheduler=Scheduler(), affinity=affinity, inflight=guard,
                            half_open=half_open)
        executor = Executor(deps)
        waiting = asyncio.Queue()
        original_wait = pacer.wait_for_change

        async def observe_wait(timeout):
            waiting.put_nowait(timeout)
            await original_wait(timeout)

        monkeypatch.setattr(pacer, "wait_for_change", observe_wait)
        h = SimpleNamespace(
            executor=executor, deps=deps, provider=provider, pacer=pacer, client=client,
            ids=ids, keys=keys, affinity=affinity, guard=guard, half_open=half_open,
            clock=clock, waiting=waiting, held=[], tasks=[], credentials=credentials, db=db)
        harnesses.append(h)
        return h

    yield make
    for h in harnesses:
        for task in h.tasks:
            task.cancel()
        await asyncio.gather(*h.tasks, return_exceptions=True)
        for permit in h.held:
            permit.close()


def _start(h, mode, request=None):
    task = asyncio.create_task(_run(h, mode, request))
    h.tasks.append(task)
    return task


def _hold(h, key, count=2, *, committed=False):
    permits = []
    for _ in range(count):
        permit = h.pacer.try_reserve(key)
        assert permit is not None
        if committed:
            permit.commit()
        h.held.append(permit)
        permits.append(permit)
    return permits


async def _next(queue):
    return await asyncio.wait_for(queue.get(), timeout=1)


def _released(h):
    assert h.pacer._inflight == {}
    assert h.guard.total() == 0
    assert h.executor._pending_probe == {}
    assert h.executor._probe_leases == {}
    assert h.half_open._probing == {}


async def test_saturated_sticky_reroutes_and_repins_after_success(build, mode):
    gate = asyncio.Event()
    h = build(scripts=[[gate, *_good()]])
    request = _request(mode)
    h.affinity.remember(request.raw, "alice", h.ids[0])
    held = _hold(h, h.keys[0])
    task = _start(h, mode, request)
    await _next(h.client.entered)
    assert h.client.calls[0].uid == "b"
    assert h.affinity.pin_for(request.raw, "alice") == h.ids[0]
    gate.set()
    assert await asyncio.wait_for(task, 1) == "ok"
    assert h.affinity.pin_for(request.raw, "alice") == h.ids[1]
    for permit in held:
        permit.close()
    _released(h)


async def test_window_full_without_inflight_is_skipped(build, mode):
    h = build()
    for permit in _hold(h, h.keys[0], committed=True):
        permit.close()
    assert h.pacer._inflight == {}
    assert h.pacer.ready_in(h.keys[0]) == 60
    h.affinity.remember(_request(mode).raw, "alice", h.ids[0])
    assert await _run(h, mode) == "ok"
    assert h.client.calls[0].uid == "b"
    assert len(h.pacer._starts[h.keys[0]]) == 2
    _released(h)


async def test_burst_spreads_accounts_and_never_exceeds_two(build, mode):
    gate = asyncio.Event()
    h = build(window=0, scripts=[[gate, *_good()] for _ in range(10)])
    tasks = [_start(h, mode, _request(mode, str(i))) for i in range(10)]
    for _ in range(4):
        await _next(h.client.entered)
    await _next(h.waiting)
    assert len(h.client.calls) == 4
    assert Counter(call.uid for call in h.client.calls) == {"a": 2, "b": 2}
    assert h.pacer._inflight == dict.fromkeys(h.keys, 2)
    assert h.guard.total() == 4
    gate.set()
    assert await asyncio.wait_for(asyncio.gather(*tasks), 1) == ["ok"] * 10
    assert all(0 < count <= 2 for call in h.client.calls for count in call.inflight.values())
    assert len(h.client.closed) == 10
    _released(h)


async def test_duplicate_credentials_for_same_uid_share_limits(build, mode):
    gate = asyncio.Event()
    h = build(("same", "same", "other"), scripts=[[gate, *_good()] for _ in range(3)])
    assert h.keys[0] == h.keys[1] != h.keys[2]
    # Prefer both duplicate rows so the idle third account is only used on saturation.
    h.db.connect().execute("UPDATE credentials SET pinned = 1 WHERE id IN (?, ?)", h.ids[:2])
    tasks = [_start(h, mode, _request(mode, str(i))) for i in range(3)]
    for _ in tasks:
        await _next(h.client.entered)
    assert Counter(call.uid for call in h.client.calls) == {"same": 2, "other": 1}
    assert len({call.ak for call in h.client.calls if call.uid == "same"}) == 2
    assert h.pacer._inflight[h.keys[0]] == 2
    gate.set()
    assert await asyncio.wait_for(asyncio.gather(*tasks), 1) == ["ok"] * 3
    _released(h)


async def test_pool_wait_wakes_for_any_account_without_leases(build, mode, caplog):
    caplog.set_level(logging.INFO)
    h = build()
    held = [_hold(h, key) for key in h.keys]
    h.executor._clock = lambda: h.clock[0]
    h.affinity.remember(_request(mode).raw, "alice", h.ids[0])
    task = _start(h, mode)
    await _next(h.waiting)
    assert not task.done()
    assert not h.client.calls
    assert h.guard.total() == 0
    assert not h.executor._pending_probe
    h.clock[0] += 0.125
    held[1][0].close()
    assert await asyncio.wait_for(task, 1) == "ok"
    assert h.client.calls[0].uid == "b"
    assert h.affinity.pin_for(_request(mode).raw, "alice") == h.ids[1]
    messages = [record.getMessage() for record in caplog.records]
    assert any("capacity_wait_ms=125.0" in message for message in messages)
    timing = next(message for message in messages if "upstream_first_ms=" in message)
    values = re.search(
        r"throttle_wait_ms=([\d.]+) upstream_first_ms=([\d.]+) generation_ms=([\d.]+)", timing)
    assert values is not None
    assert all(float(value) >= 0 for value in values.groups())
    for permits in held:
        for permit in permits:
            permit.close()
    _released(h)


async def test_window_full_pool_waits_until_slot_expires(build, mode):
    h = build(("a",))
    for permit in _hold(h, h.keys[0], committed=True):
        permit.close()
    task = _start(h, mode)
    await _next(h.waiting)
    assert h.guard.total() == 0
    assert h.pacer._inflight == {}
    h.clock[0] += 60
    h.pacer._wake.set()
    assert await asyncio.wait_for(task, 1) == "ok"
    assert h.pacer._starts[h.keys[0]] == [160.0]
    _released(h)


async def test_pool_wait_timeout_is_bounded_and_logs(build, mode, monkeypatch, caplog):
    monkeypatch.setattr(executor_module, "CHAT_CAPACITY_WAIT_SECONDS", 0.01)
    caplog.set_level(logging.INFO)
    h = build(("a",))
    held = _hold(h, h.keys[0])
    before = deepcopy(h.pacer._starts)
    if mode == "complete":
        with pytest.raises(NoHealthyCredential):
            await asyncio.wait_for(_run(h, mode), 1)
    else:
        wire = b"".join([frame async for frame in h.executor.stream_guarded(_request(mode))])
        assert b"no_healthy_credential" in wire
        assert b"[DONE]" in wire
    assert not h.client.calls
    assert h.guard.total() == 0
    assert h.pacer._starts == before
    assert h.pacer._inflight == {h.keys[0]: 2}
    record = next(record for record in caplog.records if "capacity_wait_ms=" in record.msg)
    assert "超时" in record.getMessage()
    assert record.args[0] >= 10
    for permit in held:
        permit.close()
    _released(h)


async def test_cancellation_during_pool_wait_does_not_leak(build, mode):
    h = build(("a",))
    held = _hold(h, h.keys[0])
    before = deepcopy(h.pacer._starts)
    task = _start(h, mode)
    await _next(h.waiting)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.guard.total() == 0
    assert h.pacer._inflight == {h.keys[0]: 2}
    assert h.pacer._starts == before
    assert not h.client.calls
    for permit in held:
        permit.close()
    assert await _run(h, mode) == "ok"
    _released(h)


@pytest.mark.parametrize("paced", [True, False], ids=["permit", "no-permit"])
async def test_preparation_failure_rolls_back_permit_and_half_open_probe(build, mode, paced):
    h = build(("a",))
    if not paced:
        h.provider.pacer = None
    h.db.connect().execute("UPDATE credentials SET cooling_until = 1 WHERE id = ?", h.ids)

    def fail_compress(provider, model, payload):
        assert provider == "codearts"
        assert h.pacer._inflight == ({h.keys[0]: 1} if paced else {})
        assert h.half_open.acquired((h.ids[0], ""))
        raise RuntimeError("compression failed")

    h.deps.context_compress = fail_compress
    if mode == "complete":
        with pytest.raises(RuntimeError, match="compression failed"):
            await _run(h, mode)
    else:
        wire = b"".join([frame async for frame in h.executor.stream_guarded(_request(mode))])
        assert b"internal_error" in wire
    assert not h.client.calls
    assert not any(h.pacer._starts.values())
    _released(h)
    h.deps.context_compress = None
    assert await _run(h, mode) == "ok"
    assert h.half_open.recovered((h.ids[0], ""))
    _released(h)


async def test_source_cancellation_before_first_data_releases_permit(build, mode):
    gate = asyncio.Event()
    h = build(("a",), scripts=[[gate, *_good()]])
    task = _start(h, mode)
    await _next(h.client.entered)
    assert h.guard.total() == 1
    assert h.pacer._inflight == {h.keys[0]: 1}
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.client.closed == [0]
    # The upstream started, so cancellation returns inflight but consumes its window slot.
    assert h.pacer._starts[h.keys[0]] == [100.0]
    _released(h)
    assert await _run(h, mode) == "ok"
    assert len(h.pacer._starts[h.keys[0]]) == 2
    _released(h)


async def test_stream_error_rotation_releases_previous_account(build, mode):
    error = Event(kind=EventKind.ERROR, error_kind=ErrKind.SOFT, error_message="retry")
    h = build(scripts=[[error], _good()])
    h.affinity.remember(_request(mode).raw, "alice", h.ids[0])
    assert await _run(h, mode) == "ok"
    assert [call.uid for call in h.client.calls] == ["a", "b"]
    assert h.client.calls[1].inflight == {h.keys[1]: 1}
    assert h.client.closed == [0, 1]
    assert {key: len(starts) for key, starts in h.pacer._starts.items()} == dict.fromkeys(h.keys, 1)
    _released(h)


@pytest.mark.parametrize("retry", ["queue", "effort"])
@pytest.mark.parametrize("raised", [False, True], ids=["event", "exception"])
async def test_retries_keep_account_but_reacquire_window_slot(build, mode, retry, raised):
    error = Event(kind=EventKind.ERROR,
                  error_kind=ErrKind.QUEUE if retry == "queue" else ErrKind.INVALID,
                  error_message="queued" if retry == "queue" else "unsupported reasoning_effort",
                  retry_after_seconds=0)
    exception = UpstreamStreamError(error)
    exception.retry_after_seconds = 0
    h = build(scripts=[[exception if raised else error], _good()])
    h.affinity.remember(_request(mode).raw, "alice", h.ids[0])
    assert await _run(h, mode) == "ok"
    assert [call.uid for call in h.client.calls] == ["a", "a"]
    assert [call.inflight for call in h.client.calls] == [{h.keys[0]: 1}] * 2
    assert [len(call.starts[h.keys[0]]) for call in h.client.calls] == [1, 2]
    assert [call.payload["reasoning_effort"] for call in h.client.calls] == (
        ["high", "high"] if retry == "queue" else ["high", "medium"])
    assert h.client.closed == [0, 1]
    assert h.credentials.candidates(["codearts"])[0].err_count == 0
    _released(h)


async def test_continuation_uses_initial_permit_once_then_fresh_window_slot(build, mode):
    h = build(("a",), scripts=[_good("first", "length"), _good("second")])
    h.deps.max_auto_continues = 1
    assert await _run(h, mode) == "firstsecond"
    assert [call.uid for call in h.client.calls] == ["a", "a"]
    assert [call.inflight for call in h.client.calls] == [{h.keys[0]: 1}] * 2
    assert [len(call.starts[h.keys[0]]) for call in h.client.calls] == [1, 2]
    assert h.client.calls[1].payload["messages"][-2] == {"role": "assistant", "content": "first"}
    assert h.client.closed == [0, 1]
    assert len(h.pacer._starts[h.keys[0]]) == 2
    _released(h)


async def test_continuation_full_window_preserves_partial_output_without_waiting(build, mode):
    h = build(maximum=1, window=60, scripts=[_good("partial", "length")])
    h.deps.max_auto_continues = 1
    h.affinity.remember(_request(mode).raw, "alice", h.ids[0])
    assert await asyncio.wait_for(_run(h, mode, finish_reason="length"), 0.2) == "partial"
    assert [call.uid for call in h.client.calls] == ["a"]
    assert h.client.closed == [0]
    assert h.pacer._starts == {h.keys[0]: [100.0]}
    assert h.pacer.ready_in(h.keys[0]) == 60
    assert h.clock[0] == 100.0
    assert h.waiting.empty()
    for candidate in h.credentials.candidates(["codearts"]):
        assert candidate.cooling_until is None
        assert not candidate.model_cooldowns
        assert candidate.err_count == 0
        assert not candidate.disabled
    _released(h)


async def test_continuation_without_pacer_completes_without_permits(build, mode, monkeypatch):
    h = build(("a",), scripts=[_good("first", "length"), _good("second")])
    h.provider.pacer = None
    h.deps.max_auto_continues = 1

    def unexpected_reserve(key):
        pytest.fail("CodeArts without a pacer must not reserve a permit")

    monkeypatch.setattr(h.pacer, "try_reserve", unexpected_reserve)
    assert await _run(h, mode, finish_reason="stop") == "firstsecond"
    assert [call.uid for call in h.client.calls] == ["a", "a"]
    assert [call.inflight for call in h.client.calls] == [{}, {}]
    assert [call.starts for call in h.client.calls] == [{}, {}]
    assert h.client.closed == [0, 1]
    assert h.waiting.empty()
    _released(h)


async def test_continuation_cancellation_before_source_starts_rolls_back_permit(
    build, mode, monkeypatch,
):
    h = build(("a",), scripts=[_good("partial", "length")])
    h.deps.max_auto_continues = 1
    original_stream = CodeArtsProvider.stream_chat
    permits = []

    def cancel_continuation(provider, credential_data, payload, model, *, permit=None):
        assert permit is not None
        permits.append(permit)
        if len(permits) == 2:
            assert permits[0].closed
            assert not permit.committed
            assert h.pacer._inflight == {h.keys[0]: 1}
            raise asyncio.CancelledError("before continuation source starts")
        return original_stream(provider, credential_data, payload, model, permit=permit)

    monkeypatch.setattr(CodeArtsProvider, "stream_chat", cancel_continuation)
    with pytest.raises(asyncio.CancelledError, match="before continuation source starts"):
        await _run(h, mode)
    assert len(permits) == 2
    assert permits[0] is not permits[1]
    assert permits[0].committed
    assert not permits[1].committed
    assert all(permit.closed for permit in permits)
    assert len(h.client.calls) == 1
    assert h.client.closed == [0]
    assert h.pacer._starts == {h.keys[0]: [100.0]}
    _released(h)


async def test_global_cap_waits_without_extra_lease_then_resumes_on_release(build, mode):
    gate = asyncio.Event()
    h = build(("a",), global_cap=1, scripts=[[gate, *_good()], _good()])
    first = _start(h, mode, _request(mode, "first"))
    await _next(h.client.entered)
    second = _start(h, mode, _request(mode, "second"))
    await _next(h.waiting)
    assert not second.done()
    assert len(h.client.calls) == 1
    assert h.guard.total() == 1
    assert h.pacer._inflight == {h.keys[0]: 1}
    assert len(h.pacer._starts[h.keys[0]]) == 1
    gate.set()
    # Provider release can precede executor lease release; admission also polls every second.
    assert await asyncio.wait_for(asyncio.gather(first, second), 2) == ["ok", "ok"]
    assert [call.inflight for call in h.client.calls] == [{h.keys[0]: 1}] * 2
    assert h.client.closed == [0, 1]
    _released(h)


async def test_global_cap_wait_cancellation_keeps_active_request_intact(build, mode):
    gate = asyncio.Event()
    h = build(("a",), global_cap=1, scripts=[[gate, *_good()]])
    first = _start(h, mode, _request(mode, "first"))
    await _next(h.client.entered)
    second = _start(h, mode, _request(mode, "second"))
    await _next(h.waiting)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert h.guard.total() == 1
    assert h.pacer._inflight == {h.keys[0]: 1}
    assert len(h.pacer._starts[h.keys[0]]) == 1
    assert not h.client.closed
    gate.set()
    assert await asyncio.wait_for(first, 1) == "ok"
    assert h.client.closed == [0]
    _released(h)
