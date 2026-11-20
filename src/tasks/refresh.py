"""token 预刷新：只处理进入 REFRESH_SKEW_HOURS 窗口的凭证。

终态识别：一次性 refresh_token 被消费后，每轮重试用的都是同一张废票，
表现为「每小时刷一条一模一样的 warning」而凭证其实早就死了。上游明确说
票据作废时（`UpstreamReloginRequired`）改写 disabled_reason 并停手，把处置
交给人；其余失败照旧只记 warning，等下一轮。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from ..db.crypto import CredentialDecryptError
from ..db.repo import CredentialRepository
from ..provider.base import UpstreamReloginRequired
from ..provider.token_expiry import credential_token_times
from ..request_context import credential_scope
from . import TaskReport
from .pacer import Pacer

logger = logging.getLogger(__name__)

# 落库的「需重新登录」原因：固定文案，绝不写上游原文（见 run_once 注释）。
RELOGIN_REASON = "刷新令牌已失效，需重新登录"


class RefreshTask:
    """token 预刷新：只处理进入 refresh_skew 窗口的凭证。"""

    def __init__(self, credentials: CredentialRepository, providers: dict,
                 *, skew_seconds: int, now: Callable[[], int] | None = None,
                 pacer: Pacer | None = None) -> None:
        self._credentials = credentials
        self._providers = providers
        self.skew_seconds = skew_seconds
        self._now = now or (lambda: int(time.time()))
        self._pacer = pacer

    @staticmethod
    def _outcome(*, ok: bool = True, attempted: bool = False,
                 skipped: bool = False, refreshed: bool = False,
                 rotated: bool = False, expires_at: int = 0,
                 revived: bool = False, reason: str = "",
                 error: str = "") -> dict[str, Any]:
        """刷新结果：内部字段（attempted / skipped）供周期任务计数。

        失败原因用稳定短码而不是上游原文：`refresh` 的一次性票被日志、审计和
        管理台同时消费，原文里可能带 token 片段。
        """
        return {"ok": ok, "attempted": attempted, "skipped": skipped,
                "refreshed": refreshed, "rotated": rotated,
                "expires_at": expires_at, "revived": revived,
                "reason": reason, "error": error}

    @staticmethod
    def _refresh_token(data: dict[str, Any]) -> str:
        value = data.get("refresh_token") or data.get("refreshToken") or ""
        return value if isinstance(value, str) else ""

    async def refresh_credential(self, credential_id: str, *, force: bool = True,
                                 revive: bool = True) -> dict[str, Any]:
        """手动刷新单条凭证（管理台「RT 刷新」/ 批量刷新）。

        `force=True` 时不再看 `needs_refresh` 窗口：操作员明确要求用 RT 换新
        token，哪怕当前 access token 还没到期也执行（例如刚导入的 JSON 已经
        过期，或想提前把即将到期的号统一续上）。

        并发安全依赖两层：
        1. `edit_lock` 与刷新任务、账号切换互斥；
        2. 进入锁前记录整份凭证快照，持锁后重读；若已被别的刷新改过就直接
           返回 `already_refreshed`，绝不拿同一张一次性 RT 再刷一次。

        成功后先 `save_credential_data` 再 `revive`：前者回写轮转后的 RT，
        后者解除硬禁用 / 清本地冷却（上游点名的冷却仍保留）。顺序反了会出现
        「凭证已回池，但库里还是旧 RT」的窗口。
        """
        provider_id = self._credentials.provider_of(credential_id)
        provider = self._providers.get(provider_id or "")
        if provider is None:
            return self._outcome(ok=False, error="credential not found")
        try:
            data = self._credentials.credential_data(credential_id)
        except CredentialDecryptError:
            logger.error("刷新跳过 %s：凭证解密失败（APP_SECRET 是否被更换过？）",
                         credential_id)
            return self._outcome(ok=False, error="credential decryption failed")
        if data is None:
            return self._outcome(ok=False, error="credential not found")
        return await self._refresh_locked(credential_id, provider, data,
                                          force=force, revive=revive, manual=True)

    async def _refresh_locked(self, credential_id: str, provider: Any,
                              data: dict[str, Any], *, force: bool,
                              revive: bool, manual: bool) -> dict[str, Any]:
        """刷新核心：窗口判定、快照去重、锁内重读、落库与终态处置。

        `manual=False` 是周期任务路径：不支持刷新 / 没有 RT 的渠道按 skipped
        计数，不把「本来就刷不了」记成失败。
        """
        refresher = getattr(provider, "refresh", None)
        if not callable(refresher):
            return self._outcome(
                ok=not manual, skipped=not manual,
                error="refresh not supported" if manual else "",
                reason="not_supported" if not manual else "",
            )
        if manual and not self._refresh_token(data):
            return self._outcome(ok=False, error="credential has no refresh_token")
        if not force and not _needs_refresh(provider, data, self.skew_seconds, self._now()):
            return self._outcome(skipped=True, reason="not_needed")

        # 快照放在锁外：两个并发手动刷新都读到旧值，先拿到锁的那个刷新并
        # 轮转 RT；后一个持锁重读时快照必然不等，直接返回已刷新，避免烧第二张票。
        snapshot = dict(data)
        if self._pacer is not None:
            await self._pacer.wait_turn()
        async with self._credentials.edit_lock(credential_id):
            try:
                current = self._credentials.credential_data(credential_id)
            except CredentialDecryptError:
                logger.error("刷新跳过 %s：凭证解密失败（APP_SECRET 是否被更换过？）",
                             credential_id)
                return self._outcome(attempted=True, skipped=True,
                                     error="credential decryption failed")
            if current is None:
                return self._outcome(attempted=True, skipped=True,
                                     error="credential not found")
            if current != snapshot:
                return self._outcome(skipped=True, reason="already_refreshed")
            try:
                with credential_scope(credential_id):
                    refreshed = await refresher(current)
            except UpstreamReloginRequired as error:
                # 一次性 RT 已被上游判死：重试只会继续污染状态，标成需重新登录。
                logger.warning(
                    "refresh for %s needs re-login, disabling credential: %s",
                    credential_id, error)
                self._credentials.mark_relogin_required(credential_id, RELOGIN_REASON)
                return self._outcome(attempted=True, ok=False,
                                     error="relogin_required")
            except Exception as error:  # noqa: BLE001 - 单条失败不拖垮批量
                # 上游异常文本可能含凭证片段，只进服务端日志；对外回稳定短码。
                logger.warning("refresh failed for %s: %s", credential_id, error)
                return self._outcome(attempted=True, ok=False, error="refresh_failed")
            # 顺序约束：必须先持久化轮转后的 RT，再解除硬禁用 / 触发后续探测。
            self._credentials.save_credential_data(credential_id, refreshed)
            revived = revive and self._credentials.revive(credential_id)
            _, expires_at = credential_token_times(refreshed)
            return self._outcome(
                attempted=True, refreshed=True,
                rotated=self._refresh_token(refreshed) != self._refresh_token(current),
                expires_at=expires_at, revived=revived,
            )

    async def run_once(self) -> TaskReport:
        report = TaskReport()
        for candidate in self._credentials.candidates():
            # 硬禁用与用户软暂停（enabled=0）都跳过：暂停凭证不该被预刷新占用
            # 调度窗口，线上失败的也只会是早已停用的旧凭证（掩盖真实状态）。
            if candidate.disabled or not candidate.enabled:
                report.skipped += 1
                continue
            provider = self._providers.get(candidate.provider)
            # 解不开（APP_SECRET 换过 / 密文损坏）→ 跳过这条凭证：一条坏凭证
            # 不该让整轮预刷新失败（早先直接冒泡 → 整轮任务挂掉）。
            try:
                data = self._credentials.credential_data(candidate.credential_id)
            except CredentialDecryptError:
                logger.error("预刷新跳过 %s：凭证解密失败（APP_SECRET 是否被更换过？）",
                             candidate.credential_id)
                report.skipped += 1
                continue
            if provider is None or data is None:
                report.skipped += 1
                continue
            outcome = await self._refresh_locked(
                candidate.credential_id, provider, data,
                force=False, revive=False, manual=False)
            if outcome["attempted"]:
                report.attempted += 1
            if outcome["skipped"]:
                report.skipped += 1
            if outcome["ok"] and outcome["refreshed"]:
                report.succeeded += 1
            elif not outcome["ok"] and not outcome["skipped"]:
                report.failed += 1
        return report


def _needs_refresh(provider, data: dict, skew_seconds: int, now: int) -> bool:
    builder = getattr(provider, "credential_from", None)
    if builder is None:
        return False
    credential = builder(data)
    needs = getattr(credential, "needs_refresh", None)
    return bool(needs(skew_seconds, now)) if callable(needs) else False
