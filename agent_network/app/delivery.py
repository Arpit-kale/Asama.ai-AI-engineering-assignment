"""Webhook delivery: async worker pool, HMAC signing, retries, per-agent circuit breaker.

Every notification is also persisted in ``routing_logs``, so agents without a webhook (or
whose webhook is down) can still pull it from ``GET /api/v1/inbox``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings
from .db import Database, now_iso
from .security import UnsafeURLError, check_webhook_url, sign_payload

log = logging.getLogger("agent_network.delivery")


@dataclass
class DeliveryJob:
    agent_id: str
    payload: dict[str, Any]
    routing_log_id: str | None = None     # set for post notifications (tracked); None = best effort


@dataclass
class _Breaker:
    failures: int = 0
    opened_at: float | None = None


@dataclass
class DeliveryService:
    db: Database
    settings: Settings
    transport: httpx.AsyncBaseTransport | None = None
    _queue: asyncio.Queue = field(init=False)
    _client: httpx.AsyncClient | None = field(init=False, default=None)
    _workers: list[asyncio.Task] = field(init=False, default_factory=list)
    _breakers: dict[str, _Breaker] = field(init=False, default_factory=dict)

    # ---- lifecycle -----------------------------------------------------------
    async def start(self) -> None:
        self._queue = asyncio.Queue()
        self._client = httpx.AsyncClient(timeout=self.settings.webhook_timeout, transport=self.transport)
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self.settings.delivery_workers)]

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        if self._client:
            await self._client.aclose()

    def enqueue(self, job: DeliveryJob) -> None:
        self._queue.put_nowait(job)

    async def join(self) -> None:
        """Wait until every queued job has been processed (used by tests and the demo)."""
        await self._queue.join()

    # ---- breaker -------------------------------------------------------------
    def _breaker_open(self, agent_id: str) -> bool:
        b = self._breakers.get(agent_id)
        if not b or b.opened_at is None:
            return False
        if time.monotonic() - b.opened_at >= self.settings.breaker_cooldown:
            b.opened_at = None          # half-open: allow one trial request
            return False
        return True

    def _record(self, agent_id: str, ok: bool) -> None:
        b = self._breakers.setdefault(agent_id, _Breaker())
        if ok:
            b.failures, b.opened_at = 0, None
        else:
            b.failures += 1
            if b.failures >= self.settings.breaker_threshold:
                b.opened_at = time.monotonic()

    # ---- delivery ------------------------------------------------------------
    def _update_log(self, log_id: str | None, status: str, attempts: int, error: str | None) -> None:
        if log_id:
            self.db.execute(
                "UPDATE routing_logs SET delivery_status = ?, attempts = ?, last_error = ?, updated_at = ? WHERE id = ?",
                (status, attempts, error, now_iso(), log_id),
            )

    async def _guard(self, url: str) -> None:
        """Re-check the target right before sending (defeats DNS re-binding after registration)."""
        await asyncio.to_thread(check_webhook_url, url, self.settings.allow_private_webhooks,
                                self.settings.require_https_webhooks)

    async def _post(self, url: str, secret: str, payload: dict) -> httpx.Response:
        assert self._client is not None
        await self._guard(url)
        body = json.dumps(payload, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json", **sign_payload(secret, body)}
        return await self._client.post(url, content=body, headers=headers)

    async def deliver(self, job: DeliveryJob) -> str:
        agent = self.db.one("SELECT webhook_url, webhook_secret FROM agents WHERE id = ?", (job.agent_id,))
        if not agent:
            return "FAILED"
        if not agent["webhook_url"]:
            self._update_log(job.routing_log_id, "INBOX_ONLY", 0, None)
            return "INBOX_ONLY"
        if self._breaker_open(job.agent_id):
            self._update_log(job.routing_log_id, "INBOX_ONLY", 0, "circuit open")
            return "INBOX_ONLY"

        error = None
        for attempt in range(1, self.settings.max_delivery_attempts + 1):
            try:
                resp = await self._post(agent["webhook_url"], agent["webhook_secret"], job.payload)
                if resp.is_success:
                    self._record(job.agent_id, True)
                    self._update_log(job.routing_log_id, "DELIVERED", attempt, None)
                    return "DELIVERED"
                error = f"HTTP {resp.status_code}"
                if 400 <= resp.status_code < 500 and resp.status_code not in (408, 429):
                    break                              # client error: retrying will not help
            except UnsafeURLError as exc:
                error = f"blocked: {exc}"
                break
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
            if attempt < self.settings.max_delivery_attempts:
                await asyncio.sleep(self.settings.retry_base_delay * 2 ** (attempt - 1))

        self._record(job.agent_id, False)
        self._update_log(job.routing_log_id, "FAILED", attempt, error)
        log.warning("delivery to %s failed: %s", job.agent_id, error)
        return "FAILED"

    async def _worker(self, n: int) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self.deliver(job)
            except Exception:  # noqa: BLE001 - a worker must never die
                log.exception("delivery worker %d crashed on job", n)
            finally:
                self._queue.task_done()

    # ---- webhook ownership verification --------------------------------------
    async def challenge(self, webhook_url: str, secret: str) -> tuple[bool, str]:
        """Proves the registrant controls the webhook (like Slack's URL verification)."""
        token = secrets.token_urlsafe(16)
        try:
            resp = await self._post(webhook_url, secret, {"event": "webhook.challenge", "challenge": token})
        except UnsafeURLError as exc:
            return False, f"webhook blocked: {exc}"
        except httpx.HTTPError as exc:
            return False, f"webhook unreachable: {exc}"
        if not resp.is_success:
            return False, f"webhook returned HTTP {resp.status_code}"
        try:
            ok = resp.json().get("challenge") == token
        except ValueError:
            ok = False
        return ok, "verified" if ok else "challenge mismatch"
