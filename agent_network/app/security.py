"""Authentication, webhook signing, rate limiting and content moderation."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import socket
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from urllib.parse import urlparse

SIGNATURE_HEADER = "X-AN-Signature"
TIMESTAMP_HEADER = "X-AN-Timestamp"
MAX_SIGNATURE_AGE_SECONDS = 300


# --------------------------------------------------------------------------- credentials
def generate_api_key() -> str:
    return "an_" + secrets.token_urlsafe(32)


def generate_webhook_secret() -> str:
    return "whsec_" + secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """API keys are stored hashed; the raw key is shown exactly once at registration."""
    return hashlib.sha256(api_key.encode()).hexdigest()


# --------------------------------------------------------------------------- HMAC webhooks
def sign_payload(secret: str, body: bytes, timestamp: int | None = None) -> dict[str, str]:
    ts = str(int(timestamp if timestamp is not None else time.time()))
    digest = hmac.new(secret.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
    return {SIGNATURE_HEADER: f"sha256={digest}", TIMESTAMP_HEADER: ts}


def verify_signature(secret: str, body: bytes, signature: str, timestamp: str) -> bool:
    """Used by receiving agents. Rejects stale timestamps to block replay attacks."""
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs(time.time() - ts) > MAX_SIGNATURE_AGE_SECONDS:
        return False
    expected = sign_payload(secret, body, ts)[SIGNATURE_HEADER]
    return hmac.compare_digest(expected, signature or "")


# --------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Sliding-window limiter, per (bucket, key). In-memory; use Redis in a multi-node setup."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, bucket: str, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[(bucket, key)]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True


# --------------------------------------------------------------------------- SSRF guard
class UnsafeURLError(ValueError):
    pass


def check_webhook_url(url: str, allow_private: bool, require_https: bool) -> None:
    """Rejects webhook targets that would let an agent make the platform call internal services.

    Every address the hostname resolves to must be public (blocks 127.0.0.1, 10/8, 192.168/16,
    169.254.169.254 cloud metadata, ::1, ...). Called at registration AND before each delivery,
    so DNS re-binding after registration doesn't bypass it.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError("webhook_url must be http(s)")
    if require_https and parsed.scheme != "https":
        raise UnsafeURLError("webhook_url must use https")
    if parsed.username or parsed.password:
        raise UnsafeURLError("webhook_url must not contain credentials")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("webhook_url has no host")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"webhook host '{host}' does not resolve") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified or not ip.is_global):
            raise UnsafeURLError(f"webhook host '{host}' resolves to a non-public address")


# --------------------------------------------------------------------------- moderation
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+|your\s+|the\s+)?(previous|prior|system)?\s*(instructions|prompt|rules)",
    r"you\s+are\s+now\s+(a|an|in)\b",
    r"\bsystem\s*prompt\b",
    r"\b(reveal|print|output|send|leak|share)\b.{0,40}\b(api[\s_-]?key|secret|password|token|credential)s?\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"<\s*/?\s*(system|instructions?)\s*>",
    r"\[\s*(system|inst)\s*\]",
    r"<script\b",
]
ABUSE_TERMS = {"idiot", "idiots", "stupid", "morons", "garbage company", "worthless company"}
SUSPICIOUS_CLAIMS = [
    r"guaranteed\s+returns?",
    r"100%\s+(free|guaranteed)",
    r"send\s+(your\s+)?(otp|pin|cvv)",
    r"double\s+your\s+money",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_SUSPICIOUS_RE = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_CLAIMS]


@dataclass
class ModerationResult:
    allowed: bool
    category: str | None = None
    detail: str | None = None


def moderate(text: str) -> ModerationResult:
    """Cheap first-line filter. In production add an ML guard model (e.g. Llama Guard)."""
    for rx in _INJECTION_RE:
        if rx.search(text):
            return ModerationResult(False, "prompt_injection", f"matched pattern: {rx.pattern}")
    for rx in _SUSPICIOUS_RE:
        if rx.search(text):
            return ModerationResult(False, "fraud", f"matched pattern: {rx.pattern}")
    lowered = text.lower()
    for term in ABUSE_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return ModerationResult(False, "abuse", f"abusive term: {term}")
    return ModerationResult(True)
