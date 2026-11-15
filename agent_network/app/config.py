"""Runtime configuration, loaded from environment variables (prefix ``AN_``).

``AN_ENV=production`` switches on safe defaults: HTTPS-only public webhooks (SSRF guard) and a
mandatory, non-default admin key. ``Settings.validate()`` fails fast at startup on bad config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_ADMIN_KEY = "change-me-admin-key"


def _env(name: str, default: str) -> str:
    return os.getenv(f"AN_{name}", default)


def _flag(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def _is_prod() -> bool:
    return _env("ENV", "development").lower() == "production"


@dataclass
class Settings:
    env: str = field(default_factory=lambda: _env("ENV", "development").lower())
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())
    # Absolute URL clients use to reach the API (makes webhook action links absolute).
    public_base_url: str = field(default_factory=lambda: _env("PUBLIC_BASE_URL", "").rstrip("/"))

    # Storage
    db_path: str = field(default_factory=lambda: _env("DB_PATH", "agent_network.db"))

    # Routing
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "50")))
    threshold: float = field(default_factory=lambda: float(_env("THRESHOLD", "0.45")))
    max_notified: int = field(default_factory=lambda: int(_env("MAX_NOTIFIED", "5")))
    # Route inline inside the POST request (handy for tests) instead of via the event queue.
    sync_routing: bool = field(default_factory=lambda: _flag("SYNC_ROUTING", False))

    # Providers: "local" works offline; "openai" uses OPENAI_API_KEY.
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "local"))
    reranker: str = field(default_factory=lambda: _env("RERANKER", "heuristic"))  # or "llm"
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small"))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "gpt-4o-mini"))

    # Delivery
    delivery_workers: int = field(default_factory=lambda: int(_env("DELIVERY_WORKERS", "4")))
    max_delivery_attempts: int = field(default_factory=lambda: int(_env("MAX_DELIVERY_ATTEMPTS", "4")))
    retry_base_delay: float = field(default_factory=lambda: float(_env("RETRY_BASE_DELAY", "0.5")))
    webhook_timeout: float = field(default_factory=lambda: float(_env("WEBHOOK_TIMEOUT", "5")))
    breaker_threshold: int = 5          # consecutive failures before the circuit opens
    breaker_cooldown: float = 60.0      # seconds the circuit stays open

    # Webhook SSRF guard: block private/loopback/link-local/metadata targets in production.
    allow_private_webhooks: bool = field(default_factory=lambda: _flag("ALLOW_PRIVATE_WEBHOOKS", not _is_prod()))
    require_https_webhooks: bool = field(default_factory=lambda: _flag("REQUIRE_HTTPS_WEBHOOKS", _is_prod()))

    # Trust & safety
    require_verification: bool = field(default_factory=lambda: _flag("REQUIRE_VERIFICATION", True))
    admin_key: str = field(default_factory=lambda: _env("ADMIN_KEY", DEFAULT_ADMIN_KEY))
    # If set, registration needs the X-Registration-Key header (invite-only network).
    registration_key: str = field(default_factory=lambda: _env("REGISTRATION_KEY", ""))
    registrations_per_hour_per_ip: int = field(default_factory=lambda: int(_env("REGISTRATIONS_PER_HOUR", "20")))
    posts_per_minute: int = field(default_factory=lambda: int(_env("POSTS_PER_MINUTE", "10")))
    comments_per_minute: int = field(default_factory=lambda: int(_env("COMMENTS_PER_MINUTE", "30")))
    min_trust_to_post: float = 0.3
    reports_to_flag: int = 3

    # Feedback learning
    learn_rate_engaged: float = 0.10
    learn_rate_ignored: float = 0.05
    max_delta_norm: float = 0.5

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def validate(self) -> None:
        errors = []
        if self.is_production and (self.admin_key == DEFAULT_ADMIN_KEY or len(self.admin_key) < 16):
            errors.append("AN_ADMIN_KEY must be set to a random value of at least 16 characters in production")
        if self.embedder not in ("local", "openai"):
            errors.append("AN_EMBEDDER must be 'local' or 'openai'")
        if self.reranker not in ("heuristic", "llm"):
            errors.append("AN_RERANKER must be 'heuristic' or 'llm'")
        if (self.embedder == "openai" or self.reranker == "llm") and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for AN_EMBEDDER=openai / AN_RERANKER=llm")
        if not 0 < self.threshold <= 1:
            errors.append("AN_THRESHOLD must be in (0, 1]")
        if errors:
            raise RuntimeError("Invalid configuration:\n  - " + "\n  - ".join(errors))
