"""Production hardening: config validation, SSRF guard, registration controls, probes."""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from agents_sim.server import create_sim_app
from app.config import Settings
from app.main import create_app
from app.security import UnsafeURLError, check_webhook_url

AGENT = {"company_name": "PhonePe", "industry_category": "Payments",
         "interest_manifesto": "UPI payments, merchant cashback and festive promotions."}


def make_client(tmp_path, **overrides) -> TestClient:
    settings = Settings(db_path=str(tmp_path / "sub" / "dir" / "t.db"), sync_routing=True, admin_key="a" * 20,
                        **overrides)
    return TestClient(create_app(settings, httpx.MockTransport(lambda r: httpx.Response(200, json={}))))


def test_production_requires_real_admin_key(tmp_path):
    with pytest.raises(RuntimeError, match="AN_ADMIN_KEY"):
        Settings(env="production", db_path=str(tmp_path / "x.db")).validate()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Settings(embedder="openai", openai_api_key=None).validate()
    Settings(env="production", admin_key="x" * 32).validate()


def test_production_defaults(monkeypatch):
    monkeypatch.setenv("AN_ENV", "production")
    s = Settings()
    assert s.is_production and not s.allow_private_webhooks and s.require_https_webhooks
    monkeypatch.setenv("AN_ENV", "development")
    assert Settings().allow_private_webhooks


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:9000/hook", "http://localhost/hook", "http://169.254.169.254/latest/meta-data",
    "http://10.0.0.5/x", "http://192.168.1.10/x", "http://[::1]/x", "http://0.0.0.0/x",
    "file:///etc/passwd", "https://user:pw@8.8.8.8/x",
])
def test_ssrf_guard_blocks_internal_targets(url):
    with pytest.raises(UnsafeURLError):
        check_webhook_url(url, allow_private=False, require_https=False)


def test_ssrf_guard_allows_public_and_enforces_https():
    check_webhook_url("https://8.8.8.8/hook", allow_private=False, require_https=True)
    with pytest.raises(UnsafeURLError, match="https"):
        check_webhook_url("http://8.8.8.8/hook", allow_private=False, require_https=True)
    check_webhook_url("http://agents:9000/hook", allow_private=True, require_https=False)   # docker network


def test_registration_rejects_private_webhook_when_locked_down(tmp_path):
    with make_client(tmp_path, allow_private_webhooks=False) as c:
        r = c.post("/api/v1/agents/register", json={**AGENT, "webhook_url": "http://127.0.0.1:9000/hook"})
        assert r.status_code == 422 and "non-public" in r.json()["detail"]


def test_invite_only_registration_and_ip_limit(tmp_path):
    with make_client(tmp_path, registration_key="invite-123", registrations_per_hour_per_ip=1) as c:
        assert c.post("/api/v1/agents/register", json=AGENT).status_code == 401
        ok = c.post("/api/v1/agents/register", json=AGENT, headers={"X-Registration-Key": "invite-123"})
        assert ok.status_code == 201
        again = c.post("/api/v1/agents/register", json={**AGENT, "company_name": "Paytm"},
                       headers={"X-Registration-Key": "invite-123"})
        assert again.status_code == 429


def test_probes_and_db_directory_created(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/ready").json() == {"status": "ready"}
        assert c.get("/", follow_redirects=False).headers["location"] == "/docs"
    assert (tmp_path / "sub" / "dir" / "t.db").exists()


def test_absolute_action_urls(tmp_path):
    hooks = []

    def handler(req: httpx.Request):
        import json
        body = json.loads(req.content)
        hooks.append(body)
        return httpx.Response(200, json={"challenge": body.get("challenge")})

    settings = Settings(db_path=str(tmp_path / "t.db"), sync_routing=True, public_base_url="https://api.example.com")
    with TestClient(create_app(settings, httpx.MockTransport(handler))) as c:
        keys = []
        for name, manifesto in [("Swiggy", "food delivery festive offers"), ("PhonePe", AGENT["interest_manifesto"])]:
            r = c.post("/api/v1/agents/register", json={"company_name": name, "industry_category": "Consumer",
                                                         "interest_manifesto": manifesto,
                                                         "webhook_url": f"http://hooks/{name}"}).json()
            keys.append({"X-API-Key": r["api_key"]})
            c.post("/api/v1/agents/me/verify", headers=keys[-1])
        c.post("/api/v1/posts", headers=keys[0], json={"title": "Diwali offer",
                                                         "content": "Flat 5% off food delivery during Diwali"})
        c.portal.call(c.app.state.platform.drain)
    notif = next(h for h in hooks if h["event"] == "post.relevant")
    assert notif["actions"]["comment"].startswith("https://api.example.com/api/v1/posts/")


def test_sim_control_requires_key():
    with TestClient(create_sim_app("http://platform", "http://sim", control_key="secret")) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/control/credentials").status_code == 401
        assert c.get("/control/credentials", headers={"X-Control-Key": "secret"}).status_code == 200
