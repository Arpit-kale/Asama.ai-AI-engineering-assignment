"""Simulated company agents: one FastAPI app hosting a webhook per agent.

Each agent registers with the platform, proves webhook ownership (challenge), verifies the
HMAC signature on every event, decides whether to engage and comments back - or sends
'ignored' feedback, which the platform uses to learn.

Run:  python -m agents_sim            (PORT, PLATFORM_URL, SIM_URL, SIM_CONTROL_KEY)
      then POST http://localhost:9000/control/bootstrap

The /control/* endpoints drive the demo (register agents, publish posts). They require the
X-Control-Key header when SIM_CONTROL_KEY is set; otherwise they only accept loopback clients.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request

from app.security import SIGNATURE_HEADER, TIMESTAMP_HEADER, verify_signature

from .brain import AgentBrain, LLMClient
from .profiles import PROFILES, AgentProfile

log = logging.getLogger("agents_sim")


@dataclass
class AgentState:
    profile: AgentProfile
    brain: AgentBrain
    agent_id: str | None = None
    api_key: str | None = None
    secret: str | None = None
    events: list[dict] = field(default_factory=list)


def create_sim_app(platform_url: str | None = None, self_url: str | None = None,
                   control_key: str | None = None) -> FastAPI:
    platform_url = (platform_url or os.getenv("PLATFORM_URL", "http://127.0.0.1:8000")).rstrip("/")
    self_url = (self_url or os.getenv("SIM_URL", "http://127.0.0.1:9000")).rstrip("/")
    control_key = control_key if control_key is not None else os.getenv("SIM_CONTROL_KEY", "")
    registration_key = os.getenv("AN_REGISTRATION_KEY", "")
    llm = LLMClient.from_env()
    agents = {p.slug: AgentState(p, AgentBrain(p, llm)) for p in PROFILES}
    client = httpx.AsyncClient(base_url=platform_url, timeout=30)
    background: set[asyncio.Task] = set()

    def require_control(request: Request, x_control_key: str | None = Header(None)) -> None:
        if control_key:
            if x_control_key != control_key:
                raise HTTPException(401, "invalid X-Control-Key")
        elif not request.client or request.client.host not in ("127.0.0.1", "::1", "localhost", "testclient"):
            raise HTTPException(403, "set SIM_CONTROL_KEY to use /control remotely")

    app = FastAPI(title="Simulated company agents")
    control = APIRouter(prefix="/control", dependencies=[Depends(require_control)])

    @app.get("/health")
    def health():
        return {"status": "ok", "registered": sum(1 for a in agents.values() if a.api_key)}

    # ------------------------------------------------------------------ control plane (demo only)
    @control.post("/bootstrap")
    async def bootstrap():
        """Register + verify every demo agent with the platform."""
        out = {}
        for slug, st in agents.items():
            if st.api_key:
                out[slug] = {"agent_id": st.agent_id, "status": "already registered"}
                continue
            p = st.profile
            reg_headers = {"X-Registration-Key": registration_key} if registration_key else {}
            resp = await client.post("/api/v1/agents/register", headers=reg_headers, json={
                "company_name": p.company_name, "industry_category": p.industry_category,
                "interest_manifesto": p.interest_manifesto, "blocked_industries": p.blocked_industries,
                "webhook_url": f"{self_url}/agents/{slug}/webhook",
            })
            if resp.status_code == 409:
                out[slug] = {"agent_id": None, "verified": False,
                             "detail": "company already registered on the platform (credentials live elsewhere)"}
                continue
            if resp.status_code != 201:
                raise HTTPException(502, f"register {slug} failed: {resp.text}")
            body = resp.json()
            st.agent_id, st.api_key, st.secret = body["agent"]["id"], body["api_key"], body["webhook_secret"]
            v = await client.post("/api/v1/agents/me/verify", headers={"X-API-Key": st.api_key})
            out[slug] = {"agent_id": st.agent_id, **v.json()}
        return out

    @control.get("/credentials")
    def credentials():
        return {s: {"agent_id": a.agent_id, "api_key": a.api_key} for s, a in agents.items()}

    @control.get("/agents/{slug}/events")
    def events(slug: str):
        return _state(slug).events

    @control.post("/agents/{slug}/post")
    async def publish(slug: str, body: dict):
        """Make an agent publish something it 'found' (internet sourcing is a black box)."""
        st = _state(slug)
        if not st.api_key:
            raise HTTPException(409, "agent not registered; call /control/bootstrap first")
        resp = await client.post("/api/v1/posts", json=body, headers={"X-API-Key": st.api_key})
        return {"status_code": resp.status_code, "body": resp.json()}

    # ------------------------------------------------------------------ webhooks
    def _state(slug: str) -> AgentState:
        if slug not in agents:
            raise HTTPException(404, "unknown agent")
        return agents[slug]

    @app.post("/agents/{slug}/webhook")
    async def webhook(slug: str, request: Request):
        st = _state(slug)
        raw = await request.body()
        if not st.secret or not verify_signature(st.secret, raw, request.headers.get(SIGNATURE_HEADER, ""),
                                                 request.headers.get(TIMESTAMP_HEADER, "")):
            raise HTTPException(401, "bad signature")
        event = await request.json()
        st.events.append(event)

        if event["event"] == "webhook.challenge":
            return {"challenge": event["challenge"]}
        if event["event"] == "post.relevant":
            # Acknowledge fast; think and reply asynchronously.
            task = asyncio.create_task(_handle_post(st, event))
            background.add(task)
            task.add_done_callback(background.discard)
        return {"ok": True}

    def _platform_path(action_url: str) -> str:
        """Only ever call OUR configured platform with our API key, never a host named in a payload."""
        parsed = urlparse(action_url)
        return parsed.path + (f"?{parsed.query}" if parsed.query else "")

    async def _handle_post(st: AgentState, event: dict) -> None:
        post = event["post"]
        headers = {"X-API-Key": st.api_key}
        try:
            comment = await st.brain.decide(post)
            if comment:
                r = await client.post(_platform_path(event["actions"]["comment"]), json={"content": comment}, headers=headers)
                log.info("%s commented on %s -> %s", st.profile.slug, post["id"], r.status_code)
            else:
                await client.post(_platform_path(event["actions"]["feedback"]), json={"action": "ignored"}, headers=headers)
                log.info("%s ignored %s", st.profile.slug, post["id"])
        except Exception:  # noqa: BLE001
            log.exception("%s failed to handle post", st.profile.slug)

    @control.get("/pending")
    def pending():
        return {"pending_tasks": len(background)}

    app.include_router(control)
    return app


app = create_sim_app()
