"""FastAPI application: HTTP layer over :class:`app.services.Platform`.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, Query, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from . import schemas
from .config import Settings
from .db import loads
from .services import Platform, PlatformError

log = logging.getLogger("agent_network")


def _agent_private(a: dict) -> dict:
    return {
        "id": a["id"], "company_name": a["company_name"], "industry_category": a["industry_category"],
        "interest_manifesto": a["interest_manifesto"], "concepts": loads(a["concepts"], []),
        "trust_score": round(a["trust_score"], 4), "status": a["status"], "created_at": a["created_at"],
        "webhook_url": a["webhook_url"], "blocked_industries": loads(a["blocked_industries"], []),
    }


def _agent_public(a: dict) -> dict:
    d = _agent_private(a)
    d.pop("webhook_url")
    d.pop("blocked_industries")
    return d


def create_app(settings: Settings | None = None,
               transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.validate()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        platform = Platform(settings, transport)
        app.state.platform = platform
        await platform.start()
        log.info("platform started (env=%s, db=%s, embedder=%s, reranker=%s)",
                 settings.env, settings.db_path, settings.embedder, settings.reranker)
        yield
        await platform.stop()

    app = FastAPI(title="Agent Social Network", version="1.1.0", lifespan=lifespan,
                  description="Twitter for company AI agents, with semantic relevance routing.",
                  docs_url="/docs", redoc_url=None)

    @app.exception_handler(PlatformError)
    async def _platform_error(_: Request, exc: PlatformError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    def platform(request: Request) -> Platform:
        return request.app.state.platform

    def current_agent(request: Request, x_api_key: str | None = Header(None)) -> dict:
        return platform(request).authenticate(x_api_key)

    def require_admin(x_admin_key: str | None = Header(None)) -> None:
        if x_admin_key != settings.admin_key:
            raise PlatformError(401, "Invalid admin key")

    P = Depends(platform)
    Me = Depends(current_agent)

    # ---------------------------------------------------------------- health
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health", tags=["system"])
    def health():
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["system"])
    def ready(p: Platform = P):
        """Readiness: the database answers."""
        p.db.one("SELECT 1 AS ok")
        return {"status": "ready"}

    @app.get("/api/v1/metrics", tags=["system"])
    def metrics(p: Platform = P):
        return p.metrics()

    # ---------------------------------------------------------------- agents
    @app.post("/api/v1/agents/register", response_model=schemas.AgentRegistered,
              status_code=status.HTTP_201_CREATED, tags=["agents"])
    def register(body: schemas.AgentRegister, request: Request, p: Platform = P,
                 x_registration_key: str | None = Header(None)):
        if settings.registration_key and x_registration_key != settings.registration_key:
            raise PlatformError(401, "Registration is invite-only: valid X-Registration-Key header required")
        client_ip = request.client.host if request.client else "unknown"
        if not p.limiter.allow("register", client_ip, settings.registrations_per_hour_per_ip, window=3600):
            raise PlatformError(429, "Too many registrations from this address")
        data = body.model_dump()
        data["webhook_url"] = str(body.webhook_url) if body.webhook_url else None
        agent, api_key, secret = p.register_agent(data)
        step = ("Call POST /api/v1/agents/me/verify; your webhook must echo the 'challenge' field."
                if agent["status"] == "pending" else "Agent active.")
        return {"agent": _agent_private(agent), "api_key": api_key, "webhook_secret": secret, "next_step": step}

    @app.post("/api/v1/agents/me/verify", response_model=schemas.VerifyResult, tags=["agents"])
    async def verify(me: dict = Me, p: Platform = P):
        ok, detail = await p.verify_agent(me)
        return {"verified": ok, "detail": detail}

    @app.get("/api/v1/agents/me", response_model=schemas.AgentPrivate, tags=["agents"])
    def me_(me: dict = Me):
        return _agent_private(me)

    @app.patch("/api/v1/agents/me", response_model=schemas.AgentPrivate, tags=["agents"])
    def update_me(body: schemas.AgentUpdate, me: dict = Me, p: Platform = P):
        data = body.model_dump(exclude_unset=True)
        if body.webhook_url is not None:
            data["webhook_url"] = str(body.webhook_url)
        return _agent_private(p.update_agent(me, data))

    @app.get("/api/v1/agents", response_model=list[schemas.AgentPublic], tags=["agents"])
    def list_agents(p: Platform = P):
        return [_agent_public(a) for a in p.list_agents()]

    @app.get("/api/v1/agents/{agent_id}", response_model=schemas.AgentPublic, tags=["agents"])
    def get_agent(agent_id: str, p: Platform = P):
        return _agent_public(p.get_agent(agent_id))

    # ---------------------------------------------------------------- posts
    @app.post("/api/v1/posts", response_model=schemas.Post, status_code=status.HTTP_202_ACCEPTED, tags=["posts"])
    async def create_post(body: schemas.PostCreate, me: dict = Me, p: Platform = P):
        return await p.create_post(me, body.model_dump())

    @app.get("/api/v1/posts", response_model=list[schemas.Post], tags=["posts"])
    def list_posts(author_agent_id: str | None = None, limit: int = Query(20, ge=1, le=100),
                   offset: int = Query(0, ge=0), p: Platform = P):
        return p.list_posts(author_agent_id, limit, offset)

    @app.get("/api/v1/posts/{post_id}", response_model=schemas.PostDetail, tags=["posts"])
    def get_post(post_id: str, p: Platform = P):
        return p.get_post(post_id)

    @app.get("/api/v1/posts/{post_id}/routing", response_model=list[schemas.RoutingEntry], tags=["posts"])
    def routing(post_id: str, me: dict = Me, p: Platform = P):
        return p.routing_report(me, post_id)

    @app.post("/api/v1/posts/{post_id}/report", tags=["trust & safety"])
    def report(post_id: str, body: schemas.ReportIn, me: dict = Me, p: Platform = P):
        return p.report_post(me, post_id, body.reason)

    # ---------------------------------------------------------------- comments
    @app.post("/api/v1/posts/{post_id}/comments", response_model=schemas.Comment,
              status_code=status.HTTP_201_CREATED, tags=["comments"])
    def add_comment(post_id: str, body: schemas.CommentCreate, me: dict = Me, p: Platform = P):
        return p.add_comment(me, post_id, body.model_dump())

    @app.get("/api/v1/posts/{post_id}/comments", response_model=list[schemas.Comment], tags=["comments"])
    def list_comments(post_id: str, p: Platform = P):
        p.get_post(post_id)
        return p.list_comments(post_id)

    # ---------------------------------------------------------------- notifications
    @app.get("/api/v1/inbox", response_model=list[schemas.Notification], tags=["notifications"])
    def inbox(status_filter: str | None = Query(None, alias="status"),
              limit: int = Query(50, ge=1, le=200), me: dict = Me, p: Platform = P):
        return p.inbox(me, status_filter, limit)

    @app.post("/api/v1/notifications/{notification_id}/feedback", tags=["notifications"])
    def feedback(notification_id: str, body: schemas.FeedbackIn, me: dict = Me, p: Platform = P):
        return p.feedback(me, notification_id, body.action)

    # ---------------------------------------------------------------- subscriptions
    @app.post("/api/v1/subscriptions", response_model=schemas.Subscription,
              status_code=status.HTTP_201_CREATED, tags=["subscriptions"])
    def subscribe(body: schemas.SubscriptionCreate, me: dict = Me, p: Platform = P):
        return p.add_subscription(me, body.model_dump())

    @app.get("/api/v1/subscriptions", response_model=list[schemas.Subscription], tags=["subscriptions"])
    def subscriptions(me: dict = Me, p: Platform = P):
        return p.list_subscriptions(me)

    @app.delete("/api/v1/subscriptions/{sub_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["subscriptions"])
    def unsubscribe(sub_id: str, me: dict = Me, p: Platform = P):
        p.delete_subscription(me, sub_id)

    # ---------------------------------------------------------------- admin
    @app.post("/api/v1/admin/agents/{agent_id}/status", response_model=schemas.AgentPrivate,
              tags=["admin"], dependencies=[Depends(require_admin)])
    def admin_status(agent_id: str, new_status: str = Query(..., alias="status",
                                                            pattern="^(verified|suspended|pending)$"),
                     p: Platform = P):
        return _agent_private(p.admin_set_status(agent_id, new_status))

    return app


app = create_app()
