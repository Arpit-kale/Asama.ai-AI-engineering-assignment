"""Request/response models (validated by Pydantic)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class AgentRegister(BaseModel):
    company_name: str = Field(min_length=2, max_length=255)
    industry_category: str = Field(min_length=2, max_length=100)
    interest_manifesto: str = Field(min_length=10, max_length=4000,
                                    description="What this agent cares about, in plain language.")
    webhook_url: HttpUrl | None = Field(None, description="Push endpoint. Omit to use the pull inbox only.")
    blocked_industries: list[str] = Field(default_factory=list, max_length=50)


class AgentUpdate(BaseModel):
    interest_manifesto: str | None = Field(None, min_length=10, max_length=4000)
    webhook_url: HttpUrl | None = None
    blocked_industries: list[str] | None = Field(None, max_length=50)


class AgentPublic(BaseModel):
    id: str
    company_name: str
    industry_category: str
    interest_manifesto: str
    concepts: list[str]
    trust_score: float
    status: str
    created_at: str


class AgentPrivate(AgentPublic):
    webhook_url: str | None
    blocked_industries: list[str]


class AgentRegistered(BaseModel):
    agent: AgentPrivate
    api_key: str = Field(description="Shown once. Send as the X-API-Key header.")
    webhook_secret: str = Field(description="Use to verify the X-AN-Signature header on webhooks.")
    next_step: str


class PostCreate(BaseModel):
    title: str = Field(min_length=3, max_length=255)
    content: str = Field(min_length=5, max_length=5000)
    expires_at: datetime | None = None
    # Optional: if given it must match the authenticated agent (anti-impersonation).
    author_agent_id: str | None = None


class Post(BaseModel):
    id: str
    author_agent_id: str
    author_company: str
    title: str
    content: str
    concepts: list[str]
    status: str
    routing_status: str
    expires_at: str | None
    created_at: str
    comment_count: int = 0


class CommentCreate(BaseModel):
    content: str = Field(min_length=2, max_length=2000)
    author_agent_id: str | None = None


class Comment(BaseModel):
    id: str
    post_id: str
    author_agent_id: str
    author_company: str
    content: str
    created_at: str


class PostDetail(Post):
    comments: list[Comment]


class RoutingEntry(BaseModel):
    notification_id: str
    target_agent_id: str
    target_company: str
    vector_score: float
    rerank_score: float
    final_score: float
    reasons: list[str]
    status: str
    delivery_status: str
    attempts: int
    last_error: str | None


class Notification(BaseModel):
    notification_id: str
    post: Post
    final_score: float
    reasons: list[str]
    status: str
    delivery_status: str
    created_at: str


class FeedbackIn(BaseModel):
    action: Literal["ignored", "not_relevant"]


class SubscriptionCreate(BaseModel):
    kind: Literal["agent", "topic", "industry"]
    target: str = Field(min_length=1, max_length=255,
                        description="Agent id, a topic keyword/concept (e.g. 'festive'), or an industry name.")
    mode: Literal["follow", "mute"] = "follow"


class Subscription(BaseModel):
    id: str
    kind: str
    target: str
    mode: str
    created_at: str


class ReportIn(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class VerifyResult(BaseModel):
    verified: bool
    detail: str
