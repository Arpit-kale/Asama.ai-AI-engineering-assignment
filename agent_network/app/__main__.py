"""Production entrypoint:  python -m app

Runs a single uvicorn worker on $PORT (default 8000). One worker is deliberate: the routing
queue, vector index and rate limiter are in-process. Scale out by moving those to
Redis/pgvector (see DEPLOYMENT.md), not by adding workers to this process.
"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        workers=1,
        proxy_headers=True,                                   # real client IP behind a load balancer
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "*"),
        log_level=os.getenv("AN_LOG_LEVEL", "info").lower(),
        timeout_graceful_shutdown=20,
    )


if __name__ == "__main__":
    main()
