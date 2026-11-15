"""python -m agents_sim  - serve the simulated company agents on $PORT (default 9000)."""
import logging
import os

import uvicorn

logging.basicConfig(level=os.getenv("SIM_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    uvicorn.run("agents_sim.server:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "9000")),
                proxy_headers=True, forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "*"))
