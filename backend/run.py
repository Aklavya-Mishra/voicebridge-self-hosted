import logging
import os

import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Single worker: the session store is in-memory.
    uvicorn.run("app.main:create_app", factory=True, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
