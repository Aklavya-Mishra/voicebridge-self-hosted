import logging
import os

import uvicorn
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()  # PORT can come from .env
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Single worker: the session store is in-memory.
    uvicorn.run("app.main:create_app", factory=True, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
