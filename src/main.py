import uvicorn
from fastapi import FastAPI

from src.api import router
from src.config import config

app = FastAPI()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


app.include_router(router, prefix=f"/v{config.api_version}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
    )
