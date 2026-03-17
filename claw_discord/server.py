"""Main server launcher."""

from __future__ import annotations

import uvicorn

from claw_discord.api.app import app
from claw_discord.models import init_db


def create_app(db_path: str | None = None, enable_mcp: bool = True):
    """Initialize the app: create tables, mount MCP."""
    init_db(db_path)

    if enable_mcp:
        from claw_discord.mcp.server import mount_mcp
        mount_mcp(app)

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8003,
    db_path: str | None = None,
    enable_mcp: bool = True,
    reload: bool = False,
):
    """Run the server."""
    create_app(db_path=db_path, enable_mcp=enable_mcp)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
