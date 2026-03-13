"""Gymnasium environment for Calendar tool-use RL training."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import httpx

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from claw_gcal.models import reset_engine
from claw_gcal.seed.generator import seed_database
from claw_gcal.tasks import get_task, list_tasks


def _start_server(host: str, port: int, db_path: str):
    """Start FastAPI server in a background thread."""
    import uvicorn

    from claw_gcal.server import create_app

    app = create_app(db_path=db_path, enable_mcp=False)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


if HAS_GYM:
    class GCalToolEnv(gym.Env):
        """Gymnasium environment for interacting with the Mock Calendar API via tool calls.

        Usage:
            env = GCalToolEnv(task_name="cap-calendar-01", scenario="default")
            obs, info = env.reset()
            obs, reward, terminated, truncated, info = env.step({
                "tool_name": "events_insert",
                "tool_args": {"calendarId": "primary", "summary": "Standup", ...}
            })
        """

        metadata = {"render_modes": ["human"]}

        # API tool definitions
        TOOLS = {
            "calendar_list": ("GET", "/calendar/v3/users/me/calendarList"),
            "calendar_get": ("GET", "/calendar/v3/calendars/{calendarId}"),
            "calendars_insert": ("POST", "/calendar/v3/calendars"),
            "calendars_delete": ("DELETE", "/calendar/v3/calendars/{calendarId}"),
            "events_list": ("GET", "/calendar/v3/calendars/{calendarId}/events"),
            "events_get": ("GET", "/calendar/v3/calendars/{calendarId}/events/{eventId}"),
            "events_insert": ("POST", "/calendar/v3/calendars/{calendarId}/events"),
            "events_patch": ("PATCH", "/calendar/v3/calendars/{calendarId}/events/{eventId}"),
            "events_delete": ("DELETE", "/calendar/v3/calendars/{calendarId}/events/{eventId}"),
        }

        def __init__(
            self,
            task_name: str = "cap-calendar-01",
            scenario: str | None = None,
            host: str = "127.0.0.1",
            port: int = 8100,
            db_path: str = "gym_gcal.db",
            seed: int = 42,
            max_steps: int = 50,
            step_penalty: float = -0.01,
        ):
            super().__init__()

            self.task_name = task_name
            self.host = host
            self.port = port
            self.db_path = db_path
            self.seed_val = seed
            self.max_steps = max_steps
            self.step_penalty = step_penalty
            self.base_url = f"http://{host}:{port}"

            task = get_task(task_name)
            if not task:
                raise ValueError(f"Unknown task: {task_name}. Available: {list_tasks()}")
            self.task = task
            self.scenario = scenario or task.scenario

            # Spaces
            self.action_space = spaces.Dict(
                {
                    "tool_name": spaces.Text(min_length=1, max_length=100),
                    "tool_args": spaces.Text(min_length=0, max_length=10000),
                }
            )
            self.observation_space = spaces.Dict(
                {
                    "goal": spaces.Text(min_length=0, max_length=10000),
                    "api_response": spaces.Text(min_length=0, max_length=100000),
                    "step": spaces.Discrete(max_steps + 1),
                }
            )

            self._server_thread = None
            self._step_count = 0
            self._client = None

        def reset(self, seed=None, options=None):
            """Reset environment: re-seed DB, start server, return initial observation."""
            if seed is not None:
                self.seed_val = seed

            # Reset DB
            reset_engine()
            if os.path.exists(self.db_path):
                os.unlink(self.db_path)

            seed_database(
                scenario=self.scenario,
                seed=self.seed_val,
                db_path=self.db_path,
            )

            # Start server if not running
            if self._server_thread is None or not self._server_thread.is_alive():
                reset_engine()  # Reset so server picks up new DB
                self._server_thread = threading.Thread(
                    target=_start_server,
                    args=(self.host, self.port, self.db_path),
                    daemon=True,
                )
                self._server_thread.start()
                self._wait_for_server()

            self._client = httpx.Client(base_url=self.base_url, timeout=30)
            self._step_count = 0

            # Reset action log
            self._client.post("/_admin/reset")

            obs = {
                "goal": self.task.instruction,
                "api_response": json.dumps({"status": "ready", "task": self.task_name}),
                "step": 0,
            }
            info = {
                "task_name": self.task_name,
                "scenario": self.scenario,
                "tools": list(self.TOOLS.keys()),
            }
            return obs, info

        def step(self, action: dict[str, Any]):
            """Execute a tool call and return observation."""
            self._step_count += 1

            tool_name = action.get("tool_name", "")
            tool_args_raw = action.get("tool_args", "{}")

            if isinstance(tool_args_raw, str):
                try:
                    tool_args = json.loads(tool_args_raw)
                except json.JSONDecodeError:
                    tool_args = {}
            else:
                tool_args = tool_args_raw

            # Execute tool call
            api_response = self._execute_tool(tool_name, tool_args)

            # Check task completion
            reward = self.step_penalty  # Per-step penalty
            terminated = False
            truncated = self._step_count >= self.max_steps

            if truncated or self._should_evaluate(tool_name):
                state = self._client.get("/_admin/state").json()
                diff = self._client.get("/_admin/diff").json()
                log = self._client.get("/_admin/action_log").json().get("entries", [])

                task_reward, done = self.task.evaluate(state, diff, log)
                reward += task_reward
                terminated = done

            obs = {
                "goal": self.task.instruction,
                "api_response": json.dumps(api_response)
                if isinstance(api_response, dict)
                else str(api_response),
                "step": self._step_count,
            }
            info = {"tool_name": tool_name, "step": self._step_count}

            return obs, reward, terminated, truncated, info

        def _execute_tool(self, tool_name: str, args: dict) -> dict:
            """Execute an API tool call."""
            if tool_name not in self.TOOLS:
                return {
                    "error": f"Unknown tool: {tool_name}. Available: {list(self.TOOLS.keys())}"
                }

            method, path_template = self.TOOLS[tool_name]

            # Substitute path params
            path = path_template
            for key in ["calendarId", "eventId"]:
                token = "{" + key + "}"
                if token in path:
                    value = args.pop(key, None)
                    if value is None:
                        return {"error": f"Missing required path param: {key}"}
                    path = path.replace(token, str(value))

            try:
                if method == "GET":
                    resp = self._client.get(path, params=args)
                elif method == "POST":
                    resp = self._client.post(path, json=args)
                elif method == "PATCH":
                    resp = self._client.patch(path, json=args)
                elif method == "DELETE":
                    resp = self._client.delete(path)
                else:
                    return {"error": f"Unsupported method: {method}"}

                if resp.status_code == 200 or resp.status_code == 201:
                    return resp.json()
                if resp.status_code == 204:
                    return {"status": "no_content"}
                return {"error": resp.text, "status_code": resp.status_code}
            except Exception as exc:
                return {"error": str(exc)}

        def _should_evaluate(self, tool_name: str) -> bool:
            """Heuristic: evaluate after write operations."""
            write_tools = {
                "calendars_insert",
                "calendars_delete",
                "events_insert",
                "events_patch",
                "events_delete",
            }
            return tool_name in write_tools

        def _wait_for_server(self, timeout: float = 10.0):
            """Wait for server to be ready."""
            start = time.time()
            while time.time() - start < timeout:
                try:
                    resp = httpx.get(f"{self.base_url}/health", timeout=1)
                    if resp.status_code == 200:
                        return
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
                time.sleep(0.2)
            raise TimeoutError(f"Server did not start within {timeout}s")

        def render(self):
            pass

        def close(self):
            if self._client:
                self._client.close()
else:
    class GCalToolEnv:
        """Stub when gymnasium is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "gymnasium is required for GCalToolEnv. "
                "Install with: pip install gymnasium"
            )
