"""Claude as the policy, via the Claude Agent SDK (authenticates with a Claude Pro/Max login).

Each episode gets a fresh environment and a fresh in-process MCP server whose
tools are closures over that environment. All Claude Code built-in tools
(Bash, Read, Write, ...) are disabled, and user settings, CLAUDE.md files and
other MCP servers are ignored. The agent's whole world is the telescope tools.
"""

from __future__ import annotations

import dataclasses
import tempfile
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage, SystemMessage,
                              TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock, UserMessage,
                              create_sdk_mcp_server, query, tool)

from .env import TelescopeEnv

SERVER = "telescope"


def make_tools(env: TelescopeEnv):
    tools = []
    for spec in env.tool_specs():
        async def handler(args: dict[str, Any], _name: str = spec.name) -> dict[str, Any]:
            res = env.call(_name, **args)
            return {"content": [{"type": "text", "text": res.text}], "is_error": res.error}
        tools.append(tool(spec.name, spec.description, spec.schema)(handler))
    return tools


def build_server(env: TelescopeEnv):
    server = create_sdk_mcp_server(SERVER, tools=make_tools(env))
    return server, [f"mcp__{SERVER}__{s.name}" for s in env.tool_specs()]


def _block(b) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name.removeprefix(f"mcp__{SERVER}__"), "input": b.input}
    if isinstance(b, ToolResultBlock):
        return {"type": "tool_result", "tool_use_id": b.tool_use_id, "is_error": b.is_error, "content": b.content}
    if isinstance(b, ThinkingBlock):
        return {"type": "thinking"}               # reasoning text is deliberately not stored
    return {"type": type(b).__name__}


def serialize(msg) -> dict[str, Any] | None:
    if isinstance(msg, AssistantMessage):
        return {"role": "assistant", "model": msg.model, "content": [_block(b) for b in msg.content]}
    if isinstance(msg, UserMessage):
        content = msg.content if isinstance(msg.content, str) else [_block(b) for b in msg.content]
        return {"role": "user", "content": content}
    if isinstance(msg, ResultMessage):
        d = dataclasses.asdict(msg)
        return {"role": "result", **{k: d.get(k) for k in (
            "subtype", "is_error", "num_turns", "duration_ms", "total_cost_usd", "usage",
            "stop_reason", "result", "permission_denials", "terminal_reason")}}
    if isinstance(msg, SystemMessage):
        return {"role": "system", "subtype": msg.subtype}
    return None


async def run_episode(env: TelescopeEnv, task: dict[str, Any], *, model: str, system_prompt: str,
                      max_turns: int = 40, effort: str | None = None) -> tuple[list[dict], dict[str, Any]]:
    """Run one episode. Returns (transcript, agent_info). Call env.finalize() afterwards."""
    briefing = env.reset(task)
    server, allowed = build_server(env)
    transcript: list[dict] = [{"role": "user", "content": briefing}]
    info: dict[str, Any] = {"sdk_error": None}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as empty_dir:     # nothing for the agent to discover on disk
        opts = ClaudeAgentOptions(
            system_prompt=system_prompt,
            mcp_servers={SERVER: server}, strict_mcp_config=True,
            tools=[], allowed_tools=allowed,
            setting_sources=[], cwd=empty_dir,
            model=model, max_turns=max_turns, effort=effort,
        )
        try:
            async for msg in query(prompt=briefing, options=opts):
                rec = serialize(msg)
                if rec:
                    transcript.append(rec)
                if isinstance(msg, ResultMessage):
                    info.update(num_turns=msg.num_turns, usage=msg.usage, result_subtype=msg.subtype,
                                total_cost_usd=msg.total_cost_usd, final_text=msg.result)
        except Exception as exc:                      # auth, rate limit, max_turns, CLI crash...
            info["sdk_error"] = f"{type(exc).__name__}: {exc}"[:500]
    info["wall_s"] = round(time.time() - t0, 1)
    return transcript, info


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text()
