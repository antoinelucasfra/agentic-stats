"""Agent panel: the same tools, handed to a chat model."""

from web.panels.agent.server import agent_server
from web.panels.agent.ui import agent_ui

__all__ = ["agent_server", "agent_ui"]
