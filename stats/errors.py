"""The errors the transports share.

`ToolError` lived in `stats_tools` until the session layer needed it too. It
lives here so the store, the role inference and the tools can all raise the same
class without importing each other, and `stats_tools` re-exports it so existing
imports keep working.

The message is the interface: it is written for a model to act on, naming the
columns that do exist or the call that fixes the problem.
"""

from __future__ import annotations

__all__ = ["ToolError"]


class ToolError(ValueError):
    """A user- or agent-facing error with an actionable message."""
