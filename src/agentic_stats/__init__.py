"""agentic-stats: statistical tools over an R&D DOE dataset, for agents and browsers."""

from importlib.metadata import PackageNotFoundError, version

# One place a version is written: pyproject.toml, which release-please bumps.
# Everything else reads it back from the installed metadata.
try:
    __version__ = version("agentic-stats")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0"

__all__: list[str] = []
