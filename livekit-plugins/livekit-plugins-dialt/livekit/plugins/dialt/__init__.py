"""Dialt realtime plugin for LiveKit Agents."""

from livekit.agents import Plugin

from . import realtime, utils
from .log import logger
from .models import ToolOptions
from .realtime import RealtimeModel, RealtimeSession
from .version import __version__

__all__ = [
    "ToolOptions",
    "realtime",
    "RealtimeModel",
    "RealtimeSession",
    "utils",
    "__version__",
]


class DialtPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(DialtPlugin())

__pdoc__ = {name: False for name in dir() if name not in __all__}
