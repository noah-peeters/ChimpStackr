"""
Global variables shared between all scripts.
Consts can be set by other scripts.
Consts (obviously) shouldn't be changed after they have been set once.

NOTE: globalVars is legacy. New code should prefer passing references
through constructors or Qt parent-child relationships. The config module
(src.config) provides structured configuration without globals.
"""

from src.config import SUPPORTED_IMAGE_READ_FORMATS, SUPPORTED_RAW_FORMATS


def init():
    global globalVars
    globalVars = {}
    globalVars["SupportedImageReadFormats"] = SUPPORTED_IMAGE_READ_FORMATS
    globalVars["SupportedRAWFormats"] = SUPPORTED_RAW_FORMATS


def get(key, default=None):
    """Safer accessor for globalVars."""
    return globalVars.get(key, default)
