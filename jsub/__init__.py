__author__ = """Ekaterina Biryukova"""
__email__ = "kateabr@yandex.ru"
__version__ = "0.1.0"

from . import utils

from .reference import WarodaiReference, YarxiReference, Reference
from .search import SearchMode, SearchResult
from .entry import WarodaiEntry, YarxiEntry, Entry

from .warodai import WarodaiLoader, WarodaiEid, WarodaiDictionary
from .yarxi import YarxiLoader, YarxiDictionary

__all__ = [
    "WarodaiEntry", "YarxiEntry", "Entry",
    "WarodaiReference", "YarxiReference", "Reference",
    "SearchMode", "SearchResult",
    "WarodaiLoader", "WarodaiEid", "WarodaiDictionary",
    "YarxiLoader", "YarxiDictionary"
]
