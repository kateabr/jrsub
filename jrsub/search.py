from dataclasses import dataclass
from enum import Enum
from typing import List


class SearchMode(Enum):
    consecutive = 0
    deep_only = 1
    shallow_only = 2


@dataclass
class SearchResult:
    reading: List[str]
    lexeme: List[str]
    translation: List[str]
