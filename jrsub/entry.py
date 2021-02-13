from dataclasses import dataclass
from typing import List

from jrsub import WarodaiReference, YarxiReference


@dataclass
class Entry:
    reading: List[str]
    lexeme: List[str]
    translation: List[str]
    eid: str


@dataclass
class WarodaiEntry(Entry):
    translation: {str: List[str]}
    references: {str: List[WarodaiReference]}
    lexeme_id: str = ''

    def __eq__(self, other):
        return self.eid == other.eid

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return int(self.eid.replace('-', ''))


@dataclass
class YarxiEntry(Entry):
    references: List[YarxiReference]
    kanji: List[str]

    def __eq__(self, other):
        return self.eid == other.eid

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return int(self.eid)
