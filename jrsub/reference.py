from dataclasses import dataclass, field
from typing import List


@dataclass
class Reference:
    eid: str
    mode: str = ''
    lexeme: List[str] = field(default_factory=list)
    usable: bool = False


@dataclass
class WarodaiReference(Reference):
    meaning_number: List[str] = field(default_factory=list)
    reading: List[str] = field(default_factory=list)
    prefix: str = ''
    body: str = ''

    def __eq__(self, other):
        return self.eid == other.eid and self.mode == other.mode and self.lexeme == other.lexeme and self.reading == other.reading

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return int(self.eid.replace('-', '')) + int(''.join(self.meaning_number))


@dataclass
class YarxiReference(Reference):
    verified: bool = False
