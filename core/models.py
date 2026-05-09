from dataclasses import dataclass, field
from typing import List

@dataclass
class Token:
    text: str
    reading: str
    url: str

@dataclass
class Segment:
    start: float
    end: float
    text: str
    tokens: List[Token] = field(default_factory=list)

@dataclass
class Lesson:
    filename: str
    segments: List[Segment]