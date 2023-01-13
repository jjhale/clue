import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict

import numpy as np

class CardType(Enum):
    PERSON = "person"
    WEAPON = "weapon"
    ROOM = "room"



@dataclass
class Card:
    idx: int
    name: str
    type: CardType

@dataclass
class Envelope:
    person: Card
    weapon: Card
    room: Card


PEOPLE_CARDS = tuple(
    Card(i, name, CardType.PERSON) for i,name in enumerate((
    "Miss Scarlet",
    "Colonel Mustard",
    "Mrs White",
    "Mr Green",
    "Mrs Peacock",
    "Professor Plum")
    )
)

WEAPON_CARDS = tuple(
Card(i+len(PEOPLE_CARDS), name, CardType.WEAPON) for i, name in enumerate((
    "Rope",
    "Lead Pipe",
    "Knife",
    "Wrench",
    "Candlestick",
    "Pistol",
)))

ROOM_CARDS = tuple(
Card(i + len(PEOPLE_CARDS) + len(WEAPON_CARDS), name, CardType.WEAPON) for i, name in enumerate((
    "hall",
    "lounge",
    "dinning room",
    "kitchen",
    "ball room",
    "conservatory",
    "billiard room",
    "library",
    "study",
))
)

DECK = PEOPLE_CARDS + WEAPON_CARDS + ROOM_CARDS

