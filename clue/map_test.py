from clue.cards import PEOPLE_CARDS, ROOM_CARDS
from clue.map import ROOM_NAME_TO_ROOM_INDEX, STARTING_POINT_TO_PLAYER_CARD


def test_starting_point_coherence() -> None:
    for c, m in zip(PEOPLE_CARDS, STARTING_POINT_TO_PLAYER_CARD.values()):
        assert c.name == m


def test_room_to_card_map() -> None:
    for name, idx in ROOM_NAME_TO_ROOM_INDEX.items():
        assert ROOM_CARDS[idx].name == name
