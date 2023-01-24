import numpy as np

from clue.state import CardState


def test_suggestion_translation_one_hot() -> None:
    for room_idx in range(9):
        for person_idx in range(6):
            for weapon_idx in range(6):
                one_hot = CardState.suggestion_one_hot(
                    room_idx=room_idx, person_idx=person_idx, weapon_idx=weapon_idx
                )

                r, p, w = CardState.suggestion_one_hot_decode(one_hot)

                assert r == room_idx
                assert p == person_idx
                assert w == weapon_idx


def test_suggestion_translation_deck() -> None:
    for room_idx in range(9):
        for person_idx in range(6):
            for weapon_idx in range(6):
                deck = CardState.suggestion_to_deck_vector(
                    room_idx=room_idx, person_idx=person_idx, weapon_idx=weapon_idx
                )

                assert deck.shape == (21,)

                r, p, w = CardState.suggestion_from_deck_vector(deck)

                assert r == room_idx
                assert p == person_idx
                assert w == weapon_idx


def test_encode_suggestion_history() -> None:
    room = 1
    person = 2
    weapon = 3
    suggestor = 4
    can_disprove_idx = 5

    suggestion = CardState.encode_suggestion_history(
        room_idx=room,
        person_idx=person,
        weapon_idx=weapon,
        suggestor_idx=suggestor,
        cant_disprove=np.array([1, 1, 1, 1, 1, 1]),
        can_disprove_idx=can_disprove_idx,
    )

    assert suggestion.shape == (39,)
    r, p, w = CardState.suggestion_from_deck_vector(suggestion[0:21])

    assert (r, p, w) == (room, person, weapon)

    assert suggestion[21:27].argmax() == suggestor

    assert suggestion[27:33].all()
    assert (suggestion[33:39] == np.array([0, 0, 0, 0, 0, 1])).all()


def test_encode_suggestion_history_no_disprove() -> None:
    room = 1
    person = 2
    weapon = 3
    suggestor = 4
    can_disprove_idx = -1

    suggestion = CardState.encode_suggestion_history(
        room_idx=room,
        person_idx=person,
        weapon_idx=weapon,
        suggestor_idx=suggestor,
        cant_disprove=np.array([1, 1, 1, 1, 1, 1]),
        can_disprove_idx=can_disprove_idx,
    )

    assert suggestion.shape == (39,)
    r, p, w = CardState.suggestion_from_deck_vector(suggestion[0:21])

    assert (r, p, w) == (room, person, weapon)

    assert suggestion[21:27].argmax() == suggestor

    assert suggestion[27:33].all()
    assert (suggestion[33:39] == np.array([0, 0, 0, 0, 0, 0])).all()
