import numpy as np

from clue.state import CardState, StepKind


def test_suggestion_translation_one_hot() -> None:
    for room_idx in range(9):
        for person_idx in range(6):
            for weapon_idx in range(6):
                one_hot = CardState.suggestion_one_hot(
                    person_idx=person_idx, weapon_idx=weapon_idx, room_idx=room_idx
                )

                p, w, r = CardState.suggestion_one_hot_decode(int(one_hot.argmax()))

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

                p, w, r = CardState.suggestion_from_deck_vector(deck)

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
    p, w, r = CardState.suggestion_from_deck_vector(suggestion[0:21])

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
    p, w, r = CardState.suggestion_from_deck_vector(suggestion[0:21])

    assert (r, p, w) == (room, person, weapon)

    assert suggestion[21:27].argmax() == suggestor

    assert suggestion[27:33].all()
    assert (suggestion[33:39] == np.array([0, 0, 0, 0, 0, 0])).all()


def test_card_state_init(map_csv_location: str) -> None:
    card_state = CardState(map_csv_location, max_players=6)
    assert card_state is not None


def test_legal_suggestion(map_csv_location: str) -> None:
    card_state = CardState(map_csv_location, max_players=6)

    room = 0
    # move first play to first room
    card_state.board.move_to_room(player_idx=0, room_idx=room)

    assert card_state.board.is_in_room(player_idx=0)
    assert card_state.board.which_room(player_idx=0) == room

    card_state.current_player = 0
    card_state.current_step_kind = StepKind.SUGGESTION

    legal_actions = card_state.legal_actions()

    legal_suggestions = list(legal_actions[9 : (9 + 324)])

    for i, val in enumerate(legal_suggestions):
        if val:
            _, _, room_decoded = card_state.suggestion_one_hot_decode(i)

            assert (
                room_decoded == room
            ), "Can only make suggestion about the room you are in"
