from clue.map import Board


def test_board(map_csv_location: str) -> None:
    board = Board(map_csv_location)
    assert board is not None


def test_distances_after_throw(map_csv_location: str) -> None:
    board = Board(map_csv_location)
    assert board is not None
    result = board.distances_after_throw(0, 1)
    assert result.shape == (1, 9)


def test_move_towards_room(map_csv_location: str) -> None:
    board = Board(map_csv_location)
    assert board is not None
    board.move_towards_room(player_idx=0, throw=6, room_idx=2)


def test_room_distances(map_csv_location: str) -> None:
    board = Board(map_csv_location)

    # put everyone in room r
    for r in range(9):
        for i in range(6):
            board.move_to_room(player_idx=i, room_idx=r)
            distances = board.distances[board.player_positions, :]

            assert distances[i, r] == 0
            assert distances[i, r + 9] == 0
