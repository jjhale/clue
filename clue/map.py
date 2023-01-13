import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

STARTING_POINT_TO_PLAYER_CARD = {
    "ss": "Miss Scarlet",
    "sm": "Colonel Mustard",
    "sw": "Mrs White",
    "sg": "Mr Green",
    "smp": "Mrs Peacock",
    "spp": "Professor Plum",
}

ROOM_NAME_TO_ROOM_INDEX = {
    "hall": 0,
    "lounge": 1,
    "dinning room": 2,
    "kitchen": 3,
    "ball room": 4,
    "conservatory": 5,
    "billiard room": 6,
    "library": 7,
    "study": 8,
}


def load_map(filename: str = "map49.csv") -> List[List[str]]:
    with open(filename, newline="") as csv_file:
        reader = csv.reader(csv_file)
        grid: List[List[str]] = []
        for row in reader:
            grid.append(list(row))

    return grid


def print_grid(
    grid: List[List[str]], overlay: Dict[Tuple[int, int], str] | None = None
) -> None:
    symbols_3: Dict[str, str] = {
        "0": "\u2588\u2588\u2588",  # filled in block
        "1": "[ ]",  # ""\u2591",  # Light shade block
        "dn": "\u2594\u2594\u2594",  # Upper 1/8th
        "de": "  \u2590",  # Right 1/8th
        "ds": "\u2582\u2582\u2582",  # Lower 1/8
        "dw": "\u258C  ",  # Left 1/8 block
        "": "   ",
    }

    if overlay is None:
        overlay = {}

    # could get fancy here
    #  https://www.fileformat.info/info/unicode/block/box_drawing/utf8test.htm

    for i, row in enumerate(grid):
        print()
        for j, col in enumerate(row):
            key: str
            if col == "":
                key = ""
            elif col in ["0", "1"]:
                key = str(col)
            elif col[0] == "d":
                key = col[:2]
            elif col[0] == "s":
                key = "1"
            else:
                key = ""

            if (i, j) in overlay:
                cell = symbols_3[key][0] + overlay[(i, j)] + symbols_3[key][2]
            else:
                cell = symbols_3[key]
            print(cell, end="")
    print()


@dataclass
class Square:
    """Class for keeping track of the properties of a square."""

    i: int
    j: int
    room: str | None
    connected_squares: List[int]  # index of connected squares.


@dataclass
class DoorData:
    """Class for keeping track of the properties of a square."""

    direction: str
    room: str


class Board:
    def __init__(self, map_csv: str):
        self.grid = load_map(map_csv)

        # parse_map_grid will fill in these variables
        self.starting_points: Dict[str, Square] = {}
        self.door_data: List[DoorData] = []
        self.locations: List[Square] = []

        self.room_to_doors: Dict[str, List[int]] = defaultdict(list)

        self.build_locations(self.grid)

        self.num_players = len(self.starting_points)

        self.num_doors: int = len(self.door_data)
        self.num_positions: int = len(self.locations)
        self.location_map = {
            (square.i, square.j): idx for idx, square in enumerate(self.locations)
        }

        self.secret_passages: Dict[int, int] = self.build_secret_passages()

        # Now we do the graph stuff
        self.populate_connected_squares()

        # Location index for each player as ordered by STARTING_POINT_TO_PLAYER_CARD
        self.player_positions_initial = [
            self.location_map[
                (self.starting_points[start_id].i, self.starting_points[start_id].j)
            ]
            for start_id in STARTING_POINT_TO_PLAYER_CARD
        ]
        self.player_positions = self.player_positions_initial.copy()

        # Pre create so that we don't need to reallocate each time.
        self._legal_positions_vector = np.zeros((self.num_positions,), dtype=np.int8)
        self.visited = np.zeros((self.num_positions,), dtype=np.int8)

    def reset_positions(self) -> None:
        self.player_positions = self.player_positions_initial.copy()

    @staticmethod
    def _is_a_square(code: str) -> bool:
        return code == "1" or code.startswith("s")

    def build_locations(self, grid: List[List[str]]) -> None:
        square_locations: List[Square] = []
        location_index: List[Square]

        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                if grid[i][j] == "":
                    continue

                # is it a door?
                if grid[i][j].startswith("d"):
                    # put the doors at the start of the location list
                    self.locations.append(
                        Square(i=i, j=j, room=grid[i][j], connected_squares=[])
                    )
                    # format of door name == "d[nsew]:(.*)"
                    door_datum = DoorData(direction=grid[i][j][1], room=grid[i][j][3:])
                    self.door_data.append(door_datum)

                    self.room_to_doors[door_datum.room].append(len(self.locations) - 1)

                elif grid[i][j].startswith("s"):
                    square = Square(i=i, j=j, room=None, connected_squares=[])
                    square_locations.append(square)
                    self.starting_points[grid[i][j]] = square

                elif grid[i][j] == "1":
                    square_locations.append(
                        Square(i=i, j=j, room=None, connected_squares=[])
                    )

        self.num_doors = len(self.locations)
        # first entries and the door locations, followed by the
        #  locations of the squares.
        self.locations.extend(square_locations)

    def build_secret_passages(self) -> Dict[int, int]:
        # secret passages (only works if there is one door per room)
        corner_rooms = ("lounge", "conservatory", "study", "kitchen")
        corner_doors = {
            dd.room: door_pos
            for door_pos, dd in enumerate(self.door_data)
            if dd.room in corner_rooms
        }
        return {
            corner_doors["lounge"]: corner_doors["conservatory"],
            corner_doors["conservatory"]: corner_doors["lounge"],
            corner_doors["study"]: corner_doors["kitchen"],
            corner_doors["kitchen"]: corner_doors["study"],
        }

    def populate_connected_squares(self) -> None:
        num_locations = len(self.locations)

        # Connect the squares (excluding the doors)
        for idx in range(self.num_doors, num_locations):
            square = self.locations[idx]
            i = square.i
            j = square.j
            # look to the left:
            if Board._is_a_square(self.grid[i][j - 1]):
                left_square_idx = self.location_map[(i, j - 1)]
                square.connected_squares.append(left_square_idx)
                self.locations[left_square_idx].connected_squares.append(idx)

            # Look above
            if Board._is_a_square(self.grid[i - 1][j]):
                square_above_idx = self.location_map[(i - 1, j)]
                square.connected_squares.append(square_above_idx)
                self.locations[square_above_idx].connected_squares.append(idx)

        # So graph now has all the boxes on the board connected.
        # Now need to connect the rooms?
        for idx in range(self.num_doors):
            door = self.locations[idx]
            i = door.i
            j = door.j
            direction = self.door_data[idx].direction
            connecting_square: Tuple[int, int]
            if direction == "n":
                connecting_square = i - 1, j
            elif direction == "e":
                connecting_square = i, j + 1
            elif direction == "s":
                connecting_square = i + 1, j
            else:  # direction == "w":
                connecting_square = i, j - 1

            connecting_square_idx = self.location_map[connecting_square]
            door.connected_squares.append(connecting_square_idx)
            self.locations[connecting_square_idx].connected_squares.append(idx)

    def legal_positions(self, player_idx: int, throw: int) -> np.ndarray:

        initial_position: int = self.player_positions[player_idx]
        # clear the legal positions
        self._legal_positions_vector.fill(0)

        # Can take secret passages
        if initial_position in self.secret_passages:
            self._legal_positions_vector[self.secret_passages[initial_position]] = 1

        # If we are in a room we can leave from either door:
        starting_points = []
        blocked_doors = []  # you may not exit and enter same room in a turn
        if initial_position < self.num_doors:
            starting_points = self.room_to_doors[self.door_data[initial_position].room]
            blocked_doors = starting_points
        else:
            starting_points = [initial_position]

        for starting_point in starting_points:
            # init the squares you can't go thru
            self.visited.fill(0)
            for door in blocked_doors:
                self.visited[door] = 1

            for idx in self.player_positions:
                if idx >= self.num_doors:
                    self.visited[idx] = 1

            self.follow_path(
                starting_point=starting_point,
                current_position=starting_point,
                distance=throw,
            )

        return self._legal_positions_vector

    def is_in_room(self, player_idx: int) -> bool:
        return self.player_positions[player_idx] < self.num_doors

    def which_room(self, player_idx: int) -> int:
        room_name = self.door_data[self.player_positions[player_idx]].room
        return ROOM_NAME_TO_ROOM_INDEX[room_name]

    def follow_path(
        self,
        starting_point: int,
        current_position: int,
        distance: int,
    ) -> None:
        # Need the starting point so that we don't get stuck in a room.
        if distance == 0 or (
            starting_point != current_position and self.locations[current_position].room
        ):
            # we have reached the end of the path by running out of steps
            #  or by finding a door.
            self._legal_positions_vector[current_position] = 1
            return

        # mark our position as visited
        self.visited[current_position] = 1

        # see if we can visit any neighbours
        for next_idx in self.locations[current_position].connected_squares:
            if not self.visited[next_idx]:
                self.follow_path(starting_point, next_idx, distance - 1)

        # Unmark current location as visited
        self.visited[current_position] = 0


if __name__ == "__main__":
    board = Board("map49.csv")
    player_idx = 0
    legal_positions = board.legal_positions(player_idx=player_idx, throw=6)
    initial_pos = board.player_positions[player_idx]
    initial_overlay = {
        (board.locations[initial_pos].i, board.locations[initial_pos].j): "S"
    }
    legal_overlay = {
        (board.locations[i].i, board.locations[i].j): "e"
        for i, legal in enumerate(legal_positions)
        if legal
    }

    print_grid(board.grid, {**legal_overlay, **initial_overlay})
    print(f"Num pos: {board.num_positions}")
