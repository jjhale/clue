import csv
import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

STARTING_POINT_TO_PLAYER_CARD = {
    "ss": "Miss Scarlet",
    "sm": "Colonel Mustard",
    "sw": "Mrs White",
    "sg": "Mr Green",
    "smp": "Mrs Peacock",
    "spp": "Professor Plum",
}

ONE_CHAR_PLAYER = ("S", "M", "W", "G", "P", "π")

ROOM_NAMES = [
    "hall",
    "lounge",
    "dining room",
    "kitchen",
    "ballroom",
    "conservatory",
    "billiard room",
    "library",
    "study",
]

# Where to display a player in a given room:
PLAYER_ROOM_POSITIONS = {
    "hall": (2, 10),
    "lounge": (3, 20),
    "dining room": (11, 20),
    "kitchen": (20, 20),
    "ballroom": (21, 10),
    "conservatory": (23, 1),
    "billiard room": (16, 1),
    "library": (9, 1),
    "study": (3, 1),
}


ROOM_NAME_TO_ROOM_INDEX = {name: idx for idx, name in enumerate(ROOM_NAMES)}


def load_map(filename: str = "map49.csv") -> List[List[str]]:
    with open(filename, newline="") as csv_file:
        reader = csv.reader(csv_file)
        grid: List[List[str]] = []
        for row in reader:
            grid.append(list(row))

    return grid


def print_grid(
    grid: List[List[str]], overlay: Optional[Dict[Tuple[int, int], str]] = None
) -> None:
    print(grid_string(grid, overlay))


def grid_string(
    grid: List[List[str]],
    overlay: Optional[Dict[Tuple[int, int], str]] = None,
    width: int = 3,
) -> str:
    symbols_3: Dict[str, str] = {
        "0": "\u2588\u2588\u2588",  # filled in block
        "1": "[ ]",  # ""\u2591",  # Light shade block
        "dn": "\u2594\u2594\u2594",  # Upper 1/8th
        "de": "  \u2590",  # Right 1/8th
        "ds": "\u2582\u2582\u2582",  # Lower 1/8
        "dw": "\u258C  ",  # Left 1/8 block
        "": "   ",
    }
    symbols_4: Dict[str, str] = {
        "0": "\u2588\u2588\u2588\u2588",  # filled in block
        "1": "[  ]",  # ""\u2591",  # Light shade block
        "dn": "\u2594\u2594\u2594\u2594",  # Upper 1/8th
        "de": "   \u2590",  # Right 1/8th
        "ds": "\u2582\u2582\u2582\u2582",  # Lower 1/8
        "dw": "\u258C   ",  # Left 1/8 block
        "": "    ",
    }

    symbols_lut = {
        3: symbols_3,
        4: symbols_4,
    }
    symbols = symbols_lut[width]

    if overlay is None:
        overlay = {}

    # could get fancy here
    #  https://www.fileformat.info/info/unicode/block/box_drawing/utf8test.htm

    result: List[str] = []

    for i, row in enumerate(grid):
        result.append("\n")
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
                cell = symbols[key][0] + overlay[(i, j)] + symbols[key][-1]
            else:
                cell = symbols[key]
            result.append(cell)
    result.append("\n")

    return "".join(result)


@dataclass
class Square:
    """Class for keeping track of the properties of a square."""

    i: int
    j: int
    room: Optional[str]
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

        # Now we do the graph stuff
        self.populate_connected_squares()
        self.build_secret_passages()

        # The 209x18 - 209 positions, 9 direct distances, 9 via room
        self.distances = np.concatenate(
            (
                self.build_distances(thru_room=False),
                self.build_distances(thru_room=True),
            ),
            axis=1,
        )

        self.max_distance = np.max(self.distances)

        # Location index for each player as ordered by STARTING_POINT_TO_PLAYER_CARD
        self.player_positions_initial = [
            self.location_map[
                (self.starting_points[start_id].i, self.starting_points[start_id].j)
            ]
            for start_id in STARTING_POINT_TO_PLAYER_CARD
        ]
        self.player_positions = self.player_positions_initial.copy()
        self.player_position_matrix = np.zeros(
            (self.num_players, self.num_positions), dtype=np.int8
        )
        self.reset_positions()

        # Pre create so that we don't need to reallocate each time.
        self._legal_positions_vector = np.zeros((self.num_positions,), dtype=np.int8)
        self.visited = np.zeros((self.num_positions,), dtype=np.int8)

    def reset_positions(self) -> None:
        self.player_positions = self.player_positions_initial.copy()
        for player_idx, position in enumerate(self.player_positions):
            self.player_position_matrix[player_idx][position] = 1

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

    def set_location(self, player_idx: int, pos_idx: int) -> None:
        # pos_idx = int(np.argmax(pos_vector))
        if pos_idx < 0 or pos_idx >= len(self.player_position_matrix[player_idx]):
            raise ValueError(f"Location position out of range: got {pos_idx}")
        self.player_positions[player_idx] = pos_idx
        self.player_position_matrix[player_idx].fill(0)
        self.player_position_matrix[player_idx][pos_idx] = 1

    def move_to_room(self, player_idx: int, room_idx: int) -> None:
        room_name = ROOM_NAMES[room_idx]
        door_idx = self.room_to_doors[room_name][0]
        self.set_location(player_idx, door_idx)

    def move_towards_room(self, player_idx: int, throw: int, room_idx: int) -> None:
        legal_positions = self.legal_positions(player_idx, throw)
        position_idxs = legal_positions.nonzero()[0]
        if len(position_idxs) == 1:
            self.set_location(player_idx, position_idxs[0])
            return

        direct = self.distances[position_idxs, room_idx]
        indirect = self.distances[position_idxs, room_idx + 9]

        d_min = direct.argmin()
        if direct[d_min] == 0:
            # if we can make it in this step go there!
            self.set_location(player_idx, position_idxs[d_min])
        else:
            # take shortest route that may involve stopping off in a room:
            i_min = indirect.argmin()
            self.set_location(player_idx, position_idxs[i_min])

    def build_secret_passages(self) -> None:
        # secret passages (only works if there is one door per room)
        corner_rooms = ("lounge", "conservatory", "study", "kitchen")

        # find the unconnected doors:
        corner_doors = {
            dd.room: door_pos
            for door_pos, dd in enumerate(self.door_data)
            if dd.room in corner_rooms
            and not self.locations[door_pos].connected_squares
        }

        # how to connect:
        passages = {
            "lounge": "conservatory",
            "conservatory": "lounge",
            "study": "kitchen",
            "kitchen": "study",
        }
        # wire them up
        for room, door_idx in corner_doors.items():
            self.locations[door_idx].connected_squares.append(
                corner_doors[passages[room]]
            )

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
            if connecting_square not in self.location_map:
                continue  # these are the secret passage doors.
            connecting_square_idx = self.location_map[connecting_square]
            door.connected_squares.append(connecting_square_idx)
            self.locations[connecting_square_idx].connected_squares.append(idx)

    def build_distances(self, thru_room: bool = False) -> np.array:
        # for each location we want the shortest distance to each room.
        #  initializing to a large distance.
        distances = np.ones([len(self.locations), len(self.room_to_doors)]) * 3000

        for room in self.room_to_doors:
            room_idx = ROOM_NAME_TO_ROOM_INDEX[room]

            for door_idx in self.room_to_doors[room]:
                dist_to_room = self._build_distances_to_all_from(door_idx, thru_room)
                distances[:, room_idx] = np.minimum(
                    distances[:, room_idx], dist_to_room
                )

            # fix room distances - route to one door may be longer than root to another
            for door_indicies in self.room_to_doors.values():
                distances[door_indicies, room_idx] = distances[
                    door_indicies, room_idx
                ].min()

        return distances

    def _build_distances_to_all_from(
        self, start_idx: int, thru_rooms: bool = False
    ) -> np.array:
        num_locations = len(self.locations)

        visited: np.array = np.zeros(num_locations)
        distances: np.array = np.ones(num_locations, dtype=np.int8) * 255
        distances[start_idx] = 0
        # the data in the heap is (distance, square_idx)

        min_heap: List[Tuple[int, int]] = []

        def push_to_heap(idx: int, dist: int) -> None:
            if idx < self.num_doors and thru_rooms:
                # Treat all doors in a room as a single node in the graph
                #  when we want to compute distance thru rooms.
                doors = self.room_to_doors[self.door_data[idx].room]
                for door_idx in doors:
                    heapq.heappush(min_heap, (dist, door_idx))
            else:
                heapq.heappush(min_heap, (dist, idx))

        push_to_heap(start_idx, 0)

        while min_heap:
            distance, sq_idx = heapq.heappop(min_heap)
            if visited[sq_idx]:
                continue
            distances[sq_idx] = distance
            visited[sq_idx] = 1

            for connected_idx in self.locations[sq_idx].connected_squares:
                if visited[connected_idx]:
                    continue
                push_to_heap(connected_idx, distance + 1)

        return distances

    def legal_move_towards(self, player_idx: int) -> np.array:
        legal = np.ones(len(ROOM_NAMES))
        # cant move towards the room you are in
        if self.is_in_room(player_idx):
            legal[self.which_room(player_idx)] = 0

        return legal

    def legal_positions(self, player_idx: int, throw: int) -> np.ndarray:
        """
        Figure out which positions a player can move to returns an indicator
        array of all positions with 1 if it was a legal position and 0 otherwise
        """
        initial_position: int = self.player_positions[player_idx]
        # clear the legal positions
        self._legal_positions_vector.fill(0)

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
        if not self._legal_positions_vector.any():
            # can't move - so stay put
            self._legal_positions_vector[initial_position] = 1
        return self._legal_positions_vector

    def is_in_room(self, player_idx: int) -> bool:
        return self.player_positions[player_idx] < self.num_doors

    def which_room(self, player_idx: int) -> int:
        room_name = self.door_data[self.player_positions[player_idx]].room
        return ROOM_NAME_TO_ROOM_INDEX[room_name]

    def distance_to_rooms(self, pos_idx: int) -> np.array:
        """Given a board position, return a 1x9 matrix of the min distantance from
        the pos_id to each room in card order"""
        return self.distances[pos_idx, 0:9]

    def distance_to_rooms_thru_room(self, pos_idx: int) -> np.array:
        """Given a board position, return a 1x9 matrix of the min distantance from
        the pos_id to each room in card order"""
        return self.distances[pos_idx, 9:]

    def distances_after_throw(self, player_idx: int, throw: int) -> np.array:
        """
        For a given throw of die we find the legal positions, then calculate the
        distances from those legal positions to each room, finally we calculate the
        minimum distances to each room over all those legal positions.
        """
        legal_positions = self.legal_positions(player_idx, throw)
        position_idxs = legal_positions.nonzero()

        return np.min(self.distances[position_idxs, :], axis=1).flatten()

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

    def generate_board_string(self) -> str:
        overlay: Dict[Tuple[int, int], str] = {}
        for player_idx in range(6):
            pos: tuple[int, int]
            if self.is_in_room(player_idx):
                room = self.which_room(player_idx)
                pos = PLAYER_ROOM_POSITIONS[ROOM_NAMES[room]]
                pos = (pos[0], pos[1] + player_idx)
            else:
                loc_idx = self.player_positions[player_idx]
                location = self.locations[loc_idx]
                pos = (location.i, location.j)
            overlay[pos] = ONE_CHAR_PLAYER[player_idx]

        return grid_string(self.grid, overlay)


if __name__ == "__main__":
    board = Board("map49.csv")
    # player_idx = 0
    # legal_positions = board.legal_positions(player_idx=player_idx, throw=6)
    # initial_pos = board.player_positions[player_idx]
    # initial_overlay = {
    #     (board.locations[initial_pos].i, board.locations[initial_pos].j): "S"
    # }
    # legal_overlay = {
    #     (board.locations[i].i, board.locations[i].j): "e"
    #     for i, legal in enumerate(legal_positions)
    #     if legal
    # }

    print("Starting points:")
    board_str = board.generate_board_string()
    print(board_str)

    for room in range(6):
        print(ROOM_NAMES[room])
        for player in range(6):
            board.move_to_room(player, room)
        board_str = board.generate_board_string()
        print(board_str)

    # print_grid(board.grid, {**legal_overlay, **initial_overlay})
    # print(f"Num pos: {board.num_positions}")

    # print out distance maps to rooms:

    for room_idx in range(9):
        print(f"\n\nRoom: {room_idx}")
        overlay = {
            (board.locations[i].i, board.locations[i].j): f"{int(dist):2}"
            for i, dist in enumerate(board.distances[:, room_idx])
        }
        overlay_2 = {
            (board.locations[i].i, board.locations[i].j): f"{int(dist):2}"
            for i, dist in enumerate(board.distances[:, room_idx + 9])
        }
        board_str = grid_string(board.grid, overlay, width=4)
        print(board_str)
        print(grid_string(board.grid, overlay_2, width=4))
