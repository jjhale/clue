import csv
from collections import defaultdict
from typing import List, Dict, Tuple, Iterator


def load_map(filename="map49.csv"):
    with open(filename, newline='') as csv_file:
        reader = csv.reader(csv_file)
        grid: List[List[str]] = []
        for row in reader:
            grid.append(list(row))

    return grid


def print_grid(grid: List[List[str]],
               overlay: Dict[Tuple[int, int], str] | None = None):
    symbols_3: Dict[str, str] = {
        "0": "\u2588\u2588\u2588",  # filled in block
        "1": "[ ]",  # ""\u2591",  # Light shade block
        "dn": "\u2594\u2594\u2594",  # Upper 1/8th
        "de": "  \u2590",  # Right 1/8th
        "ds": "\u2582\u2582\u2582",  # Lower 1/8
        "dw": "\u258C  ",  # Left 1/8 block
        "": "   "
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


from dataclasses import dataclass


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

        self.num_doors: int = len(self.door_data)
        self.num_positions: int = len(self.locations)
        self.location_map = {
            (square.i, square.j): idx
            for idx, square in enumerate(self.locations)
        }

        self.secret_passages: Dict[int, int] = self.build_secret_passages()

        # Now we do the graph stuff
        self.populate_connected_squares()

        # Build the subgraphs for each
        self.player_positions_initial = {
            player: self.location_map[(square.i, square.j)]
            for player, square in self.starting_points.items()
        }
        self.player_positions = self.player_positions_initial.copy()

    @staticmethod
    def _is_a_square(code: str):
        return code == "1" or code.startswith("s")

    def build_locations(self, grid: List[List[str]]):
        square_locations: List[Square] = []
        location_index: List[Square]

        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                if grid[i][j] == "":
                    continue

                # is it a door?
                if grid[i][j].startswith("d"):
                    # put the doors at the start of the location list
                    self.locations.append(Square(
                        i=i, j=j, room=grid[i][j], connected_squares=[]
                    ))
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

    def legal_positions(self,
                        initial_position: int,
                        throw: int) -> List[bool]:
        legal_positions = [False] * self.num_positions

        # Can take secret passages
        if initial_position in self.secret_passages:
            legal_positions[self.secret_passages[initial_position]] = True

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
            visited = [False] * self.num_positions
            for door in blocked_doors:
                visited[door] = True

            for idx in self.player_positions.values():
                if idx >= self.num_doors:
                    visited[idx] = True

            self.follow_path(
                starting_point=starting_point,
                current_position=starting_point,
                legal_positions=legal_positions,
                visited=visited,
                distance=throw,
            )

        return legal_positions

    def follow_path(
            self,
            starting_point: int,
            current_position: int,
            legal_positions: List[bool],
            visited: List[bool],
            distance: int,
    ) -> None:
        # Need the starting point so that we don't get stuck in a room.
        if distance == 0 or (starting_point != current_position and self.locations[
            current_position].room):
            # we have reached the end of the path by running out of steps
            #  or by finding a door.
            legal_positions[current_position] = True
            return

        # mark our position as visited
        visited[current_position] = True

        # see if we can visit any neighbours
        for next_idx in self.locations[current_position].connected_squares:
            if not visited[next_idx]:
                self.follow_path(starting_point, next_idx, legal_positions, visited,
                                 distance - 1)

        # Unmark current location as visited
        visited[current_position] = False


if __name__ == "__main__":
    board = Board("map49.csv")
    initial_pos = 0
    legal_positions = board.legal_positions(initial_position=initial_pos, throw=6)

    initial_overlay = {
        (board.locations[initial_pos].i, board.locations[initial_pos].j): "S"}
    legal_overlay = {
        (board.locations[i].i, board.locations[i].j): "e"
        for i, legal in enumerate(legal_positions) if legal
    }

    print_grid(board.grid, {**legal_overlay, **initial_overlay})
