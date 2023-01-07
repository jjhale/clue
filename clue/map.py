import csv
from typing import List, Dict, Tuple, Iterator
import networkx as nx

def load_map(filename="map49.csv"):
    with open(filename, newline='') as csv_file:
        reader = csv.reader(csv_file)
        grid: List[List[str]] = []
        for row in reader:
            grid.append(list(row))

    return grid


def print_grid(grid: List[List[str]]):
    symbols : Dict[str, str] = {
        "0": "\u2588",  # filled in block
        "1": "\u2B1A", #""\u2591",  # Light shade block
        "dn": "\u2594",  # Upper 1/8th
        "de": "\u2595",  # Right 1/8th
        "ds": "\u2581",  # Lower 1/8
        "dw": "\u258F",  # Left 1/8 block
        "": " "
    }

    symbols_2: Dict[str, str] = {
        "0": "\u2588\u2588",  # filled in block
        "1": "[]",  # ""\u2591",  # Light shade block
        "dn": "\u2594\u2594",  # Upper 1/8th
        "de": "\u2595\u2595",  # Right 1/8th
        "ds": "\u2581\u2581",  # Lower 1/8
        "dw": "\u258F\u258F",  # Left 1/8 block
        "": "  "
    }
    symbols_3: Dict[str, str] = {
        "0": "\u2588\u2588\u2588",  # filled in block
        "1": "[ ]",  # ""\u2591",  # Light shade block
        "dn": "\u2594\u2594\u2594",  # Upper 1/8th
        "de": "  \u2590",  # Right 1/8th
        "ds": "\u2582\u2582\u2582",  # Lower 1/8
        "dw": "\u258C  ",  # Left 1/8 block
        "": "   "
    }

    # could get fancy here
    #  https://www.fileformat.info/info/unicode/block/box_drawing/utf8test.htm

    for row in grid:
        print()
        for col in row:
            key : str
            if col == "":
                key = ""
            elif col in ["0","1"]:
                key = str(col)
            elif col[0] == "d":
                key = col[:2]
            elif col[0] == "s":
                key = "1"
            else:
                key = ""
            print(symbols_3[key], end="")
    print()


# def parse_map_grid(grid: List[List[str]]) -> Tuple[
#                                              nx.Graph,
# Dict[str, int],
# Dict[Tuple[int, int], str],
# Dict[Tuple[int, int], Tuple[int, int]]]:
#     # iterate over the grid considering two neighbors
#     #              (i-1,j)
#     #                  |
#     #                  ?
#     #                  |
#     #   (i, j-1) -?- (i,j)
#     # The grid should have an empty border around it so can just
#     # handle up to the n-1th element.
#     # where the i is the row and the j is the column.
#     # We are using an origin at the top left of the board.
#     #
#     # This should return the graph of the board which should
#     # be used to calculate movement options.
#
#     graph = nx.Graph()
#
#     num_locations = len(location_index)
#
#     # Each location becomes a node in the graph
#     graph.add_nodes_from(range(num_locations))
#
#     # Connect the squares
#     for loc in range(num_doors, num_locations):
#         i,j = node_id = location_index[loc]
#         # look to the left:
#         if is_a_square(grid[i][j-1]):
#             left_node = location_map[(i, j - 1)]
#             graph.add_edge( loc, left_node)
#
#         # Look above
#         if is_a_square(grid[i-1][j]):
#             node_above = location_map[(i-1,j)]
#             graph.add_edge(loc, node_above)
#
#     # So graph now has all the boxes on the board connected.
#     # Now need to connect the rooms?
#     for loc in range(num_doors):
#         door = door_data[loc]
#         i,j = location_index[loc]
#         direction = door[1]
#         connecting_square : Tuple[int,int]
#         if direction == "n":
#             connecting_square = i-1, j
#         elif direction == "e":
#             connecting_square = i, j+1
#         elif direction == "s":
#             connecting_square = i+1, j
#         else:  # direction == "w":
#             connecting_square = i, j-1
#
#         graph.add_edge(loc, location_map[connecting_square])
#
#     # secret passages (only works if there is one door per room)
#     corner_rooms = ("lounge", "conservatory","study", "kitchen")
#     corner_doors = {
#         door_name[3:]:door_pos
#             for door_pos, door_name in enumerate(door_data)
#             if door_name[3:] in corner_rooms
#     }
#     secret_passages : Dict[int, int] = {
#         corner_doors["lounge"]:corner_doors["conservatory"],
#         corner_doors["conservatory"]:corner_doors["lounge"],
#         corner_doors["study"]: corner_doors["kitchen"],
#         corner_doors["kitchen"]: corner_doors["study"],
#     }
#
#     starting_points_by_index = {person:location_map[pos] for person, pos in starting_points.items()}
#
#     return graph, starting_points_by_index, location_index, location_map, door_data, secret_passages

class Board:
    def __init__(self, map_csv: str):
        self.grid = load_map(map_csv)

        # parse_map_grid will fill in these variables
        self.starting_points: Dict[str, int] = {}
        self.door_data: List[str] = []
        self.location_index: List[Tuple[int, int]] = []

        self.build_locations(self.grid)

        self.num_doors: int = len(self.door_data)
        self.num_positions : int = len(self.location_index)
        self.location_map = {pos: i for i, pos in enumerate(self.location_index)}

        self.secret_passages: Dict[int, int] = self.build_secret_passages()

        # Now we do the graph stuff
        self.graph = self.build_graph()

        # Build the subgraphs for each


        self.player_positions = self.starting_points.copy()

    @staticmethod
    def _is_a_square(code: str):
        return code == "1" or code.startswith("s")

    def build_locations(self, grid:List[List[str]]):
        starting_points: Dict[str, Tuple[int, int]] = {}
        square_locations: List[Tuple[int, int]] = []
        location_index: List[Tuple[int, int]]

        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                node_id = (i, j)
                if grid[i][j] == "":
                    continue

                # is it a door?
                if grid[i][j].startswith("d"):
                    self.location_index.append(node_id)
                    self.door_data.append(grid[i][j])
                if grid[i][j].startswith("s"):
                    starting_points[grid[i][j]] = (i, j)

                if Board._is_a_square(grid[i][j]):
                    square_locations.append(node_id)

        self.num_doors = len(self.location_index)
        # first entries and the door locations, followed by the
        #  locations of the squares.

        self.location_index.extend(square_locations)
        self.starting_points = {
            person:self.location_map[pos]
            for person, pos in starting_points.items()
        }

    def build_secret_passages(self)->Dict[int,int]:
        # secret passages (only works if there is one door per room)
        corner_rooms = ("lounge", "conservatory", "study", "kitchen")
        corner_doors = {
            door_name[3:]: door_pos
            for door_pos, door_name in enumerate(self.door_data)
            if door_name[3:] in corner_rooms
        }
        return {
            corner_doors["lounge"]: corner_doors["conservatory"],
            corner_doors["conservatory"]: corner_doors["lounge"],
            corner_doors["study"]: corner_doors["kitchen"],
            corner_doors["kitchen"]: corner_doors["study"],
        }

    def build_graph(self) -> nx.Graph:
        graph = nx.Graph()

        num_locations = len(self.location_index)

        # Each location becomes a node in the graph
        graph.add_nodes_from(range(num_locations))

        # Connect the squares
        for loc in range(self.num_doors, num_locations):
            i, j = node_id = self.location_index[loc]
            # look to the left:
            if Board._is_a_square(grid[i][j - 1]):
                left_node = self.location_map[(i, j - 1)]
                graph.add_edge(loc, left_node)

            # Look above
            if Board._is_a_square(grid[i - 1][j]):
                node_above = self.location_map[(i - 1, j)]
                graph.add_edge(loc, node_above)

        # So graph now has all the boxes on the board connected.
        # Now need to connect the rooms?
        for loc in range(self.num_doors):
            door = self.door_data[loc]
            i, j = self.location_index[loc]
            direction = door[1]
            connecting_square: Tuple[int, int]
            if direction == "n":
                connecting_square = i - 1, j
            elif direction == "e":
                connecting_square = i, j + 1
            elif direction == "s":
                connecting_square = i + 1, j
            else:  # direction == "w":
                connecting_square = i, j - 1

            graph.add_edge(loc, self.location_map[connecting_square])

        return graph

    def build_subgraphs(self)->List[nx.Graph]:
        lengths : Iterator[Tuple[int, Dict[int,int]]] = nx.all_pairs_shortest_path_length(self.graph, cutoff=6)
        # make the dict:
        empty_graph = nx.Graph()
        subgraphs : List[nx.Graph] = ...

    def legal_positions(self, graph: nx.Graph,
                        initial_position: int,
                        other_player_positions: List[Tuple[int, int]],
                        throw: int) -> List[bool]:
        legal_positions = [False] * self.num_positions

        # Can take secret passages
        if initial_position in self.secret_passages:
            legal_positions[self.secret_passages[initial_position]] = True

        # The locations of other players outside of rooms.
        blocked_locations = [
            v
            for v in self.player_positions.values()
            if v >= self.num_doors and v != initial_position
        ]

        # make the subgraph of things that are upto dice throw away:
        subgraph = nx.generators.ego_graph(graph,
                                           initial_position,
                                           radius=throw,
                                           undirected=True)







if __name__ == "__main__":
    grid = load_map()
    print_grid(grid)



