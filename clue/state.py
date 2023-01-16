import random
from enum import Enum
from typing import Dict, List, cast

import numpy as np

from clue.cards import DECK, PEOPLE_CARDS, ROOM_CARDS, WEAPON_CARDS, Envelope
from clue.map import Board

# Alway repreresent the state of the world from the
#  current player's point of view
PLAYER_REMAP = (
    (0, 1, 2, 3, 4, 5),
    (5, 0, 1, 2, 3, 4),
    (4, 5, 0, 1, 2, 3),
    (3, 4, 5, 0, 1, 2),
    (2, 3, 4, 5, 0, 1),
    (1, 2, 3, 4, 5, 0),
)

SUGGESTION_REMAP = tuple(
    tuple(
        list(range(21))
        + [21 + j for j in PLAYER_REMAP[i]]
        + [21 + 6 + j for j in PLAYER_REMAP[i]]
        + [21 + 6 + 6 + j for j in PLAYER_REMAP[i]]
    )
    for i in range(6)
)


class StepKind(Enum):
    MOVE = 0  # choosing where to place your token
    SUGGESTION = 1  # choosing a suggestion to make
    DISPROVE_SUGGESTION = 2  # choosing how to disprove a suggestion
    ACCUSATION = 3  # deciding to make an accusation or no accusation


class CardState:
    MAX_PLAYERS = 6
    SEQUENCE_MEMORY = 50

    def __init__(self, map_csv: str, max_players: int) -> None:
        """
        max_players : Constrain the complexity of the game by reducing the number
         of players.
        """

        self.board = Board(map_csv)
        self.max_players = max_players
        # Who has the weapons
        self.card_locations: List[int] = []
        self.players: List[int] = self.pick_players()
        self.active_players = np.zeros(self.max_players, dtype=np.int8)

        self.envelope: Envelope = Envelope(
            PEOPLE_CARDS[0], WEAPON_CARDS[0], ROOM_CARDS[0]
        )

        # The card knowledge of each player
        self.player_card_knowledge = np.zeros(
            (self.max_players, self.max_players, len(DECK)), dtype=np.int8
        )

        # History of the suggestions represented as a sequence:
        #    - DECK vector pick 3,  n=21
        #    - the suggestor pick 1,  n=6
        #    - can't disprove  pick all that apply, n=6
        #    - can disprove - pick 0 or 1, n=6
        #  so length == 21+6+6+6 = 39
        # ordered with the most recent first

        self.suggestions = np.zeros(
            (self.max_players, len(DECK) + (3 * len(PEOPLE_CARDS))),
            dtype=np.int8,
        )
        self.suggestion_count = 0

        #
        self.current_player = 0
        self.current_step_kind = StepKind.MOVE
        self.current_die_roll = random.randrange(1, 6)
        self.who_am_i = np.identity(self.MAX_PLAYERS, dtype=np.int8)
        self.current_step_kind_matrix = np.identity(len(StepKind), dtype=np.int8)
        self.player_position_matrix = np.zeros(
            (self.MAX_PLAYERS, self.board.num_positions), dtype=np.int8
        )

        self.new_game(self.players)

    def pick_players(self) -> List[int]:
        # You can have between 3 and 6 players:
        num_players = random.randint(3, self.max_players)

        # Miss Scarlett always goes first, so only get to pick from the other five:
        players = [0] + random.sample(range(1, 6), k=num_players - 1)
        # Sort inplace
        players.sort()
        return players

    def new_game(self, players: List[int]) -> None:
        """Start a new game
        players - the player ids for the folks playing.
        """
        if players[0] != 0:
            raise ValueError(
                "Miss Scarlet was not included in the player list - she "
                "must be the first player! Unable to player_idx without her."
            )

        self.players = players
        self.num_players = len(players)
        self.active_players.fill(0)
        for player_idx in self.players:
            self.active_players[player_idx] = 1

        # grab cards for the envelope:
        self.envelope = Envelope(
            person=random.choice(PEOPLE_CARDS),
            weapon=random.choice(WEAPON_CARDS),
            room=random.choice(ROOM_CARDS),
        )

        # Shuffle the other cards
        remaining_cards = [
            idx
            for idx in list(range(len(DECK)))
            if (
                idx != self.envelope.person.idx
                and idx != self.envelope.weapon.idx
                and idx != self.envelope.room.idx
            )
        ]
        random.shuffle(remaining_cards)

        # Assign the cards to the players:
        self.player_card_knowledge.fill(0)
        for i, card_idx in enumerate(remaining_cards):
            player_idx = self.players[i % self.num_players]
            # Initially each player only knows their own cards
            self.player_card_knowledge[player_idx][player_idx][card_idx] = 1

        # Reset the suggestions and disproving info
        #  - this knowledge is public:
        self.suggestions.fill(0)
        self.suggestion_count = 0

        self.current_player = 0
        self.current_step_kind = StepKind.MOVE
        self.current_die_roll = random.randint(1, 6)

    def get_player_knowledge(self, player_idx: int) -> Dict:
        # TODO: decide if we make the order consistent?
        # want to make the order of these consistent, i.e.
        #  The current player should be the first entry
        #  followed by the next person in the order of player_idx.
        #
        # Which character you player_idx as should not really affect you
        # set the order to start with player_idx followed by the others
        # in order

        pm = PLAYER_REMAP[player_idx]
        spm = SUGGESTION_REMAP[player_idx]

        return {
            # Public knowledge:
            "step_kind": self.current_step_kind_matrix[self.current_step_kind.value][
                :
            ],  # 1x4
            "active players": self.active_players[pm],  # 1x6
            "player locations": self._make_player_positions()[pm, :],  # 6x205 (1230)
            "suggestions": self.suggestions[:, spm],  # 50x39 (1950)
            # Private knowledge:
            "card locations": self.player_card_knowledge[
                player_idx, pm, :
            ],  # 6x21 (126)
        }

    def legal_actions(self) -> tuple:
        # Action space:
        #  - move positions: 1x205
        if self.current_step_kind == StepKind.MOVE:
            legal_positions = self.board.legal_positions(
                player_idx=self.current_player, throw=self.current_die_roll
            )
        else:
            legal_positions = np.zeros(self.board.num_positions)

        # Suggestions or accusations 9x6x6 (324)
        if self.current_step_kind == StepKind.SUGGESTION:
            suggestions_or_accusations = self._legal_suggestions()
        elif self.current_step_kind == StepKind.ACCUSATION:
            suggestions_or_accusations = self._legal_accusation()
        else:
            suggestions_or_accusations = np.zeros(
                (
                    len(ROOM_CARDS),
                    len(PEOPLE_CARDS),
                    len(WEAPON_CARDS),
                ),
                dtype=np.int8,
            )

        # disprove suggestion
        #  choosing which card to show room, weapon or person 1x21
        legal_disprove = self._legal_disprove()

        # TODO return legal actions...
        return legal_positions, suggestions_or_accusations, legal_disprove

    def _legal_suggestions(self) -> np.ndarray:
        suggestion = np.zeros(
            (
                len(ROOM_CARDS),
                len(PEOPLE_CARDS),
                len(WEAPON_CARDS),
            ),
            dtype=np.int8,
        )

        # Can only make a suggestion if player is in a room.
        if not self.board.is_in_room(player_idx=self.current_player):
            return suggestion

        current_room_idx = self.board.which_room(self.current_player)

        # allow any combination of people and weapons in that room
        suggestion[current_room_idx].fill(1)

        return suggestion

    def _legal_disprove(self) -> np.ndarray:
        # Can disprove using the intersection of the cards in the
        #  suggestion and the cards the player holds.
        return cast(
            np.ndarray,
            self.suggestions[0]
            * self.player_card_knowledge[self.current_player][self.current_player],
        )

    def _legal_accusation(self) -> np.ndarray:
        # Don't need to be in the room to make the accusation and so can always
        # make any accusation.
        #
        # Forcing it not to make any accusation
        #   which includes any cards that it has seen
        seen_cards = cast(
            np.ndarray, np.any(self.player_card_knowledge[self.current_player])
        )

        suggestion = np.ones(
            (
                len(ROOM_CARDS),
                len(PEOPLE_CARDS),
                len(WEAPON_CARDS),
            ),
            dtype=np.int8,
        )

        # Exclude any containing the seen people cards
        suggestion[:, seen_cards[0:6], :] = 0

        # Exclude seen weapons:
        suggestion[:, :, seen_cards[6:12]] = 0

        # Exclude seen Rooms
        suggestion[seen_cards[12:], :, :] = 0

        return suggestion

    def _make_player_positions(self) -> np.ndarray:
        self.player_position_matrix.fill(0)

        for player_idx, pos_idx in enumerate(self.board.player_positions):
            self.player_position_matrix[player_idx][pos_idx] = 1

        return self.player_position_matrix
