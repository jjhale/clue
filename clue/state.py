import random
from enum import Enum
from typing import Dict, List, Optional, Tuple, cast

import numpy as np

from clue.cards import DECK, PEOPLE_CARDS, ROOM_CARDS, WEAPON_CARDS, Envelope
from clue.map import ROOM_NAMES, Board

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

PLAYER_ORDERBY_PLAYER = (
    (0, 1, 2, 3, 4, 5),
    (1, 2, 3, 4, 5, 0),
    (2, 3, 4, 5, 0, 1),
    (3, 4, 5, 0, 1, 2),
    (4, 5, 0, 1, 2, 3),
    (5, 0, 1, 2, 3, 4),
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
    NUM_ROOMS = len(ROOM_CARDS)

    def __init__(
        self, map_csv: str, max_players: int, log_actions: bool = True
    ) -> None:
        """
        max_players : Constrain the complexity of the game by reducing the number
         of players.
        """

        self.board = Board(map_csv)
        self.max_players = max_players
        # Who has the weapons
        self.card_locations: List[int] = []
        self.players: List[int] = self.pick_players()

        # Who is holding cards:
        self.active_players = np.zeros(self.max_players, dtype=np.int8)

        # Who has made false accusations and can now only show cards
        self.false_accusers = np.zeros(self.max_players, dtype=np.int8)

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
            (
                CardState.SEQUENCE_MEMORY,
                len(DECK) + (3 * len(PEOPLE_CARDS)),
            ),
            dtype=np.int8,
        )
        self.suggestion_count = 0

        #
        self.current_player = 0
        self.current_step_kind = StepKind.MOVE
        self.current_die_roll = random.randrange(1, 6)
        self.who_am_i = np.identity(self.MAX_PLAYERS, dtype=np.int8)
        self.current_step_kind_matrix = np.identity(len(StepKind), dtype=np.int8)

        self.game_over = False
        self.winner: Optional[int] = None

        self.what_just_happened: List[str] = []
        self.should_log_actions = log_actions
        self.new_game(self.players)

    @staticmethod
    def suggestion_one_hot(
        person_idx: int,
        weapon_idx: int,
        room_idx: int,
    ) -> np.ndarray:
        suggestion = np.zeros(
            [len(PEOPLE_CARDS), len(WEAPON_CARDS), len(ROOM_CARDS)], dtype=np.int8
        )
        suggestion[person_idx, weapon_idx, room_idx] = 1
        return suggestion.ravel()

    @staticmethod
    def suggestion_one_hot_decode(one_hot_index: int) -> Tuple[int, int, int]:
        person, weapon, room = np.unravel_index(
            one_hot_index,
            (len(PEOPLE_CARDS), len(WEAPON_CARDS), len(ROOM_CARDS))
            # idx = one_hot.argmax()
        )
        return int(person), int(weapon), int(room)

    @staticmethod
    def suggestion_to_deck_vector(
        person_idx: int, weapon_idx: int, room_idx: int
    ) -> np.ndarray:
        """The 21 element vector representing each"""
        deck = np.zeros((21,), dtype=np.int8)
        deck[person_idx] = 1
        deck[6 + weapon_idx] = 1
        deck[6 + 6 + room_idx] = 1
        return deck

    @staticmethod
    def suggestion_from_deck_vector(deck: np.ndarray) -> Tuple[int, int, int]:
        """The 21 element vector representing each"""
        person = deck[0:6].argmax()
        weapon = deck[6:12].argmax()
        room = deck[12:21].argmax()
        return int(person), int(weapon), int(room)

    @staticmethod
    def encode_suggestion_history(
        room_idx: int,
        person_idx: int,
        weapon_idx: int,
        suggestor_idx: int,
        cant_disprove: np.ndarray,
        can_disprove_idx: int,
    ) -> np.ndarray:
        """Suggestion history vector is:
        s[0:21] - DECK vector pick 3,  n=21
        s[21:27]- the suggestor pick 1,  n=6
        s[27:33] can't disprove  pick all that apply, n=6
        s[33:39]- can disprove - pick 0 or 1, n=6
        """
        suggestion = np.zeros(39, dtype=np.int8)

        deck = CardState.suggestion_to_deck_vector(person_idx, weapon_idx, room_idx)
        suggestion[0:21] = deck

        suggestion[21 + suggestor_idx] = 1

        suggestion[27:33] = cant_disprove

        if can_disprove_idx != -1:
            suggestion[33 + can_disprove_idx] = 1

        return suggestion

    def log_action(self, message: str) -> None:
        if self.should_log_actions:
            self.what_just_happened.append(message)

    def make_suggestion(
        self, room_idx: int, person_idx: int, weapon_idx: int
    ) -> Tuple[np.ndarray, int]:
        """Given the current suggestion state figure out who can't disprove it
        returns - the flags of those who could not disprove and the idx of the
        player who can or -1
        """
        self.log_action(
            f"{PEOPLE_CARDS[self.current_player].name} suggested:"
            f"{PEOPLE_CARDS[person_idx].name} in the {ROOM_CARDS[room_idx].name} "
            f"with the {WEAPON_CARDS[weapon_idx].name}"
        )

        # move the person to the right room:
        self.board.move_to_room(person_idx, room_idx)

        suggested_cards = CardState.suggestion_to_deck_vector(
            person_idx, weapon_idx, room_idx
        )

        cant_disprove = np.zeros(6)
        can_disprove = -1

        for other_idx in PLAYER_ORDERBY_PLAYER[self.current_player][1:6]:
            other_cards = self.player_card_knowledge[other_idx][other_idx]
            if (suggested_cards * other_cards).any():
                can_disprove = other_idx
                self.log_action(f" - {PEOPLE_CARDS[other_idx].name} CAN disprove it")
                break
            else:
                cant_disprove[other_idx] = 1
                self.log_action(f" - {PEOPLE_CARDS[other_idx].name} can't disprove it")

        if can_disprove == -1:
            # No one was able to disprove us so go straight to making an accusation
            #  if desired
            self.current_player = self.current_player
            self.current_step_kind = StepKind.ACCUSATION
        else:
            # Let the disprover choose which card to show
            self.current_player = can_disprove
            self.current_step_kind = StepKind.DISPROVE_SUGGESTION

        return cant_disprove, can_disprove

    def show_player_card(self, disprover_idx: int, deck_idx: int) -> None:
        """A player shows one of their cards
        to the player which disproves their suggestion,
        the next step is for the suggestor to make an accusation
        or not
        """
        suggestor_idx = self.get_last_suggestor()

        # Update the suggestors knowledege:
        self.player_card_knowledge[suggestor_idx][disprover_idx][deck_idx] = 1

        self.log_action(
            f"{PEOPLE_CARDS[disprover_idx].name} disproves "
            f"{PEOPLE_CARDS[suggestor_idx].name}'s "
            f"suggestion by showing them {DECK[deck_idx].name}."
        )

        # Next it is the suggestors turn or make an accusation (or not)
        self.current_player = suggestor_idx
        self.current_step_kind = StepKind.ACCUSATION

    def get_last_suggestor(self) -> int:
        return int(self.suggestions[0, 21:27].argmax())

    def pick_players(self) -> List[int]:
        # You can have between 3 and 6 players:
        # num_players = random.randint(3, self.max_players)
        num_players = 6
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

        self.log_action(
            f"New game started with {len(players)} players: "
            + ", ".join(PEOPLE_CARDS[i].name for i in players)
            + "."
        )

        self.players = players
        self.num_players = len(players)
        self.active_players.fill(0)
        for player_idx in self.players:
            self.active_players[player_idx] = 1

        self.false_accusers = np.zeros(self.max_players, dtype=np.int8)

        # grab cards for the envelope:
        self.envelope = Envelope(
            person=random.choice(PEOPLE_CARDS),
            weapon=random.choice(WEAPON_CARDS),
            room=random.choice(ROOM_CARDS),
        )

        self.log_action(
            f"OMG there has been a murder: {self.envelope.person.name} did it in the "
            f"{self.envelope.room.name} with the {self.envelope.weapon.name}!"
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

        self.log_action("The cards were dealt:")

        for player_idx in self.players:
            card_names = [
                DECK[i].name
                for i, has_card in enumerate(
                    self.player_card_knowledge[player_idx][player_idx]
                )
                if has_card
            ]
            self.log_action(
                f"{PEOPLE_CARDS[player_idx].name} got:\n  - "
                + "\n  - ".join(card_names)
            )

        # Reset the suggestions and disproving info
        #  - this knowledge is public:
        self.suggestions.fill(0)
        self.suggestion_count = 0

        self.current_player = 0
        self.current_step_kind = StepKind.MOVE
        self.current_die_roll = random.randint(1, 6)

        self.log_action(
            f"{PEOPLE_CARDS[self.current_player].name} rolled {self.current_die_roll}"
        )

        self.game_over = False
        self.winner = None

    def next_move(self) -> None:
        self.current_player = self.next_player()
        self.current_step_kind = StepKind.MOVE
        self.current_die_roll = random.randint(1, 6)

        self.log_action(
            f"{PEOPLE_CARDS[self.current_player].name} rolled {self.current_die_roll}"
        )

    def move_player(self, new_position: int, towards_room: bool = False) -> None:
        if towards_room:
            self.board.move_towards_room(
                self.current_player, self.current_die_roll, new_position
            )
        else:
            self.board.set_location(self.current_player, new_position)

        if self.board.is_in_room(self.current_player):
            # TODO: help out by not allowing same player to repeat a suggestion they
            #  already made - if you have no legal suggestions, transition straight
            #  to accusation.
            self.current_step_kind = StepKind.SUGGESTION
            if self.should_log_actions:
                self.log_action(
                    f"{PEOPLE_CARDS[self.current_player].name} entered the "
                    f"{ROOM_NAMES[self.board.which_room(self.current_player)]} "
                )
                self.log_action(self.board.generate_board_string())

        else:
            if self.should_log_actions:
                pos_idx = self.board.player_positions[self.current_player]
                cell = self.board.locations[pos_idx]
                self.log_action(
                    f"{PEOPLE_CARDS[self.current_player].name} moved to cell "
                    f"{cell.i}, {cell.j}"
                )
                self.log_action(self.board.generate_board_string())
            self.next_move()

    def make_accusation(self, accusation_one_hot_idx: int) -> Optional[bool]:
        """The current player makes an accusation"""
        accuser_idx = self.current_player

        if accusation_one_hot_idx == 324:
            # The action is the don't make accusation action
            #  Just move on to the next player
            self.log_action(
                f"{PEOPLE_CARDS[self.current_player].name} decided not to make an"
                " accusation"
            )
            self.next_move()
            return None

        p, w, r = CardState.suggestion_one_hot_decode(accusation_one_hot_idx)
        correct = (
            ROOM_CARDS[r] == self.envelope.room
            and PEOPLE_CARDS[p] == self.envelope.person
            and WEAPON_CARDS[w] == self.envelope.weapon
        )

        self.log_action(
            f"{PEOPLE_CARDS[self.current_player].name} made an accusation: "
            f"{PEOPLE_CARDS[p].name} in the {ROOM_CARDS[r].name} with the "
            f"{WEAPON_CARDS[w].name}"
        )

        if correct:
            # This player just won!
            self.game_over = True
            self.winner = self.current_player
            self.log_action(
                f"{PEOPLE_CARDS[self.current_player].name} is correct and has won!"
            )

        else:
            self.false_accusers[accuser_idx] = 1
            self.log_action(
                f"{PEOPLE_CARDS[self.current_player].name} is wrong - they are out!"
            )
            if np.dot(self.active_players, self.false_accusers) == self.num_players:
                # All players make false accusations!
                self.log_action("There are no players left - GAME OVER")
                self.game_over = True
            else:
                self.next_move()

        return correct

    def is_false_accuser(self, player_idx: int) -> bool:
        return cast(bool, self.false_accusers[player_idx] == 1)

    def get_player_knowledge(self, player_idx: int) -> Dict:
        # TODO: decide if we make the order consistent?
        # want to make the order of these consistent, i.e.
        #  The current player should be the first entry
        #  followed by the next person in the order of player_idx.
        #
        # Which character you player_idx as should not really affect you
        # set the order to start with player_idx followed by the others
        # in order

        return {
            # Public knowledge:
            "step_kind": self.current_step_kind.value,  # 0-3
            "active players": np.concatenate(
                (self.active_players[player_idx:6], self.active_players[0:player_idx])
            ),  # 1x6
            "player locations": np.array(
                self.board.player_positions[player_idx:6]
                + self.board.player_positions[0:player_idx],
                dtype=np.int8,
            ),  # 1 x 6 each in range 0-204
            "suggestions": np.concatenate(
                (
                    self.suggestions[:, 0:21],  # pick 3 of 21
                    # Suggestor starting with current player
                    self.suggestions[:, 21 + player_idx : 27],
                    self.suggestions[:, 21 : 21 + player_idx],
                    # Can't Disprove
                    self.suggestions[:, 27 + player_idx : 33],
                    self.suggestions[:, 27 : 27 + player_idx],
                    # Can disprove
                    self.suggestions[:, 33 + player_idx : 39],
                    self.suggestions[:, 33 : 33 + player_idx],
                ),
                axis=1,
            ),  # 50x39 (1950)
            # Private knowledge:
            "card locations": np.concatenate(
                (
                    self.player_card_knowledge[player_idx, player_idx:6, :],
                    self.player_card_knowledge[player_idx, 0:player_idx, :],
                )
            ),  # 6x21 (126)
        }

    def get_player_knowledge_v1(self, player_idx: int) -> Dict:
        # TODO: decide if we make the order consistent?
        # want to make the order of these consistent, i.e.
        #  The current player should be the first entry
        #  followed by the next person in the order of player_idx.
        #
        # Which character you player_idx as should not really affect you
        # set the order to start with player_idx followed by the others
        # in order

        return {
            # Public knowledge:
            "step_kind": self.current_step_kind.value,  # 0-3
            "active players": np.concatenate(
                (self.active_players[player_idx:6], self.active_players[0:player_idx])
            ),  # 1x6
            "player room distances": self.get_player_room_distances(
                player_idx
            ),  # 6 x 18
            "suggestions": np.concatenate(
                (
                    self.suggestions[:, 0:21],  # pick 3 of 21
                    # Suggestor starting with current player
                    self.suggestions[:, 21 + player_idx : 27],
                    self.suggestions[:, 21 : 21 + player_idx],
                    # Can't Disprove
                    self.suggestions[:, 27 + player_idx : 33],
                    self.suggestions[:, 27 : 27 + player_idx],
                    # Can disprove
                    self.suggestions[:, 33 + player_idx : 39],
                    self.suggestions[:, 33 : 33 + player_idx],
                ),
                axis=1,
            ),  # 50x39 (1950)
            # Private knowledge:
            "card locations": np.concatenate(
                (
                    self.player_card_knowledge[player_idx, player_idx:6, :],
                    self.player_card_knowledge[player_idx, 0:player_idx, :],
                )
            ),  # 6x21 (126)
        }

    def get_player_room_distances(self, player_idx: int) -> np.ndarray:
        locations = (
            self.board.player_positions[player_idx:6]
            + self.board.player_positions[0:player_idx]
        )
        return cast(
            np.ndarray, self.board.distances[locations, :] / self.board.max_distance
        )

    def get_knowledge_score(self, player_idx: int) -> int:
        seen_cards = cast(
            np.ndarray, np.any(self.player_card_knowledge[player_idx], axis=0)
        )
        return cast(int, seen_cards.sum())

    def legal_actions(self) -> np.ndarray:
        # Action space:
        #  - move positions: 1x9
        if self.current_step_kind == StepKind.MOVE:
            legal_positions = self.board.legal_move_towards(
                player_idx=self.current_player
            )
        else:
            legal_positions = np.zeros(len(ROOM_CARDS))

        # Suggestions or accusations 9x6x6 (324)
        if self.current_step_kind == StepKind.SUGGESTION:
            suggestions_or_accusations = self._legal_suggestions()
        elif self.current_step_kind == StepKind.ACCUSATION:
            suggestions_or_accusations = self._legal_accusation()
        else:
            suggestions_or_accusations = np.zeros(
                (
                    len(PEOPLE_CARDS),
                    len(WEAPON_CARDS),
                    len(ROOM_CARDS),
                ),
                dtype=np.int8,
            )
        suggestions_or_accusations = suggestions_or_accusations.reshape((-1,))

        # Whether or not to make an accusation
        if self.current_step_kind == StepKind.ACCUSATION:
            make_accusation = 1
        else:
            make_accusation = 0
        make_accusation_array = np.array([make_accusation])

        # disprove suggestion
        #  choosing which card to show room, weapon or person 1x21
        if self.current_step_kind == StepKind.DISPROVE_SUGGESTION:
            legal_disprove = self._legal_disprove()
        else:
            legal_disprove = np.zeros(21)

        return np.concatenate(
            (
                legal_positions,
                suggestions_or_accusations,
                make_accusation_array,
                legal_disprove,
            )
        )

    def _legal_suggestions(self) -> np.ndarray:
        suggestion = np.zeros(
            (
                len(PEOPLE_CARDS),
                len(WEAPON_CARDS),
                len(ROOM_CARDS),
            ),
            dtype=np.int8,
        )

        # Can only make a suggestion if player is in a room.
        if not self.board.is_in_room(player_idx=self.current_player):
            return suggestion

        current_room_idx = self.board.which_room(self.current_player)

        # allow any combination of people and weapons in that room
        suggestion[:, :, current_room_idx].fill(1)

        return suggestion

    def _legal_disprove(self) -> np.ndarray:
        # Can disprove using the intersection of the cards in the
        #  suggestion and the cards the player holds.
        return cast(
            np.ndarray,
            self.suggestions[0][0:21]
            * self.player_card_knowledge[self.current_player][self.current_player],
        )

    def _legal_accusation(self) -> np.ndarray:
        # Don't need to be in the room to make the accusation and so can always
        # make any accusation.
        # Forcing it not to make any accusation
        #   which includes any cards that it has seen
        seen_cards = cast(
            np.ndarray, np.any(self.player_card_knowledge[self.current_player], axis=0)
        )

        if self.current_player != 0 and seen_cards.sum() < (21 - 3):
            # assume non 0 players are using random policy
            return np.zeros(
                (
                    len(PEOPLE_CARDS),
                    len(WEAPON_CARDS),
                    len(ROOM_CARDS),
                ),
                dtype=np.int8,
            )

        suggestion = np.ones(
            (
                len(PEOPLE_CARDS),
                len(WEAPON_CARDS),
                len(ROOM_CARDS),
            ),
            dtype=np.int8,
        )

        # Exclude any containing the seen people cards
        suggestion[seen_cards[0:6], :, :] = 0

        # Exclude seen weapons:
        suggestion[:, seen_cards[6:12], :] = 0

        # Exclude seen Rooms
        suggestion[:, :, seen_cards[12:]] = 0

        return suggestion

    def next_player(self) -> int:
        order_of_play = PLAYER_ORDERBY_PLAYER[self.current_player]
        for i in range(1, 6):
            player_idx = order_of_play[i]
            if (
                self.active_players[player_idx] == 1
                and self.false_accusers[player_idx] == 0
            ):
                return player_idx

        if self.false_accusers[self.current_player]:
            raise ValueError("No one left to play.")

        return self.current_player

    def render(self) -> str:
        if self.should_log_actions:
            events = "\n".join(self.what_just_happened)
            self.what_just_happened = []
            return events
        else:
            return ""
