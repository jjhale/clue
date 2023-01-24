import os
import random
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ActionType, AECEnv, ObsType

from clue.state import CardState, StepKind

MAP_LOCATION = PARENT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../map49.csv"


class ClueEnvironment(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "clue_v0",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, max_players: int = CardState.MAX_PLAYERS) -> None:
        super().__init__()
        if max_players < 3 or max_players > CardState.MAX_PLAYERS:
            raise ValueError("Max players needs to be between 3 and 6 inclusive")
        self.max_players = max_players
        self.clue = CardState(MAP_LOCATION, max_players=max_players)
        self.possible_agents = list(range(self.max_players))
        self.agents = self.clue.players
        self.num_suggestions = 0

        self.action_spaces = {
            i: spaces.Discrete(205 + 324 + 21) for i in self.agents
        }  # (550 actions)
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Dict(
                        {
                            "step_kind": spaces.Discrete(4),  # 1x4
                            "active players": spaces.Discrete(6),  # 1x6
                            "player locations": spaces.Box(
                                0, 1, (6, 205), dtype=np.int8
                            ),  # 6x205 (1230)
                            "suggestions": spaces.Box(
                                0, 1, (50, 39), dtype=np.int8
                            ),  # 50x39 (1950)
                            # Private knowledge:
                            "card locations": spaces.Box(
                                0, 1, (6, 21), dtype=np.int8
                            ),  # 6x21 (126)
                        }  # 1x3316 flattened
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(205 + 324 + 21,), dtype=np.int8
                    ),  # (550 actions)
                }
            )
            for i in self.agents
        }

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> None:
        random.seed(seed)

        self.clue.new_game(self.clue.pick_players())
        self.possible_agents = list(range(self.max_players))
        self.agents = self.clue.players
        self.num_suggestions = 0

    def observe(self, agent: str) -> Optional[ObsType]:
        player_idx = int(agent)
        knowledge = self.clue.get_player_knowledge(player_idx)
        legal = self.clue.legal_actions()

        return {
            "observation": knowledge,
            "action_mask": legal,
        }

    def step(self, action: ActionType) -> tuple:
        if self.clue.game_over:
            for agent in self.agents:
                if agent == self.clue.winner:
                    self.rewards[agent] = self.rewards[agent] + 1
                else:
                    self.rewards[agent] = self.rewards[agent] - 1
                self.terminations[agent] = True

        # assume that the action is legal
        if self.clue.current_step_kind == StepKind.MOVE:
            # The first 205 values represent the position output
            self.clue.board.set_location(self.clue.current_player, action[0:205])

            if self.clue.board.is_in_room(self.clue.current_player):
                self.clue.current_step_kind = StepKind.SUGGESTION
            else:
                self.clue.current_player = self.clue.next_player()
                self.clue.current_step_kind = StepKind.MOVE
                self.clue.current_die_roll = random.randint(1, 6)

        elif self.clue.current_step_kind == StepKind.SUGGESTION:
            self.num_suggestions = self.num_suggestions + 1
            # Decode the suggestion
            room, person, weapon = CardState.suggestion_one_hot_decode(
                action[205 : (205 + (9 * 6 * 6))]
            )

            # Who is the suggestor:
            suggestor_idx = self.clue.current_player

            # Go around the table until someone can disprove it
            cant_disprove, can_disprove_idx = self.clue.make_suggestion(
                room, person, weapon
            )

            # push down the old history and store the new history
            self.clue.suggestions[1:] = self.clue.suggestions[0:-1]
            self.clue.suggestions[0] = CardState.encode_suggestion_history(
                room,
                person,
                weapon,
                suggestor_idx=suggestor_idx,
                cant_disprove=cant_disprove,
                can_disprove_idx=can_disprove_idx,
            )

            if can_disprove_idx == -1:
                # No one was able to disprove us so go straight to making an accusation
                #  if desired
                self.clue.current_player = self.clue.current_player
                self.clue.current_step_kind = StepKind.ACCUSATION
            else:
                # Let the disprover choose which card to show
                self.clue.current_player = can_disprove_idx
                self.clue.current_step_kind = StepKind.DISPROVE_SUGGESTION

        elif self.clue.current_step_kind == StepKind.DISPROVE_SUGGESTION:
            # The card to show is in the last 21
            chosen_card_deck = action[-21:]
            suggestor_idx = self.clue.get_last_suggestor()
            self.clue.show_player_card(
                suggestor_idx=suggestor_idx,
                disprover_idx=self.clue.current_player,
                deck=chosen_card_deck,
            )

            # Next it is the suggestors turn or make an accusation (or not)
            self.clue.current_player = suggestor_idx
            self.clue.current_step_kind = StepKind.ACCUSATION

        elif self.clue.current_step_kind == StepKind.ACCUSATION:
            # Check for an accusation:
            accusation = action[205 : (205 + (9 * 6 * 6))]
            if not accusation.any():
                # Just move on to the next player
                self.clue.current_player = self.clue.next_player()
                self.clue.current_step_kind = StepKind.MOVE
                self.clue.current_die_roll = random.randint(1, 6)

            # Check the accusation
            correct = self.clue.make_accusation(accusation)
            if correct:
                self.rewards[self.clue.current_player] = (
                    self.rewards[self.clue.current_player] + 1
                )
                self.terminations[self.clue.current_player] = True
                # remaining players just now lost -end of game
                for player, terminated in self.terminations.items():
                    if not terminated:
                        self.rewards[player] = self.rewards[player] - 1
                        self.terminations[player] = True
            else:
                # We lost :(
                self.rewards[self.clue.current_player] = (
                    self.rewards[self.clue.current_player] + 1
                )

                #

        return tuple("replace me")

    def render(self) -> None | np.ndarray | str | list:
        pass

    def seed(seed: Optional[int] = None) -> None:
        pass

    def close(self) -> None:
        pass
