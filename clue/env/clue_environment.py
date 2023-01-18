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

    def observe(self, agent: str) -> Optional[ObsType]:
        player_idx = int(agent)
        knowledge = self.clue.get_player_knowledge(player_idx)
        legal = self.clue.legal_actions()

        return {
            "observation": knowledge,
            "action_mask": legal,
        }

    def step(self, action: ActionType) -> tuple:
        # assume that the action is legal
        if self.clue.current_step_kind == StepKind.MOVE:
            self.clue.player_position_matrix[self.clue.current_player] = action[0:205]

            if self.clue.board.is_in_room(self.clue.current_player):
                # TODO HERE - probably should combine the board stuff with
                #  the card state - want a single source of truth for
                #  the location of a player
                ...
        return tuple("replace me")

    def render(self) -> None | np.ndarray | str | list:
        pass

    def seed(seed: Optional[int] = None) -> None:
        pass

    def close(self) -> None:
        pass
