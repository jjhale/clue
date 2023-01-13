import os
from typing import Optional

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ActionType, AECEnv, ObsType

from clue.state import CardState

MAP_LOCATION = PARENT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../map49.csv"


class ClueEnvironment(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "clue_v0",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self) -> None:
        super().__init__()

        self.clue = CardState(MAP_LOCATION)
        self.possible_agents = list(range(CardState.MAX_PLAYERS))
        self.agents = self.clue.players

        self.action_spaces = {
            i: spaces.Discrete(205 + 324 + 21) for i in self.agents
        }  # (550 actions)
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Dict(
                        {
                            "who_am_i": spaces.Discete(6),  # 1x6
                            "step_kind": spaces.Discete(4),  # 1x4
                            "active players": spaces.Discete(6),  # 1x6
                            "player locations": spaces.Box(
                                0, 1, (6, 205), dtype=np.bool_
                            ),  # 6x205 (1230)
                            "suggestions": spaces.Box(
                                0, 1, (50, 39), dtype=np.bool_
                            ),  # history of last 50 suggestions
                            # Private knowledge:
                            "card locations": spaces.Box(
                                0, 1, (6, 21), dtype=np.bool_
                            ),  # 6x21 (126)
                        }
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
        pass

    def observe(self, agent: str) -> Optional[ObsType]:
        pass

    def step(self, action: ActionType) -> tuple:
        pass

    def render(self) -> None | np.ndarray | str | list:
        pass

    def seed(seed: Optional[int] = None) -> None:
        pass

    def close(self) -> None:
        pass
