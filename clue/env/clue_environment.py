import os
import random
from typing import Dict, Optional

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
        self.num_suggestions = 0

        random.seed(seed)

        self.clue.new_game(self.clue.pick_players())

        # The standard AEC variables
        self.agents = self.clue.players
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos: Dict[int, Dict] = {i: {} for i in self.agents}

        # Selects the first agent (who is always Miss Scarlet)
        self.agent_selection = self.clue.current_player

    def observe(self, agent: str) -> Optional[ObsType]:
        player_idx = int(agent)
        knowledge = self.clue.get_player_knowledge(player_idx)
        legal = self.clue.legal_actions()

        return {
            "observation": knowledge,
            "action_mask": legal,
        }

    def step(self, action: ActionType) -> None:
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        assert self.agent_selection == self.clue.current_player

        # assume that the action is legal?
        if self.clue.current_step_kind == StepKind.MOVE:
            # The first 205 values represent the position output
            self.clue.move_player(action[0:205])

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

        elif self.clue.current_step_kind == StepKind.DISPROVE_SUGGESTION:
            # The card to show is in the last 21 of the action array
            self.clue.show_player_card(
                disprover_idx=self.agent_selection,
                deck=action[-21:],
            )

        elif self.clue.current_step_kind == StepKind.ACCUSATION:
            # Check for an accusation:
            accusation = action[205 : (205 + (9 * 6 * 6))]
            correct = self.clue.make_accusation(accusation)

            if correct and self.clue.game_over:
                self.rewards[self.agent_selection] = (
                    self.rewards[self.agent_selection] + 1
                )
                self.terminations[self.agent_selection] = True
                # remaining players just now lost - end of game
                for player, terminated in self.terminations.items():
                    if not terminated:
                        if not self.clue.is_false_accuser(player):
                            self.rewards[player] = self.rewards[player] - 1
                        self.terminations[player] = True
            else:
                # We lost, but still need to stick around to tell the other players
                # which cards we have.
                self.rewards[self.agent_selection] = (
                    self.rewards[self.agent_selection] - 1
                )

        self.agent_selection = self.clue.current_player

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def render(self) -> None | np.ndarray | str | list:
        return self.clue.render()

    def seed(seed: Optional[int] = None) -> None:
        pass

    def close(self) -> None:
        pass
