import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

from clue.state import CardState, StepKind

MAP_LOCATION = PARENT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../map49.csv"
"""
This one should have the board maintain distances from every board position to the
closest door of each room. Then we can use that as a proxy for the position of the
agents and just let them choose to move towards a given room - rather than selecting
a specific board square to land on.

No partial points - winner takes all at the end
"""


def env(render_mode: Optional[str] = None) -> AECEnv:
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = ClueEnvironment(render_mode=internal_render_mode, max_episode_steps=500)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)

    return env


class ClueEnvironment(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "clue_v1",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self,
        max_players: int = CardState.MAX_PLAYERS,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 0,
        log_actions: bool = False,
    ) -> None:
        super().__init__()
        if max_players < 3 or max_players > CardState.MAX_PLAYERS:
            raise ValueError("Max players needs to be between 3 and 6 inclusive")
        self.max_players = max_players
        self.clue = CardState(
            MAP_LOCATION,
            max_players=max_players,
            log_actions=(render_mode == "human") or log_actions,
        )

        self.agent_map = {f"player_{i}": i for i in range(self.max_players)}
        self.possible_agents = list(self.agent_map.keys())

        self.agents = [self.possible_agents[i] for i in self.clue.players]
        self.num_suggestions = 0

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        # Actions
        # 9 - move towards rooms
        # 324 - accusations or suggestion ( 9 rooms x 6 people x 6 weapons)
        # 1 - choose to make an accusation or not
        # 21 - which card to show accusor to disprove it
        self.action_spaces = {
            i: spaces.Discrete(9 + 324 + 1 + 21) for i in self.possible_agents
        }  # (9+324+1+21 = 355 actions)

        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Dict(
                        {
                            "step_kind": spaces.Discrete(4),  # 1x4
                            "active players": spaces.MultiBinary(6),  # 1x6
                            "player room distances": spaces.Box(
                                0, 1, (6, 18)
                            ),  # 18*6 (108)
                            "suggestions": spaces.MultiBinary([50, 39]),  # 50x39 (1950)
                            # Private knowledge:
                            "card locations": spaces.MultiBinary([6, 21]),  # 6x21 (126)
                        }  # 2194 flattened
                    ),
                    "action_mask": spaces.MultiBinary(
                        9 + 324 + 1 + 21
                    ),  # (355 actions)
                }
            )
            for i in self.possible_agents
        }

        self.flat_obs_space = {
            i: spaces.Dict(
                {
                    "observation": spaces.flatten_space(
                        self.observation_spaces[i]["observation"]
                    ),
                    "action_mask": self.observation_spaces[i]["action_mask"],
                }
            )
            for i in self.possible_agents
        }

        self.render_mode = render_mode
        self.reset()

    @staticmethod
    def legal_action2human(legal: np.ndarray) -> Dict[str, np.ndarray]:
        # 9 - move towards rooms
        # 324 - accusations or suggestion ( 6 people x 6 weapons x 9 rooms )
        # 1 - choose to make an accusation or not
        # 21 - which card to show accusor to disprove it
        result = {
            "moves": legal[0:9],
            "sug_or_acc": np.zeros(21),
            "make_accusation": legal[9 + 324 : 9 + 324 + 1],
            "show_card": legal[9 + 324 + 1 : 324 + 9 + 1 + 21],
        }

        # Collapse the suggestions into a deck vector.
        if legal[9 : 324 + 9].any():
            suggestion_legal = legal[9 : 324 + 9].reshape((6, 6, 9))
            people = suggestion_legal.any(axis=(1, 2))
            weapons = suggestion_legal.any(axis=(0, 2))
            rooms = suggestion_legal.any(axis=(0, 1))

            result["sug_or_acc"] = np.concatenate([people, weapons, rooms])

        return result

    @staticmethod
    def observation2human(obs: np.ndarray) -> str:
        # probably sorted alphabetically by key?!?
        # ['active players', 'card locations',
        # 'player room distances', 'step_kind', 'suggestions']

        active_players = obs[0:6]
        card_locations = obs[6 : 6 + 126].reshape((6, 21))  # 6x21
        distances = obs[6 + 126 : 6 + 126 + 108].reshape((6, 18))  # 6x18
        step_kind = obs[6 + 126 + 108 : 6 + 126 + 108 + 4]
        suggestions = obs[6 + 126 + 108 + 4 : 6 + 126 + 108 + 4 + 1950].reshape(
            (50, 39)
        )  # 50x39

        summary: List[str] = []
        summary.append(f"Total players = {sum(active_players)}")

        # What cards am I holding?
        #     People      Weapons     Room
        #     0 1 2 3 4 5 0 1 2 3 4 5 0 1 2 3 4 5 6 7 8
        # me
        #  1
        #  2
        #  3
        #  4
        #  5
        deck_lut = list(range(6)) + list(range(6)) + list(range(9))
        summary.append("Your Cards:")
        summary.append("    People      Weapons     Room")
        summary.append("    0 1 2 3 4 5 0 1 2 3 4 5 0 1 2 3 4 5 6 7 8 ")
        for i in range(6):
            checks = [
                str(deck_lut[j]) if x else " " for j, x in enumerate(card_locations[i])
            ]
            summary.append(f"{i}   " + " ".join(checks))

        summary.append("Distance to rooms (direct:via rooms)")
        #     0         1 ....
        summary.append("    " + "".join([f"{r}          " for r in range(9)]))
        # 0   0.03:0.02
        for i in range(6):
            ds = [f"{distances[i,j]:.2f}:{distances[i, j+9]:.2f}  " for j in range(9)]
            summary.append(f"{i}   " + "".join(ds))

        summary.append("")
        summary.append(f"Step kind: {StepKind(step_kind.argmax()).name}")

        # Suggestion history
        # s[0:21] - DECK vector pick 3,  n=21
        # s[21:27]- the suggestor pick 1,  n=6
        # s[27:33] can't disprove  pick all that apply, n=6
        # s[33:39]- can disprove - pick 0 or 1, n=6
        summary.append("")
        summary.append("Suggestions:")
        #     |People      Weapons     Room              | Can't disp  |
        # By  |0 1 2 3 4 5 0 1 2 3 4 5 0 1 2 3 4 5 6 7 8 | 0 1 2 3 4 5 | Disprover
        summary.append("    |People      Weapons     Room              | Can't disp  |")
        summary.append(
            "By  |0 1 2 3 4 5 0 1 2 3 4 5 0 1 2 3 4 5 6 7 8 | 0 1 2 3 4 5 | Disprover"
        )

        for i in range(50):
            if not suggestions[i, :].any():
                continue
            by = suggestions[i, 21:27].argmax()
            deck = [
                str(deck_lut[j]) if x else " "
                for j, x in enumerate(suggestions[i, 0:21])
            ]
            cant = ["x" if x else " " for x in suggestions[i, 27:33]]
            if suggestions[i, 33:39].any():
                disprover = f"disproved by {suggestions[i, 33:39].argmax()}"
            else:
                disprover = ""
            summary.append(
                f" {by}  |" + " ".join(deck) + f" | {' '.join(cant)} | {disprover}"
            )

        result = "\n".join(summary)
        return result

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> None:
        self.num_suggestions = 0
        self._elapsed_steps = 0

        random.seed(seed)

        self.clue.new_game(self.clue.pick_players())

        # The standard AEC variables
        self.agents = [self.possible_agents[i] for i in self.clue.players]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos: Dict[str, Dict] = {i: {} for i in self.agents}

        # TODO: HERE Need to make the agent names not be ints coz it fails assert
        # current_player when current player is 0
        # Selects the first agent (who is always Miss Scarlet)
        self.agent_selection = self.possible_agents[self.clue.current_player]

    def observe(self, agent: str) -> Optional[ObsType]:
        player_idx = self.agent_map[agent]
        knowledge = self.clue.get_player_knowledge_v1(player_idx)
        legal = self.clue.legal_actions()

        flat_knowledge = spaces.flatten(
            self.observation_spaces[agent]["observation"], knowledge
        )  # here copy what FlattenSpaceWrapper does.

        return {
            "observation": flat_knowledge,
            "action_mask": legal,
        }

    def step(self, action: ActionType) -> None:
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        # Actions
        # 9 - rooms to move towards
        # 324 - accusations or suggestion ( 9 rooms x 6people x 6 weapons)
        # 1 - choose not to make an accusation
        # 21 - which card to show accusor to disprove it

        assert self.agent_selection == self.possible_agents[self.clue.current_player]

        # # the agent which stepped last had its _cumulative_rewards accounted for
        # # (because it was returned by last()), so the _cumulative_rewards for this
        # # agent should start again at 0
        self._cumulative_rewards[self.agent_selection] = 0
        # for k in self._cumulative_rewards:
        #     self._cumulative_rewards[k] = 0
        self._clear_rewards()

        # assume that the action is legal?
        if self.clue.current_step_kind == StepKind.MOVE:
            if action >= self.clue.NUM_ROOMS:
                with open("test_output_crashed.txt", "w") as f:
                    f.write(cast(str, self.render()))
            # The first 9 values represent the position output
            self.clue.move_player(action, towards_room=True)

        elif self.clue.current_step_kind == StepKind.SUGGESTION:
            self.num_suggestions = self.num_suggestions + 1
            # Decode the suggestion
            (
                person,
                weapon,
                room,
            ) = CardState.suggestion_one_hot_decode(action - 9)

            # Who is the suggestor:
            suggestor_idx = self.clue.current_player

            # Go around the table until someone can disprove it
            cant_disprove, can_disprove_idx = self.clue.make_suggestion(
                room_idx=room, person_idx=person, weapon_idx=weapon
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
            assert self.clue.get_last_suggestor() == suggestor_idx

        elif self.clue.current_step_kind == StepKind.DISPROVE_SUGGESTION:
            suggestor_idx = self.clue.get_last_suggestor()
            # The card to show is in the last 21 of the action array
            self.clue.show_player_card(
                disprover_idx=self.agent_map[self.agent_selection],
                deck_idx=action - (9 + 324 + 1),
            )

        elif self.clue.current_step_kind == StepKind.ACCUSATION:
            # Check for an accusation:
            accusation = action - 9

            correct = self.clue.make_accusation(accusation)

            if correct is True:
                self.clue.log_action(
                    f"Before Setting rewards on win\n "
                    f"- cum_rew {self._cumulative_rewards.values()}, \n"
                    f" - rewards {self.rewards.values()} ."
                )

                self.rewards[self.agent_selection] = (
                    self.rewards[self.agent_selection] + 100
                )
                self.terminations[self.agent_selection] = True
                # remaining players just now lost - end of game
                for player, terminated in self.terminations.items():
                    if not terminated:
                        self.rewards[player] = self.rewards[player] - 100
                        self.terminations[player] = True
            elif correct is False:
                # We lost, but still need to stick around to tell the other players
                # which cards we have.
                self.clue.log_action(
                    f"before Setting rewards on lose\n - cum_rew "
                    f" {self._cumulative_rewards.values()}, \n"
                    f" - rewards {self.rewards.values()} ."
                )

                # if everyone else is terminated then we need to terminate too
                if self.clue.game_over:
                    self.terminations[self.agent_selection] = True
                    for player, terminated in self.terminations.items():
                        self.terminations[player] = True
                        self.rewards[self.agent_selection] = (
                            self.rewards[self.agent_selection] - 100
                        )
            # Else correct is None and they decided against making accusation

        self.agent_selection = self.possible_agents[self.clue.current_player]

        self._elapsed_steps = self._elapsed_steps + 1

        if self._max_episode_steps and self._elapsed_steps >= self._max_episode_steps:
            self.clue.log_action(
                f"About to truncate \n - cum_rew {self._cumulative_rewards.values()},"
                f" \n - rewards {self.rewards.values()} ."
            )
            for player in self.truncations:
                self.truncations[player] = True
                self.rewards[player] = self.rewards[player] - 100

        self._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def render(self) -> Optional[Union[np.ndarray, str, list]]:
        render_response = self.clue.render()
        print(render_response)
        return render_response

    def seed(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)

    def close(self) -> None:
        pass

    def last(
        self, observe: bool = True
    ) -> Tuple[Optional[ObsType], float, bool, bool, Dict[str, Any]]:
        return super().last(observe)  # type: ignore

    def observation_space(self, agent: AgentID) -> spaces.Space:
        """Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        """
        return self.flat_obs_space[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]
