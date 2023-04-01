"""Uses Ray's RLLib to train agents to play Leduc Holdem.
Author: Rohan (https://github.com/Rohan138)
"""

import os
from typing import Any, Dict

import ray
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import AECEnv
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.tune.registry import register_env

from clue.env import clue_environment_v3
from clue.ray.masked_random import MaskedRandomPolicy

torch, nn = try_import_torch()


class TorchMaskedActions(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
        self,
        obs_space: Box,
        action_space: Discrete,
        num_outputs: int,
        model_config: Any,
        name: str,
        **kw: Dict[str, Any],
    ) -> None:
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        obs_len = obs_space.shape[0] - action_space.n

        orig_obs_space = Box(
            shape=(obs_len,), low=obs_space.low[:obs_len], high=obs_space.high[:obs_len]
        )
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict: Any, state: Any, seq_lens: Any) -> Any:
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model(
            {"obs": input_dict["obs"]["observation"]}
        )
        # turns probit action mask into logit action mask
        # inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        return action_logits + inf_mask, state

    def value_function(self) -> Any:
        return self.action_embed_model.value_function()


if __name__ == "__main__":
    ray.init(num_cpus=8)

    alg_name = "DQN"
    ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)
    # function that outputs the environment you wish to register.

    def env_creator() -> AECEnv:
        env = clue_environment_v3.env()
        return env

    env_name = "clue_environment_v3"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    def policy_mapping_fn(
        agent_id: str, episode: Any, worker: Any, **kwargs: Dict[str, Any]
    ) -> str:
        agent_idx = int(agent_id[-1])  # player_x -> x
        # agent_id = "player_[0-6]" -> policy depends on episode ID
        # This way, we make sure that both policies sometimes play player1
        # (start player) and sometimes player2 (player to move 2nd).
        return (
            "learning_policy"
            if episode.episode_id % 6 == agent_idx
            else "random_policy"
        )

    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        .training(
            train_batch_size=200,
            hiddens=[],
            dueling=False,
            model={"custom_model": "pa_model"},
        )
        .multi_agent(
            policies={
                # "player_0": (None, obs_space, act_space, {}),
                # "player_1": (None, obs_space, act_space, {}),
                # "player_2": (None, obs_space, act_space, {}),
                # "player_3": (None, obs_space, act_space, {}),
                # "player_4": (None, obs_space, act_space, {}),
                # "player_5": (None, obs_space, act_space, {}),
                # <- use default class & infer obs-/act-spaces from env.
                "learning_policy": PolicySpec(),
                "random_policy": PolicySpec(policy_class=MaskedRandomPolicy),
            },
            policy_mapping_fn=policy_mapping_fn,
            # Specify a (fixed) list (or set) of policy IDs that should be updated.
            policies_to_train=["learning_policy"],
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(
            log_level="ERROR"
        )  # TODO: change to ERROR to match pistonball example
        .framework(framework="torch")
        .exploration(
            exploration_config={
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 0.2,
                "final_epsilon": 0.0,
                "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
            }
        )
    )

    tune.run(
        alg_name,
        name="DQN",
        stop={"timesteps_total": 10000000},
        checkpoint_freq=10,
        config=config.to_dict(),
        # resume=True,
    )
