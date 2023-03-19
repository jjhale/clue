# type: ignore
import os
from functools import partial
from typing import Tuple

import gym
import torch
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager
from tianshou.utils.net.common import Net

from clue.env import clue_environment


def _get_env(render_mode=None):
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(clue_environment.env(render_mode=render_mode))


def _get_agents() -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )

    sizes = [128, 256]
    policies = []
    for size in sizes:
        net = Net(
            state_shape=observation_space["observation"].shape
            or observation_space["observation"].n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[size] * 4,
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )
        model_load_path = os.path.join("log", "rps", "dqn", f"policy_v{size}.pth")
        agent_learn.load_state_dict(torch.load(model_load_path))

        policies.append(agent_learn)

    # agents = [agent_learn, RandomPolicy(), RandomPolicy(),
    # RandomPolicy(), RandomPolicy(),RandomPolicy()]
    agents = [policies[0]] * 6  # + [policies[0]] * 2 + [RandomPolicy()] * 2
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


if __name__ == "__main__":
    env = DummyVectorEnv([partial(_get_env, render_mode="human")])
    policy, optim, agents = _get_agents()
    policy.eval()
    collector = Collector(policy, env, exploration_noise=False)
    result = collector.collect(n_episode=1, render=0.0001)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")

    print(rews)
