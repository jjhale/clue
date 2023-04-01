import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelWeights, TensorStructType, TensorType


class MaskedRandomPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        if self.config.get("ignore_action_bounds", False) and isinstance(
            self.action_space, Box
        ):
            self.action_space_for_sampling = Box(
                -float("inf"),
                float("inf"),
                shape=self.action_space.shape,
                dtype=self.action_space.dtype,
            )
        else:
            self.action_space_for_sampling = self.action_space

    @override(Policy)
    def init_view_requirements(self) -> None:
        super().init_view_requirements()
        # Disable for_training and action attributes for SampleBatch.INFOS column
        # since it can not be properly batched.
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False

    @override(Policy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        **kwargs: Any
    ) -> Union[List[TensorStructType], TensorStructType, List[TensorType], TensorType]:
        # Alternatively, a numpy array would work here as well.
        # e.g.: np.array([random.choice([0, 1])] * len(obs_batch))
        # Working call to _unpack_obs
        unpacked_obs = _unpack_obs(
            np.array(obs_batch, dtype=np.float32),
            self.observation_space.original_space,
            tensorlib=np,
        )

        # My environment returns observation of the form:
        # {agent_id: {'real_obs': real_obs, 'action_mask': action_mask}}
        action_masks = unpacked_obs["action_mask"]
        # breakpoint()
        # Convert action_masks to from np.float32 to np.int8
        action_masks = action_masks.astype(np.int8)
        # obs_batch_size = len(tree.flatten(obs_batch)[0])

        return (
            [
                self.action_space_for_sampling.sample(action_mask)
                for action_mask in action_masks
            ],
            [],
            {},
        )

    @override(Policy)
    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        """No learning."""
        return {}

    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions: TensorStructType,
        obs_batch: TensorStructType,
        state_batches: Any | None = None,
        prev_action_batch: Any | None = None,
        prev_reward_batch: Any | None = None,
    ) -> TensorType:
        return np.array([random.random()] * len(obs_batch))

    @override(Policy)
    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    @override(Policy)
    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    @override(Policy)
    def _get_dummy_batch_from_view_requirements(
        self, batch_size: int = 1
    ) -> SampleBatch:
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(
                    lambda s: s[None], self.observation_space.sample()
                ),
            }
        )
