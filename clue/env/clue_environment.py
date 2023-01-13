from typing import Optional

import numpy as np
from pettingzoo.utils.env import ActionType, AECEnv, ObsType


class ClueEnvironment(AECEnv):
    def __init__(self) -> None:
        pass

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
