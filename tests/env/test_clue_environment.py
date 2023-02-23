from typing import cast

from pettingzoo.test import api_test

from clue.env import clue_environment

env = clue_environment.ClueEnvironment(max_players=6)


def test_api_test() -> None:
    try:
        api_test(env, num_cycles=1000, verbose_progress=False)
    except Exception as e:
        with open("test_output.txt", "w") as f:
            f.write(cast(str, env.render()))
        raise e
