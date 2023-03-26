from typing import cast

from pettingzoo.test import api_test

from clue.env import clue_environment_v0, clue_environment_v1


def test_api_test_v0() -> None:
    env = clue_environment_v0.ClueEnvironment(max_players=6)
    try:
        api_test(env, num_cycles=1000, verbose_progress=False)
    except Exception as e:
        with open("test_output.txt", "w") as f:
            f.write(cast(str, env.render()))
        raise e


def test_api_test_v1() -> None:
    env = clue_environment_v1.ClueEnvironment(max_players=6)

    try:
        api_test(env, num_cycles=1000, verbose_progress=False)
    except Exception as e:
        with open("test_output.txt", "w") as f:
            f.write(cast(str, env.render()))
        raise e
