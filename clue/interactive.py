import datetime
import logging

import numpy as np

from clue.env import clue_environment_v2
from clue.env.clue_environment_v2 import ClueEnvironment
from clue.state import CardState


def choose_action(legal: np.ndarray) -> int:
    chosen_action = np.zeros(len(legal))
    actions = ClueEnvironment.legal_action2human(legal)

    while not (chosen_action * legal).any():
        if actions["moves"].any():
            print("Pick a room to move towards:")
            print(" ".join([str(i) for i, v in enumerate(actions["moves"]) if v]))
            i = int(input("room number:"))
            chosen_action[i] = 1
        elif actions["make_accusation"]:
            # Need to check this before checking for suggestion / accusation
            i = int(input("Do you want to make an accusation? (1/0)"))

            if i == 0:
                chosen_action[9 + 324] = 1
            else:
                print("Choose a person, weapon and room:")
                print("People      Weapons     Room")
                print("0 1 2 3 4 5 0 1 2 3 4 5 0 1 2 3 4 5 6 7 8")
                checks = ["x" if x else " " for x in actions["sug_or_acc"]]
                print(" ".join(checks))
                p = int(input("Person number:"))
                w = int(input("Weapon number:"))
                r = int(input("Room number:"))
                idx = CardState.suggestion_one_hot(
                    person_idx=p, weapon_idx=w, room_idx=r
                ).argmax()
                chosen_action[9 + idx] = 1
        elif actions["sug_or_acc"].any():
            print("Choose a person, weapon and room:")
            print("People      Weapons     Room")
            print("0 1 2 3 4 5 0 1 2 3 4 5 0 1 2 3 4 5 6 7 8")
            checks = ["x" if x else " " for x in actions["sug_or_acc"]]
            print(" ".join(checks))
            p = int(input("Person number:"))
            w = int(input("Weapon number:"))
            r = int(input("Room number:"))
            idx = CardState.suggestion_one_hot(
                person_idx=p, weapon_idx=w, room_idx=r
            ).argmax()
            chosen_action[9 + idx] = 1
        elif actions["show_card"].any():
            print(
                "Which card do you want to show? [pwr]n (p: person, w: weapon, r: room)"
            )
            print("People      Weapons     Room")
            print("0 1 2 3 4 5 0 1 2 3 4 5 0 1 2 3 4 5 6 7 8")
            checks = ["x" if x else " " for x in actions["show_card"]]
            print(" ".join(checks))
            res = input(":")
            d = {"p": 0, "w": 6, "r": 12}
            i = d.get(res[0], 9999999) + int(res[1])
            chosen_action[9 + 324 + 1 + i] = 1

    return int(chosen_action.argmax())


if __name__ == "__main__":
    print("Starting interactive play.")
    logging.basicConfig(level=logging.DEBUG)

    env = clue_environment_v2.ClueEnvironment(log_actions=True)

    timestamp = datetime.datetime.now().strftime("%m-%d_%H%M")
    with open(f"clue/interactive.{timestamp}.log", "w", buffering=1) as f:

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc or env.clue.game_over:
                print(f"Game over in {env._elapsed_steps} steps.")

                break
            if obs is None:
                print("Obs was None - giving up")
                break

            mask = obs["action_mask"]
            if agent == "player_0":
                print(f"agent: {agent}")
                data = env.render()
                o2h = ClueEnvironment.observation2human(obs["observation"])
                print(data)
                print(o2h)
                print(f"iteration {env._elapsed_steps}")

                action = choose_action(mask)
            else:
                data = env.render()
                choices = np.flatnonzero(mask)

                action = np.random.choice(choices)
            f.write(f"agent: {agent}\n")
            f.write(f"iteration {env._elapsed_steps}\n")

            f.write(str(data))
            f.write(o2h)
            f.write(f"\n{agent} chose {action}\n")

            env.step(action)
