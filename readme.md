# Clue

This repo implements an environment for the board game Clue that can be used
to train a reinforcement learning agent to play the game.

It uses the OpenAI gym interface and petting zoo to represent clue as a
multi-agent reinforcement learning problem.

The key files are:
 - `clue/env/clue_environment_<version>.py` - The environment.
 - `clue/map.py` - The map of the board and utility functions for working with it.
 - `clue/map49.csv` - A csv representation of the board from the 1949 rules
 - `clue/state.py` - The logic of the game and the state representation.
 - `clue/train_agents_<version>.py` - different attemps to train agents to play the game.
 - `clue/interactive.py` - A script to play the game interactively against a trained agent.

# Current status

The environment captures the rules of the game ok, but the trained agents
don't play very well.

I've tried a few different approaches to training agents, but none of them
are much better than random.

I suspect that there is a problem with my training setup or model architecture.

# Map

A csv file represents the board layout.

Key:
 - `""` - part of a room
 - 0 - unaccessible part of the grid
 - 1 - a regular square on the grid
 - d[nsew]:(.*) a door into a room
    - the second char indicates which side of the cell the door way is
    - the text after the colon is the name of the room
 - s[a-z]{1,2} - the starting point of a player
 - [A-Z].* - The name of a room



# Precommit setup
Install the git hooks:
```shell
poetry run pre-commit install
```


# Interactive mode
You can check out what a policy would experience by running the interactive script:

```
poetry run python clue/interactive.py
```

It will create a log of the session in a file called `interactive.<timestamp>.log`
