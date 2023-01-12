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
