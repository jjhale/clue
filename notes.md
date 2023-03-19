# 2023-03-19

Getting stuck in a draw with just a win/lose, because it is never winning.

The "winning" policies just don't do anything.

I tried truncating the game after 300 steps and imposing a -1 loss in that case.

I guess i want to incentize the agent to explore and learn - maybe you get rewards
for increasing your knowledge?

It also looks like it might not be following the rule around only making suggestions
for the room that you are in.

Fixed the legal suggestion bug


Tried to reduce failures by requiring that you are certain before making accusation,
which should force the game to end in a win.  With 6 random agents, I've seen games
take 4000 and 5000 steps to complete.

our observation space is 3316 elements, the action space is 551 wide.

The rewards seems to be all messed up :(

I'm seeing rewards of over 700 when just doing a single run to watch

Not making much progress.


## move maze solving out of policy
I should simplfiy the environment so that it just has to decide which room to
head towards - ie on move step we just look up disace to each room for all legal pos
and present closest to each to the move action only has 9 choices (one per room).

can represent all positions as the 6x9 vector of distances to rooms - down from
6x205 (1230) to 54.
