# 2023-03-19

Getting stuck in a draw with just a win/lose, because it is never winning.

The "winning" policies just don't do anything.

I tried truncating the game after 300 steps and imposing a -1 loss in that case.

I guess i want to incentize the agent to explore and learn - maybe you get rewards
for increasing your knowledge?

It also looks like it might not be following the rule around only making suggestions
for the room that you are in.
