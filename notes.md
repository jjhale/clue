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

## Overnight run of 1024 model

got:  best_reward: -46.800000 ± 80.900927 in #64
using
        max_epoch=500,
        step_per_epoch=5000,
        step_per_collect=50,
        episode_per_test=10,
and 10_000 max steps in the clue env.



## move maze solving out of policy
I should simplfiy the environment so that it just has to decide which room to
head towards - ie on move step we just look up disace to each room for all legal pos
and present closest to each to the move action only has 9 choices (one per room).

can represent all positions as the 6x9 vector of distances to rooms - down from
6x205 (1230) to 54. Include the t-1 postions too, so that we can see where folks are
heading.


## Something wrong about the rewards still
The environment test is mad about rewards and cumulative rewards not doing something..
Should dig into it at some point to get the test working again.
 - [ ] fix the failing test

## Distance maps using dijstra
simple case since unweighted

I should treat the secret passages as just doors which are unconnected, but that
we wire up to their location.

- [ ] add sp1 and sp2 to the map
- [ ] treat as special doors.

hmmm, so what about how you need to terminal inside rooms? like we should probably
collapse doors in the same room into the same square - and add the secret passageways

Maybe just include distance via rooms as one metric, and distance without allowing
access thru doors as an input - e.g. double it there.

 - [ ] calc distances treating all doors of a room as the same square

 Should make distances be values between 0 and 1 (looks like max no room short cuts is
 31) - via rooms and not via rooms will have different scales

 Could calculate the distance to a room given the current throw - then that
 takes care of some of the decisions? i.e. for each legal position, get the min
 distances

 or could have min number of turns to get to room
 and max number of turns to get to the room.


2023-04-28

Running in interactive mode I was able to win in 398 total steps by focusing on just
exploring each room first, then the people and weapons. I ignored other peoples
suggestions. So setting a max of 500 steps seems like a reasonable upper limit for a
simple approach.

Steps: 398
Total suggestions: 71

2023-05-02
963k steps and it is getting a reward of 117 , best result was at round #45 and
currently on round #193
best_reward: 117.300000 ± 15.648131 in #45
