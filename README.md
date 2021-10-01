# Learn to build bridges.

A gym environment to build a bridge spanning a gap.

The environment is currently represented as h x w grid with ground at positions 0 and w-1, and a hole everywhere in between.

## Current Actions

Only a 1x2 bricks can be placed. The action space consists of integers [0,w), signifying where to put the left-most edge of the brick. Bricks are dropped vertically from above the grid and fall until any part of the brick encounters something below it.

## Usage
`environment_name = "gym_bridges.envs:Bridges-v0"`  
`env = gym.make(environment_name)`  
`env.setup(height, width)`

## Installation
Use the virtual environment from rome-wasnt-built-in-a-day.

Use reinstall.sh within the virtual environment both to install gym-bridges and to reinstall it after making any changes to the environment.

## Sample Bridge

  [][][][]
[][]    [][]  [][][][]
@@        @@@@@@    @@
