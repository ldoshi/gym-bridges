import pdb

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from random import choice
from random import randrange
from typing import NamedTuple

class InitialBlock(NamedTuple):
  index: int
  width: int
  height: int

class BridgesEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    pass

  # The max height of the ground of either end will be [1, width).
  # The max height of the env will be 1.5 times the width.
  # In the current implementation, a bridge will always be possible.
  def setup(self, width, max_gap_count=1, force_standard_config=False):
    assert width > max_gap_count * 2, "The max gap count must be less than half the width"
    self.shape = (int(1.5 * width), width)

    self.nA = width
    self.action_space = spaces.Discrete(self.nA)
    self.max_gap_count = max_gap_count
    self.force_standard_config = force_standard_config

    self.start = (self.shape[0]-1, 0)
    self.end = (self.shape[0]-1, self.shape[1]-1)
          
    self.reset()
 
    self.brick = 2
    
  def _check_row(self, action, index, brick_width):
    section = self.state[index][action:action+brick_width]
    return len(section) == brick_width and not section.any()

  def _place_brick(self, action, index, brick_width):
    self.state[index][action:action+brick_width] = 2

  def _is_bridge_complete(self):
    # Quick check.
    if not self.state.any(axis=0).all():
      return False
    
    # Run BFS from start to end.
    queue = []
    expanded = set()
    queue.append(self.start)
    expanded.add(self.start)

    while queue:
      node = queue.pop(0)
      if node == self.end:
        return True
      
      if node[1] - 1 >= 0:
        left = (node[0], node[1]-1)
        if left not in expanded and self.state[left[0]][left[1]]:
          queue.append(left)
          expanded.add(left)          

      if node[1] + 1 < self.shape[1]:
        right = (node[0], node[1]+1)
        if right not in expanded and self.state[right[0]][right[1]]:
          queue.append(right)
          expanded.add(right)

      if node[0] - 1 >= 0:
        up = (node[0] - 1, node[1])
        if up not in expanded and self.state[up[0]][up[1]]:
          queue.append(up)
          expanded.add(up)

      if node[0] + 1 < self.shape[0]:
        down = (node[0] + 1, node[1])
        if down not in expanded and self.state[down[0]][down[1]]:
          queue.append(down)
          expanded.add(down)
          
    return False
    
  def step(self, action):
    i = -1
    while i < self.shape[0] - 1:
      if not self._check_row(action, i + 1, self.brick):
        break
      i += 1

    placed_successfully = i > -1 and i < self.shape[0] - 1
      
    if placed_successfully:
      self._place_brick(action, i, self.brick)

    reward = -1 if placed_successfully else -5
    done = False

    if self._is_bridge_complete():
      reward = 100
      done = True
        
    return self.state.copy(),reward,done,{}
  
  def reset(self, state=None, gap_count=None):
    if state is not None:
      self.state = state.copy()
      return self.state.copy()

    self.state = np.zeros(shape=self.shape)

    if not gap_count:
      gap_count = randrange(1, self.max_gap_count + 1)
      
    initial_blocks = []

    index = self.start[1]
    for i in range(gap_count+1):
      if gap_count:
        # We must ensure we leave enough room for the remaining gaps
        # and blocks at the minimum width of 1.
        width = 1 + randrange(self.shape[1] - index - 2 * gap_count)
      else:
        # The final block must go until the end of the environment.
        width = self.shape[1] - index

      height = randrange(1, self.shape[1])
      initial_blocks.append(InitialBlock(index, width, height))

      gap_count -= 1
      if gap_count < 0:
        break

      # Build a random size gap that ensures we still have room for
      # the remaining gaps and blocks at the minimum width of 1.
      index = index + width + 1 + randrange(self.shape[1] - (index + width + 1) - (2 * gap_count))

    if self.force_standard_config:
      initial_blocks = [
        InitialBlock(index=self.start[1],width=1,height=1),
        InitialBlock(index=self.end[1],width=1,height=1)
      ]
      
    for initial_block in initial_blocks:
      for env_index in range(initial_block.index, initial_block.index + initial_block.width):
        for height_adjustment in range(initial_block.height):
          self.state[self.start[0]-height_adjustment][env_index] = 1

    return self.state.copy()
      
  def render(self, mode='human'):
    print((("%s"*self.shape[1]+"\n")*self.shape[0]) % tuple(["@@" if x == 1 else "[]" if x == 2 else "  " for x in self.state.flatten()]))
      
  def close(self):
    pass

#env = BridgesEnv()
#env.setup(9, 2)

#for _ in range(40):
#    observation, reward, done, _ = env.step(env.action_space.sample())
#    env.render()
#    if done:
#      print ("DONE")
#      break
#env.close()
