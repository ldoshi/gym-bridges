import pdb

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class BridgesEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    pass
  
  def setup(self, height, width):
    self.shape = (height, width)

    self.nA = width
    self.action_space = spaces.Discrete(self.nA)

    self.start = (self.shape[0]-1, 0)
    self.end = (self.shape[0]-1, self.shape[1]-1)
        
    self.reset()
 
    self.brick = 2

  def _check_row(self, action, index, brick_width):
    section = self.state[index][action:action+brick_width]
    return len(section) == brick_width and not section.any()

  def _place_brick(self, action, index, brick_width):
    self.state[index][action:action+brick_width] = 1

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
      
  def reset(self, state=None):
    if state is None:
      self.state = np.zeros(shape=self.shape)
      self.state[self.start[0]][self.start[1]] = 1
      self.state[self.end[0]][self.end[1]] = 1
    else:
      self.state = state.copy()

    return self.state.copy()
      
  def render(self, mode='human'):
    print((("%s"*self.shape[1]+"\n")*self.shape[0]) % tuple(["X" if x else " " for x in self.state.flatten()]))
      
  def close(self):
    pass

#env = BridgesEnv()
#env.setup(6,10)
#env.reset()

#for _ in range(100):
#    observation, reward, done, _ = env.step(env.action_space.sample())
#    env.render()
#    if done:
#      print ("DONE")
#      break

#env.close()
