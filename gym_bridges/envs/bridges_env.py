import pdb

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from random import choice
from random import randrange
from typing import NamedTuple
from enum import IntEnum


class InitialBlock(NamedTuple):
    index: int
    width: int
    height: int


class StateType(IntEnum):
    _order_ = "EMPTY GROUND BRICK"
    EMPTY = 0
    GROUND = 1
    BRICK = 2


class BridgesEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        pass

    # The max height of the ground of either end will be [1, width).
    # The max height of the env will be 1.5 times the width.
    # In the current implementation, a bridge will always be possible.
    def setup(self, width, max_gap_count=1, force_standard_config=False):
        assert (
            width > max_gap_count * 2
        ), "The max gap count must be less than half the width"
        self.shape = (int(1.5 * width), width)

        self.nA = width
        self.action_space = spaces.Discrete(self.nA)
        self.max_gap_count = max_gap_count
        self.force_standard_config = force_standard_config

        # These 4 block-related members are set during reset().
        self._initial_blocks = None
        # Set of positions at the surface of the starting block.
        self._starting_block_surface = None
        # Set of positions at the surface of the ending block.
        self._ending_block_surface = set()
        # List of sets of positions at the surface of each central block.
        self._central_block_surfaces = {}

        self.reset()

        self.brick = 2

    def _check_row(self, action, index, brick_width):
        section = self.state[index][action : action + brick_width]
        return len(section) == brick_width and not section.any()

    def _place_brick(self, action, index, brick_width):
        self.state[index][action : action + brick_width] = StateType.BRICK

    def _is_bridge_complete(self):
        # Quick check.
        if not self.state.any(axis=0).all():
            return False

        # Run BFS from start to end.

        # 1. All nodes at the starting block surface are added to the
        # queue as potential starting points for the path. This
        # represents being able to traverse any amount of the starting
        # block surface.
        
        # 2. Any node located in the ending block surface represents
        # reaching the goal. This represents being able to traverse
        # any amount of the ending block surface.
        #
        # 3. The first time any node in a central block surface is
        # encounted, all nodes from that particular central block
        # surface are added to the queue. This represents being able
        # to traverse any amount of that central blocks surface before
        # continuing to the next bridge component.
        queue = []
        expanded = set()

        def _bfs_helper(next_node):
            if next_node in expanded:
                return False

            if next_node in self._ending_block_surface:
                return True

            in_central_block_surface = False
            for central_block_surface in self._central_block_surfaces:
                if next_node in central_block_surface:
                    in_central_block_surface = True

                    for central_block in central_block_surface:
                        queue.append(central_block)
                        expanded.add(central_block)
                    # By construction central block surfaces are disconnected so
                    # we can break after finding a match.
                    break

            if self.state[next_node[0]][next_node[1]] == StateType.BRICK:
                assert not in_central_block_surface
                queue.append(next_node)
                expanded.add(next_node)

            return False

        for starting_block in self._starting_block_surface:
            queue.append(starting_block)
            expanded.add(starting_block)

        while queue:
            node = queue.pop(0)

            if node[1] - 1 >= 0:
                left = (node[0], node[1] - 1)
                if _bfs_helper(left):
                    return True

            if node[1] + 1 < self.shape[1]:
                right = (node[0], node[1] + 1)
                if _bfs_helper(right):
                    return True

            if node[0] - 1 >= 0:
                up = (node[0] - 1, node[1])
                if _bfs_helper(up):
                    return True

            if node[0] + 1 < self.shape[0]:
                down = (node[0] + 1, node[1])
                if _bfs_helper(down):
                    return True

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

        return self.state.copy(), reward, done, {}

    def reset(self, state=None, gap_count=None):
        if state is not None:
            assert state.shape == self.shape

            self.state = state.copy()

            # Initialize initial_blocks based on the provided state.
            self._initial_blocks = []

            state_base_height = self.shape[0]
            state_width = self.shape[1]

            index = 0
            width = 0
            # The loop goes one extra iteration to capture the ending block.
            for x in range(state_width + 1):
                if (
                    x < state_width
                    and state[state_base_height - 1][x] == StateType.GROUND
                ):
                    width += 1
                else:
                    # End of block. Compute height and save block.
                    height = 0
                    while (
                        state[state_base_height - height - 1][x - 1] == StateType.GROUND
                    ):
                        height += 1
                    self._initial_blocks.append(InitialBlock(index, width, height))
                    index = x + 1
                    width = 0

        else:
            self.state = np.zeros(shape=self.shape)

            if not gap_count:
                gap_count = randrange(1, self.max_gap_count + 1)

            self._initial_blocks = []

            index = 0
            for i in range(gap_count + 1):
                if gap_count:
                    # We must ensure we leave enough room for the remaining gaps
                    # and blocks at the minimum width of 1.
                    width = 1 + randrange(self.shape[1] - index - 2 * gap_count)
                else:
                    # The final block must go until the end of the environment.
                    width = self.shape[1] - index

                height = randrange(1, self.shape[1])
                self._initial_blocks.append(InitialBlock(index, width, height))

                gap_count -= 1
                if gap_count < 0:
                    break

                # Build a random size gap that ensures we still have room for
                # the remaining gaps and blocks at the minimum width of 1.
                index = (
                    index
                    + width
                    + 1
                    + randrange(self.shape[1] - (index + width + 1) - (2 * gap_count))
                )

            if self.force_standard_config:
                self._initial_blocks = [
                    InitialBlock(index=0, width=1, height=1),
                    InitialBlock(index=self.shape[1] - 1, width=1, height=1),
                ]

            for initial_block in self._initial_blocks:
                for env_index in range(
                    initial_block.index, initial_block.index + initial_block.width
                ):
                    for height_adjustment in range(1, initial_block.height + 1):
                        self.state[self.shape[0] - height_adjustment][
                            env_index
                        ] = StateType.GROUND

        base_height = self.shape[0]

        self._starting_block_surface = [
            (
                base_height - self._initial_blocks[0].height,
                self._initial_blocks[0].index + x,
            )
            for x in range(self._initial_blocks[0].width)
        ]
        self._ending_block_surface = set(
            [
                (
                    base_height - self._initial_blocks[-1].height,
                    self._initial_blocks[-1].index + x,
                )
                for x in range(self._initial_blocks[-1].width)
            ]
        )

        self._central_block_surfaces = []
        for i in range(1, len(self._initial_blocks) - 1):
            self._central_block_surfaces.append(
                [
                    (
                        base_height - self._initial_blocks[i].height,
                        self._initial_blocks[i].index + x,
                    )
                    for x in range(self._initial_blocks[i].width)
                ]
            )

        return self.state.copy()

    def render(self, mode="human"):
        print(
            (("%s" * self.shape[1] + "\n") * self.shape[0])
            % tuple(
                [
                    "@@"
                    if x == StateType.GROUND
                    else "[]"
                    if x == StateType.BRICK
                    else "  "
                    for x in self.state.flatten()
                ]
            )
        )

    def close(self):
        pass
