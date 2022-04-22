import gym
from gym import error, spaces, utils
from gym.utils import seeding

import dataclasses
import numpy as np
import random
import pygame
import gym_bridges.renderer.renderer_config as renderer_config
import sys
import time
import os
import itertools
from typing import Union, Optional
from enum import IntEnum
from collections import deque


@dataclasses.dataclass
class InitialBlock:
    index: int
    width: int
    height: int


class BridgesEnv(gym.Env):
    metadata = {"render.modes": ["human", "pygame"]}

    class StateType(IntEnum):
        _order_ = "EMPTY GROUND BRICK"
        EMPTY = 0
        GROUND = 1
        BRICK = 2

    # The max height of the ground of either end will be [1, width).
    # The max height of the env will be 1.5 times the width.
    # In the current implementation, a bridge will always be possible.
    def __init__(
        self,
        width,
        max_gap_count=1,
        force_standard_config=False,
        seed: Union[int, float, None] = None,
    ):
        super().__init__()

        assert (
            width > max_gap_count * 2
        ), "The max gap count must be less than half the width"
        self.shape = (int(1.5 * width), width)

        self._state: Optional[np.ndarray] = None
        self.nA = width
        self.action_space = spaces.Discrete(self.nA)
        self._max_gap_count = max_gap_count
        self._force_standard_config = force_standard_config

        # These 4 block-related members are set during reset().
        self._initial_blocks = None
        # List of positions at the surface of the starting block.
        self._starting_block_surface = None
        # Set of positions at the surface of the ending block.
        self._ending_block_surface = set()
        # List of sets of positions at the surface of each central block.
        self._central_block_surfaces = {}

        # TODO(lyric): Consider exposing brick size. We can revisit
        # when we have multiple brick sizes or if we find we're
        # hardcoding the brick size in multiple places. Currently
        # _step_helper in the unit tests does hardcode the brick width.
        self._brick = 2
        random.seed(seed)

        self._initialize_pygame = True

    def _check_row(self, action, index, brick_width):
        section = self._state[index, action : action + brick_width]
        return len(section) == brick_width and not section.any()

    def _place_brick(self, action, index, brick_width):
        self._state[index, action : action + brick_width] = BridgesEnv.StateType.BRICK

    def _is_bridge_complete(self):
        # Quick check.
        if not self._state.any(axis=0).all():
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
        queue = deque()
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

            if self._state[next_node] == BridgesEnv.StateType.BRICK:
                assert not in_central_block_surface
                queue.append(next_node)
                expanded.add(next_node)

            return False

        for starting_block in self._starting_block_surface:
            queue.append(starting_block)
            expanded.add(starting_block)

        while queue:
            node = queue.popleft()

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
        while (i < self.shape[0] - 1) and self._check_row(action, i + 1, self._brick):
            i += 1

        placed_successfully = i > -1 and i < self.shape[0] - 1

        if placed_successfully:
            self._place_brick(action, i, self._brick)

        done = self._is_bridge_complete()
        reward = 100 if done else -1 if placed_successfully else -5

        return self._state.copy(), reward, done, {}

    def reset(self, state=None, gap_count=None):
        if state is not None:
            assert state.shape == self.shape

            self._state = state.copy()

            # The indices for all empty cells in the lowest row
            gap_indices = np.argwhere(state[-1, :] == BridgesEnv.StateType.EMPTY)
            gap_indices = np.insert(gap_indices, 0, -1)

            # Compute the widths of all blocks
            widths = np.diff(gap_indices, append=self.shape[1]) - 1
            mask = widths > 0
            widths = widths[mask]

            # Compute the starting (leftmost) indices for all blocks
            indices = gap_indices[mask] + 1

            # Flipping the state upside down, then looking at the columns at
            # the rightmost ends of each of the blocks
            upside_down_spaces = (
                state[::-1, indices + widths - 1] == BridgesEnv.StateType.EMPTY
            )
            # Since the row index in `upside_down_spaces` increases with height,
            # this will return the lowest index at which an empty slot appears
            # at the end of each block, i.e. the height of the block
            heights = np.argmax(upside_down_spaces, axis=0)

            # Initialize initial_blocks based on the provided state.
            self._initial_blocks = [
                InitialBlock(*args) for args in zip(indices, widths, heights)
            ]

        else:
            self._state = np.zeros(shape=self.shape, dtype=int)

            if not gap_count:
                gap_count = random.randrange(1, self._max_gap_count + 1)

            if self._force_standard_config:
                self._initial_blocks = [
                    InitialBlock(index=0, width=1, height=1),
                    InitialBlock(index=self.shape[1] - 1, width=1, height=1),
                ]
            else:
                index = 0

                # We must ensure that all blocks and gaps have minimum
                # width of 1. This is done by sampling twice as many
                # starting indices for blocks without replacement and
                # retaining every other index.
                positions = random.sample(range(1, self.shape[1]), 2 * gap_count)
                # Ensure the first block starts at index 0 and the last
                # block can compute its width.
                positions = np.array(sorted([0] + positions + [self.shape[1]]))
                widths = np.diff(positions)[::2]
                indices = positions[::2]
                # We constrain the height of any given block to the
                # *width* of the environment. The environment's height
                # is set at 1.5*environment width to ensure a bridge can
                # always be built without hitting the top of the
                # environment. The height must be at least 1.
                heights = random.choices(range(1, self.shape[1]), k=len(indices))
                self._initial_blocks = [
                    InitialBlock(*args) for args in zip(indices, widths, heights)
                ]

            for initial_block in self._initial_blocks:
                index, width, height = (
                    initial_block.index,
                    initial_block.width,
                    initial_block.height,
                )
                self._state[
                    -height:, index : index + width
                ] = BridgesEnv.StateType.GROUND
        self._central_block_surfaces = []
        for initial_block in self._initial_blocks:
            index, width, depth = (
                initial_block.index,
                initial_block.width,
                self.shape[0] - initial_block.height,
            )
            self._central_block_surfaces.append(
                {(depth, index + x) for x in range(width)}
            )

        # It's slightly better for _starting_block_surface to be a
        # list because its usage is only to be iterated through.
        self._starting_block_surface = sorted(list(self._central_block_surfaces.pop(0)))
        self._ending_block_surface = self._central_block_surfaces.pop()

        return self._state.copy()

    def _draw_state(self) -> None:
        block_size = renderer_config.BLOCK_SIZE * 0.5

        x_window_coordinates = np.arange(
            renderer_config.WINDOW_WIDTH // 2 - block_size * self.shape[1] // 2,
            renderer_config.WINDOW_WIDTH // 2 + block_size * self.shape[1] // 2,
            block_size,
        )
        y_window_coordinates = np.arange(
            renderer_config.WINDOW_HEIGHT // 2 - block_size * self.shape[0] // 2,
            renderer_config.WINDOW_HEIGHT // 2 + block_size * self.shape[0] // 2,
            block_size,
        )

        for state_block, (y_window_coordinate, x_window_coordinate) in zip(
            self._state.flatten(),
            itertools.product(y_window_coordinates, x_window_coordinates),
        ):
            rect = pygame.Rect(
                x_window_coordinate,
                y_window_coordinate,
                block_size,
                block_size,
            )
            self._screen.blit(self._textures[state_block], rect)

    def _initialize_pygame_if_necessary(self) -> None:
        if self._initialize_pygame:
            pygame.init()

            self._screen = pygame.display.set_mode(
                (renderer_config.WINDOW_WIDTH, renderer_config.WINDOW_HEIGHT)
            )
            self._screen.fill(renderer_config.BLACK)

            # Load the textures with a path relative to the game source code.
            base_path = os.path.dirname(os.path.dirname(__file__))
            textures_path = os.path.join(base_path, "renderer", "assets")

            self._textures = {
                BridgesEnv.StateType.GROUND: pygame.image.load(
                    os.path.join(textures_path, "grass_block.png")
                ),
                BridgesEnv.StateType.BRICK: pygame.image.load(
                    os.path.join(textures_path, "mossy_stone_bricks.png")
                ),
                BridgesEnv.StateType.EMPTY: pygame.image.load(
                    os.path.join(textures_path, "light_blue_wool.png")
                ),
            }

            self._initialize_pygame = False

    def render(self, mode="human"):
        if mode == "human":
            mapping = {
                BridgesEnv.StateType.GROUND: "@@",
                BridgesEnv.StateType.BRICK: "[]",
                BridgesEnv.StateType.EMPTY: "  ",
            }
            flat_repr = tuple([mapping[x] for x in self._state.flatten()])
            print((("%s" * self.shape[1] + "\n") * self.shape[0]) % flat_repr)
            return
        if mode == "pygame":
            self._initialize_pygame_if_necessary()
            self._draw_state()

            # Drawing the state takes some time.
            time.sleep(1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()
            return
