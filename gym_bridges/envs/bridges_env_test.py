import numpy as np
import pytest
import unittest

from collections import defaultdict
from parameterized import parameterized

from bridges_env import BridgesEnv


def _check_block(state, index, width_check_value):
    """Verifies that the block start at index is grounded in the environment and
    maintains a level height until the next gap. Returns the index following this
    block on success. Gaps are checked as blocks of height 0."""
    env_height = state.shape[0] - 1
    block_height = 0

    while state[env_height - block_height][index] == 1:
        block_height += 1

    i = index + 1
    while i < state.shape[1] and state[env_height][i] == width_check_value:
        i_height = 0
        while state[env_height - i_height][i] == 1:
            i_height += 1

        if i_height != block_height:
            return False

        i += 1

    if state[:, index:i].sum() != block_height * (i - index):
        return False

    return i


def _state_builder(shape, heights):
    state = np.zeros(shape)
    state_height = state.shape[0]
    for i, height in enumerate(heights):
        state[state_height - height : state_height, i] = 1
    return state


def _step_helper(action, env, width, mirror=False):
    if mirror:
        action = width - 2 - action

    env.step(action)


def _reset_helper(heights, env, mirror=False):
    if mirror:
        heights.reverse()

    env.reset(state=_state_builder(env.shape, heights))


class TestBridgesEnv(unittest.TestCase):
    def setUp(self):
        self.env = BridgesEnv()

    @parameterized.expand([(3,), (9,)])
    def test_force_standard_config(self, width):
        """The standard configuration only has a value of 1 at the bottom left
        and right locations of the env."""
        self.env.setup(width=width, force_standard_config=True)
        height = self.env.shape[0] - 1
        self.assertEqual(self.env.state[height][0], 1)
        self.assertEqual(self.env.state[height][width - 1], 1)
        self.assertEqual(self.env.state.sum(), 2)

    @parameterized.expand([(3,), (9,)])
    def test_gap_count(self, width):
        """Verifies the requested number of gaps exist. This test also verifies
        that the env is well-formed in terms of having initial blocks of ground
        (1 values) properly grounded to the bottom of the env and level between
        gaps."""
        self.env.setup(width=width)
        for gap_count in range(1, ((width + 1) // 2)):
            self.env.reset(gap_count=gap_count)
            # Verify gap columns are completely empty.
            index = 0
            width_check_value = 1
            counter = 0
            while index < width:
                index = _check_block(self.env.state, index, width_check_value)
                self.assertTrue(index)

                width_check_value = (width_check_value + 1) % 2
                # Misses the first value so this should equal the gap count.
                counter += width_check_value

            self.assertEqual(counter, gap_count)

    @parameterized.expand([(3,), (9,)])
    def test_max_gap_count(self, width):
        """Verifies the gap count varies between 1 and the max gap count."""
        for gap_count in range(1, ((width + 1) // 2)):
            self.env.setup(width=width, max_gap_count=gap_count)
            height = self.env.shape[0] - 1
            counts = defaultdict(lambda: 0)
            for _ in range(100):
                self.env.reset()
                counts[
                    int(
                        np.sum(
                            [
                                self.env.state[height][i]
                                if self.env.state[height][i]
                                - self.env.state[height][i + 1]
                                else 0
                                for i in range(self.env.state.shape[1] - 1)
                            ]
                        )
                    )
                ] += 1

            self.assertEqual(len(counts), gap_count)

    @parameterized.expand([(False,), (True,)])
    def test_is_bridge_complete_simple(self, mirror):
        """Verifies bridge completion detection in a very simple set-up."""
        # Test case:
        #
        # [][]
        # [][][][]
        # @@    @@

        heights = [1, 0, 0, 1]
        width = len(heights)
        self.env.setup(width=width)

        _reset_helper(heights, self.env, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        self.env.step(0)
        self.assertFalse(self.env._is_bridge_complete())
        self.env.step(0)
        self.assertFalse(self.env._is_bridge_complete())
        self.env.step(2)
        self.assertTrue(self.env._is_bridge_complete())

    @parameterized.expand([(False,), (True,)])
    def test_is_bridge_complete_side_completion(self, mirror):
        """Verifies bridge completion detection when the bridge is completed via
        a side connection to an endpoint instead being placed on top of one of
        the endpoints."""
        # Test case:
        #
        # @@[][]
        # @@  @@

        heights = [2, 0, 1]
        width = len(heights)
        self.env.setup(width=width)

        # Side completion to starting point.
        _reset_helper(heights, self.env, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(1, self.env, width, mirror)
        self.assertTrue(self.env._is_bridge_complete())

    @parameterized.expand([(False,), (True,)])
    def test_is_bridge_complete_wide_endpoint(self, mirror):
        """Verifies bridge completion detection when the bridge completes by
        touching any section of a wide endpoint."""
        # Test cases:
        #
        #   [][][][]
        # @@@@@@  @@
        #
        # [][][][]
        # [][]  [][]
        # @@@@@@  @@

        heights = [1, 1, 1, 0, 1]
        width = len(heights)
        self.env.setup(width=width)

        # Test two completions with a wide starting point.
        _reset_helper(heights, self.env, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(3, self.env, width, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(1, self.env, width, mirror)
        self.assertTrue(self.env._is_bridge_complete())

        _reset_helper(heights, self.env, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(3, self.env, width, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(0, self.env, width, mirror)
        _step_helper(2, self.env, width, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(0, self.env, width, mirror)
        self.assertTrue(self.env._is_bridge_complete())

    @parameterized.expand([(False,), (True,)])
    def test_is_bridge_complete_cliff_side_insufficient(self, mirror):
        """Verifies bridge completion detection when the bridge touches
        the side of a cliff on one end, but not the top."""
        # Test case:
        #
        # [][]@@
        # [][]@@
        # @@  @@

        heights = [1, 0, 3]
        width = len(heights)
        self.env.setup(width=width)

        _reset_helper(heights, self.env, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(0, self.env, width, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(0, self.env, width, mirror)
        self.assertTrue(self.env._is_bridge_complete())

    @parameterized.expand([(False,), (True,)])
    def test_is_bridge_complete_center_ground_surface(self, mirror):
        """Verifies bridge completion detection when the bridge touches a
        central ground component that's not part of the left or the right.
        Crossing over central ground is permitted as part of connecting the
        left and right. This test verifies the case where bridges adjoin
        the center ground at surface height."""
        # Test case:
        #
        # [][]@@@@[][]
        # @@  @@@@  @@

        heights = [1, 0, 2, 2, 0, 1]
        width = len(heights)
        self.env.setup(width=width)

        _reset_helper(heights, self.env, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(0, self.env, width, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(4, self.env, width, mirror)
        self.assertTrue(self.env._is_bridge_complete())

    @parameterized.expand([(False,), (True,)])
    def test_is_bridge_complete_center_ground_build_up(self, mirror):
        """Verifies bridge completion detection when the bridge touches a
        central ground component that's not part of the left or the right.
        Crossing over central ground is permitted as part of connecting the
        left and right. This test verifies the case where bridges connect
        to the center ground by building up from it."""
        # Test case:
        #
        # [][][][]  [][][][]
        # @@    @@@@@@    @@

        heights = [1, 0, 0, 1, 1, 1, 0, 0, 1]
        width = len(heights)
        self.env.setup(width=width)

        _reset_helper(heights, self.env, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(0, self.env, width, mirror)
        _step_helper(7, self.env, width, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(2, self.env, width, mirror)
        _step_helper(5, self.env, width, mirror)
        self.assertTrue(self.env._is_bridge_complete())

    @parameterized.expand([(False,), (True,)])
    def test_is_bridge_complete_center_ground_surface_and_build_up(self, mirror):
        """Verifies bridge completion detection when the bridge touches a
        central ground component that's not part of the left or the right.
        Crossing over central ground is permitted as part of connecting the
        left and right. This test verifies the case where one bridge connects
        to the center ground by building up from it and the other adjoins the
        center ground at surface height."""
        # Test case:
        #
        #         [][][][]
        # [][]@@@@@@    @@
        # @@  @@@@@@    @@

        heights = [1, 0, 2, 2, 2, 0, 0, 2]
        width = len(heights)
        self.env.setup(width=width)

        _reset_helper(heights, self.env, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(0, self.env, width, mirror)
        _step_helper(6, self.env, width, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(4, self.env, width, mirror)
        self.assertTrue(self.env._is_bridge_complete())

    @parameterized.expand([(False,), (True,)])
    def test_is_bridge_complete_skip_center_ground(self, mirror):
        """Verifies bridge completion detection when the bridge avoids touching
        a central ground component that's not part of the left or the right.
        Touching central ground is not required."""
        # Test case:
        #
        #   [][][][]
        # [][]    [][]
        # @@  @@@@  @@

        heights = [1, 0, 1, 1, 0, 1]
        width = len(heights)
        self.env.setup(width=width)

        _reset_helper(heights, self.env, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(0, self.env, width, mirror)
        _step_helper(1, self.env, width, mirror)
        _step_helper(4, self.env, width, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(3, self.env, width, mirror)
        self.assertTrue(self.env._is_bridge_complete())

    @parameterized.expand([(False,), (True,)])
    def test_is_bridge_complete_multiple_center_segments(self, mirror):
        """Verifies bridge completion detection when the bridge touches
        multiple central land segments."""
        # Test case:
        #
        #               [][]@@
        #       [][]@@@@@@  @@
        # [][]@@@@  @@@@@@  @@
        # @@  @@@@  @@@@@@  @@

        heights = [1, 0, 2, 2, 0, 3, 3, 3, 0, 4]
        width = len(heights)
        self.env.setup(width=width)

        _reset_helper(heights, self.env, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(0, self.env, width, mirror)
        _step_helper(7, self.env, width, mirror)
        self.assertFalse(self.env._is_bridge_complete())
        _step_helper(3, self.env, width, mirror)
        self.assertTrue(self.env._is_bridge_complete())


if __name__ == "__main__":
    unittest.main()
