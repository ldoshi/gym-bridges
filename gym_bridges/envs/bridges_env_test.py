import numpy as np
import pytest
import time
import unittest

from collections import defaultdict
from parameterized import parameterized

from bridges_env import BridgesEnv


def _check_block(state, index, state_type):
    """Verifies that the block start at index is grounded in the environment and
    maintains a level height until the next gap. Returns the index following this
    block on success. Gaps are checked as blocks of height 0."""
    env_height = state.shape[0] - 1
    block_height = 0

    while (
        block_height < state.shape[0]
        and state[env_height - block_height, index] == state_type
    ):
        block_height += 1

    i = index + 1
    while i < state.shape[1] and state[env_height, i] == state_type:
        i += 1

    # Verify the block is all of state_type.
    if not np.where(state[-block_height:, index:i] == state_type, True, False).all():
        return False

    # Verify everything above the block is *not* of state_type.
    if np.where(state[:-block_height, index:i] == state_type, True, False).any():
        return False

    return i


def _state_builder(shape, heights):
    state = np.zeros(shape)
    state_height = state.shape[0]
    for i, height in enumerate(heights):
        state[state_height - height :, i] = BridgesEnv.StateType.GROUND

    return state


def _step_helper(action, env, width, mirror=False):
    """Takes the desired step and returns done status

    Args:
      mirror: Flips the action horizontally across a vertical axis
      down the center of the environment, adjusted for brick width.

    Returns:
      True if the bridge is complete following this action.

    """
    if mirror:
        # The -2 comes from the following:
        # * -1 to account for 0-indexing
        # * -1 to account for the brick width of 2. The action space
        #   is actually (width - (brick_width - 1)) in size.
        action = width - 2 - action

    _, _, done, _ = env.step(action)
    return done


def _reset_helper(heights, env, mirror=False):
    if mirror:
        heights.reverse()

    env.reset(state=_state_builder(env.shape, heights))


class TestBridgesEnv(unittest.TestCase):
    def test_step_simple(self):
        """Verifies the rewards and done for step."""

        heights = [1, 0, 0, 1]
        width = len(heights)
        env = BridgesEnv(width=width)
        _reset_helper(heights, env)
        _, reward, done, _ = env.step(1)
        self.assertEqual(reward, -5)
        self.assertFalse(done)
        _, reward, done, _ = env.step(2)
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        _, reward, done, _ = env.step(0)
        self.assertEqual(reward, 100)
        self.assertTrue(done)

    @parameterized.expand([(3,), (9,)])
    def test_force_standard_config(self, width):
        """The standard configuration only has ground at the bottom left
        and right locations of the env."""
        env = BridgesEnv(width=width, force_standard_config=True)
        height = env.shape[0] - 1
        initial_state = env.reset()
        self.assertEqual(initial_state[height, 0], BridgesEnv.StateType.GROUND)
        self.assertEqual(initial_state[height, width - 1], BridgesEnv.StateType.GROUND)
        self.assertEqual(
            len(np.where(initial_state == BridgesEnv.StateType.GROUND)[0]), 2
        )

    @parameterized.expand([(3,), (9,)])
    def test_gap_count(self, width):
        """Verifies the requested number of gaps exist. This test also verifies
        that the env is well-formed in terms of having initial blocks of ground
        properly grounded to the bottom of the env and level between gaps. The
        check also ensures that gap sections are completely empty."""
        env = BridgesEnv(width=width)
        for gap_count in range(1, ((width + 1) // 2)):
            initial_state = env.reset(gap_count=gap_count)

            index = 0
            state_type = BridgesEnv.StateType.GROUND
            gap_counter = 0
            while index < width:
                index = _check_block(initial_state, index, state_type)
                self.assertTrue(isinstance(index, int))
                self.assertGreater(index, 0)

                if state_type == BridgesEnv.StateType.GROUND:
                    state_type = BridgesEnv.StateType.EMPTY
                else:
                    state_type = BridgesEnv.StateType.GROUND
                    # Increment the gap everytime we reach the end of
                    # a gap.
                    gap_counter += 1

            self.assertEqual(gap_counter, gap_count)

    @parameterized.expand([(3,), (9,)])
    def test_max_gap_count(self, width):
        """Verifies the gap count varies between 1 and the max gap count."""
        for gap_count in range(1, ((width + 1) // 2)):
            env = BridgesEnv(width=width, max_gap_count=gap_count)
            height = env.shape[0] - 1
            counts = defaultdict(lambda: 0)
            for _ in range(100):
                initial_state = env.reset()
                counts[
                    int(
                        np.sum(
                            [
                                1
                                if (
                                    initial_state[height, i]
                                    == BridgesEnv.StateType.EMPTY
                                    and initial_state[height, i + 1]
                                    == BridgesEnv.StateType.GROUND
                                )
                                else 0
                                for i in range(env.shape[1] - 1)
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
        env = BridgesEnv(width=width)

        _reset_helper(heights, env, mirror)
        self.assertFalse(_step_helper(0, env, width, mirror))
        self.assertFalse(_step_helper(0, env, width, mirror))
        self.assertTrue(_step_helper(2, env, width, mirror))

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
        env = BridgesEnv(width=width)

        # Side completion to starting point.
        _reset_helper(heights, env, mirror)
        self.assertTrue(_step_helper(1, env, width, mirror))

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
        env = BridgesEnv(width=width)

        # Test two completions with a wide starting point.
        _reset_helper(heights, env, mirror)
        self.assertFalse(_step_helper(3, env, width, mirror))
        self.assertTrue(_step_helper(1, env, width, mirror))

        _reset_helper(heights, env, mirror)
        self.assertFalse(_step_helper(3, env, width, mirror))
        _step_helper(0, env, width, mirror)
        self.assertFalse(_step_helper(2, env, width, mirror))
        self.assertTrue(_step_helper(0, env, width, mirror))

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
        env = BridgesEnv(width=width)

        _reset_helper(heights, env, mirror)
        self.assertFalse(_step_helper(0, env, width, mirror))
        self.assertTrue(_step_helper(0, env, width, mirror))

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
        env = BridgesEnv(width=width)

        _reset_helper(heights, env, mirror)
        self.assertFalse(_step_helper(0, env, width, mirror))
        self.assertTrue(_step_helper(4, env, width, mirror))

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
        env = BridgesEnv(width=width)

        _reset_helper(heights, env, mirror)
        _step_helper(0, env, width, mirror)
        self.assertFalse(_step_helper(7, env, width, mirror))
        _step_helper(2, env, width, mirror)
        self.assertTrue(_step_helper(5, env, width, mirror))

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
        env = BridgesEnv(width=width)

        _reset_helper(heights, env, mirror)
        _step_helper(0, env, width, mirror)
        self.assertFalse(_step_helper(6, env, width, mirror))
        self.assertTrue(_step_helper(4, env, width, mirror))

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
        env = BridgesEnv(width=width)

        _reset_helper(heights, env, mirror)
        _step_helper(0, env, width, mirror)
        _step_helper(1, env, width, mirror)
        self.assertFalse(_step_helper(4, env, width, mirror))
        self.assertTrue(_step_helper(3, env, width, mirror))

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
        env = BridgesEnv(width=width)

        _reset_helper(heights, env, mirror)
        _step_helper(0, env, width, mirror)
        self.assertFalse(_step_helper(7, env, width, mirror))
        self.assertTrue(_step_helper(3, env, width, mirror))

    def test_reset_random(self):
        """Verifies the usage of the random seed for consistent resets."""

        env = BridgesEnv(width=8)
        expected_states = [env.reset() for _ in range(100)]
        regeneration_env = BridgesEnv(width=8)
        regenerated_states = [regeneration_env.reset() for _ in range(100)]
        self.assertFalse(
            all(
                [
                    (expected == regenerated).all()
                    for expected, regenerated in zip(
                        expected_states, regenerated_states
                    )
                ]
            )
        )

        seed = int(time.time() * 1e6)
        env = BridgesEnv(width=8, seed=seed)
        expected_states = [env.reset() for _ in range(100)]
        regeneration_env = BridgesEnv(width=8, seed=seed)
        regenerated_states = [regeneration_env.reset() for _ in range(100)]
        self.assertTrue(
            all(
                [
                    (expected == regenerated).all()
                    for expected, regenerated in zip(
                        expected_states, regenerated_states
                    )
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
