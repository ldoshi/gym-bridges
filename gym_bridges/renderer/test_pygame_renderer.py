import numpy as np
import pytest
import time

from gym_bridges.envs.bridges_env import BridgesEnv


@pytest.mark.parametrize("width", [3, 9])
def test_force_standard_config_render(width):
    """The standard configuration only has ground at the bottom left
    and right locations of the env."""
    env = BridgesEnv(width=width, force_standard_config=True)
    height = env.shape[0] - 1
    initial_state = env.reset()
    env.render(mode="pygame")
    assert initial_state[height, 0] == BridgesEnv.StateType.GROUND
    assert initial_state[height, width - 1] == BridgesEnv.StateType.GROUND
    assert len(np.where(initial_state == BridgesEnv.StateType.GROUND)[0]) == 2
    time.sleep(5)
