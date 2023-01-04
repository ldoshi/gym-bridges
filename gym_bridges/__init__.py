from gym.envs.registration import register

register(
    id="Bridges-v0",
    entry_point="gym_bridges.envs:BridgesEnv",
)
