from gymnasium.envs.registration import register

register(
     id="PushBlock-v0",
     entry_point="research_main.envs:KukaPushBlockEnv",
)