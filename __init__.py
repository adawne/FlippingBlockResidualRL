from gymnasium.envs.registration import register

register(
     id="PushBlock-v0",
     entry_point="researcj_main.envs:KukaPushBlockEnv",
)