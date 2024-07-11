from gymnasium.envs.registration import register

register(
     id="research_main/FlipBlock-v0",
     entry_point="research_main.envs:URFlipBlockEnv",
)