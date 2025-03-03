from gymnasium.envs.registration import register

register(
    id="Stricker-v0",
    entry_point="src.envs:StrickerEnv",
    max_episode_steps=3600
)