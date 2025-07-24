from gymnasium.envs.registration import register

register(
    id="Stricker-v0",
    entry_point="vss_env.envs.stricker:StrickerEnv",
    max_episode_steps=3600
)