from gymnasium.envs.registration import register

register(
    id="Striker-v0",
    entry_point="vss_env.envs.striker:StrikerEnv",
    max_episode_steps=3600,
)
