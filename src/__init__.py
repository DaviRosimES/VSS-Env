from gymnasium.envs.registration import register

register(
    id="Stricker-Env", entry_point="VSS-Env.src.envs.stricker:StrickerEnv", max_episode_steps=3600
)