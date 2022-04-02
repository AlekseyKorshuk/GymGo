from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env

from gym_go.envs import GoEnv
env = GoEnv(size=7, komi=0, reward_method='real', learn_rules=True)
env = make_vec_env(lambda: env, n_envs=16)
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("test2")
