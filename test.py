from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env

from gym_go.envs import GoEnv

env = GoEnv(size=7, komi=0, reward_method='real')
# env = make_vec_env(lambda: env, n_envs=1)
#
# model = PPO2.load("test2")
#
# # Enjoy trained agent
# obs = env.reset()
# dones = False
# while not dones:
#     env.render("terminal")
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)


print(env.action_masks())
