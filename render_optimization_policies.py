import os
from os.path import exists

import numpy as np
from scipy.ndimage import zoom
import supersuit as ss
from array2gif import write_gif
from stable_baselines3 import PPO
import meltingpot_env
from meltingpot.python import substrate

env_name = "clean_up"
env_config = substrate.get_config(env_name)
n_agents = 7
num_frames = 4

env = meltingpot_env.env(
    env_config=env_config
)
env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
env = ss.frame_stack_v1(env, num_frames)

policies = os.listdir("./optimization_policies/")

for policy in policies:
    filepath = "./optimization_policies/" + policy + "/best_model"
    if not exists(filepath + '.zip'):
        continue
    print("Loading new policy ", filepath)
    model = PPO.load(filepath)

    obs_list = []
    i = 0
    env.reset()
    total_reward = 0

    while True:
        for agent in env.agent_iter():
            observation, reward, done, _ = env.last()
            action = (model.predict(observation, deterministic=True)[0] if not done else None)
            total_reward += reward

            env.step(action)
            i += 1
            if i % (len(env.possible_agents) + 1) == 0:
                obs = np.transpose(env.render(mode="rgb_array"), axes=(1, 0, 2))
                obs_list.append(obs)

        break

    total_reward = total_reward / n_agents
    print("writing gif")
    write_gif(
        obs_list, "./optimization_gifs/" + policy + "_" + str(total_reward)[:5] + ".gif", fps=5
    )
