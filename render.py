import os
import sys

import numpy as np
from scipy.ndimage import zoom
import supersuit as ss
from array2gif import write_gif
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecTransposeImage

os.environ["SDL_VIDEODRIVER"] = "dummy"
num = sys.argv[1]

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

policies = os.listdir("./mature_policies/" + str(num) + "/")

for policy in policies:
    model = PPO.load("./mature_policies/" + str(num) + "/" + policy)

    for j in ['a','b','c','d','e']:

        obs_list = []
        i = 0
        env.reset()
        total_reward = 0

        while True:
            for agent in env.agent_iter():
                observation, reward, done, _ = env.last()
                action = (model.predict(observation, deterministic=False)[0] if not done else None)
                total_reward += reward

                env.step(action)
                i += 1
                if i % (len(env.possible_agents) + 1) == 0:
                    obs = np.transpose(env.render(mode="rgb_array"), axes=(1, 0, 2))
                    obs_list.append(obs)

            break

        total_reward = total_reward / n_agents

        if total_reward > 40:
            print("writing gif")
            write_gif(
                obs_list, "./mature_gifs/" + num + "_" + policy.split("_")[0] + j + '_' + str(total_reward)[:5] + ".gif", fps=5
            )

env.close()
