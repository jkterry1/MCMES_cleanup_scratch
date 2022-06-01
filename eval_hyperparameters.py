import json
import sys

import gym
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
import meltingpot_env
from meltingpot.python import substrate

num = sys.argv[1]
n_evaluations = 20
n_agents = 7
n_cpus = 4
n_envs = 8
n_timesteps = 2000000
env_name = "clean_up"
env_config = substrate.get_config(env_name)

with open("./hyperparameter_jsons/" + "hyperparameters_" + num + ".json") as f:
    params = json.load(f)

print(params)

num_frames = params["num_frames"] if "num_frames" in params else 4


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=256,
        num_frames=4,
        activation_fn=nn.ReLU,
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.conv = nn.Sequential(
            nn.Conv2d(num_frames * 3, num_frames * 6, kernel_size=8, stride=4, padding=0),
            activation_fn(), # 24 * 21 * 21
            nn.Conv2d(num_frames * 6, num_frames * 12, kernel_size=5, stride=2, padding=0),
            activation_fn(), # 48 * 9 * 9
            nn.Flatten()
        )
        flat_out = num_frames * 12 * 9 * 9
        self.fc = nn.Sequential(
            nn.Linear(in_features=flat_out, out_features=features_dim),
            activation_fn(),
        )

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = observations.permute(0, 3, 1, 2)
        features = self.conv(observations)
        features = self.fc(features)
        return features

activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[params["activation_fn"]]
net_arch = {"small": [dict(pi=[32, 32], vf=[32, 32])], "medium": [dict(pi=[128, 64], vf=[128, 64])], "large": [dict(pi=[256, 256], vf=[256, 256])], "extra_large": [dict(pi=[1024, 512, 256], vf=[1024, 512, 256])]}[params["net_arch"]]

params["policy_kwargs"] = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(
        num_frames=num_frames,
        activation_fn=activation_fn
    ),
    net_arch=net_arch,
)

del params["net_arch"]
del params["activation_fn"]

env = meltingpot_env.parallel_env(
    env_config=env_config
)
env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
env = ss.frame_stack_v1(env, num_frames)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, n_envs, num_cpus=n_cpus, base_class="stable_baselines3")
env = VecTransposeImage(env, skip=True)
env = VecMonitor(env)

eval_env = meltingpot_env.parallel_env(
    env_config=env_config
)
eval_env = ss.observation_lambda_v0(eval_env, lambda x, _: x["RGB"], lambda s: s["RGB"])
eval_env = ss.frame_stack_v1(eval_env, num_frames)
eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
eval_env = ss.concat_vec_envs_v1(
    eval_env, 1, num_cpus=n_cpus, base_class="stable_baselines3"
)
eval_env = VecTransposeImage(eval_env, skip=True)
eval_env = VecMonitor(eval_env)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_envs * n_agents), 1)

all_mean_rewards = []

for i in range(10):
    model = PPO("CnnPolicy", env, verbose=1, **params)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./eval_logs/" + num + "/" + str(i) + "/",
        log_path="./eval_logs/" + num + "/" + str(i) + "/",
        eval_freq=eval_freq,
        deterministic=False,
        render=False,
    )
    model.learn(total_timesteps=n_timesteps, callback=eval_callback)
    model = PPO.load("./eval_logs/" + num + "/" + str(i) + "/" + "best_model")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, deterministic=False, n_eval_episodes=25
    )
    print(mean_reward)
    print(std_reward)
    all_mean_rewards.append(mean_reward)
    if mean_reward > 40:
        model.save(
            "./mature_policies/"
            + str(num)
            + "/"
            + str(i)
            + "_"
            + str(mean_reward).split(".")[0]
            + ".zip"
        )

if len(all_mean_rewards) > 0:
    print(sum(all_mean_rewards) / len(all_mean_rewards))
else:
    print("No mature policies found")
