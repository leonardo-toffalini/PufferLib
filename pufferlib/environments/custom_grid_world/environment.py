from pdb import set_trace as T

import gymnasium
import functools

import pufferlib.emulation
import pufferlib.environments
from pufferlib.pufferlib import EpisodeStats
from .custom_gridworld import GridWorldEnv

gymnasium.register(
    id="CustomGridWorld-v0",
    entry_point=GridWorldEnv,
)

def env_creator(name='CustomGridWorld-v0'):
    return functools.partial(make, name=name)

def make(name, size=11, num_envs=1, buf=None):
    if name == "custom_grid_world":
        name = "CustomGridWorld-v0"

    env = gymnasium.make(name, size=size, num_envs=num_envs)
    env = EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
