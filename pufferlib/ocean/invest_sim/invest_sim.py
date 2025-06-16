'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.invest_sim import binding

class InvestSim(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, time_horizon=100, window_size=10, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4 + window_size,), dtype=np.float32
        )
        # self.single_action_space = gymnasium.spaces.Box(
        #     low=-100.0, high=100.0, shape=(1,), dtype=np.float32
        # )
        self.single_action_space = gymnasium.spaces.Discrete(100)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.actions = self.actions.flatten()
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, time_horizon=time_horizon, window_size=window_size)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        actions = np.clip(actions.flatten(), -100.0, 100.0)
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 1024

    env = InvestSim(num_envs=N, time_horizon=20, window_size=10)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(-100, 100, (CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    sps = int(steps / (time.time() - start))
    print(f'InvestSim SPS: {sps:,}')
