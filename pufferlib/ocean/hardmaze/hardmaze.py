import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.hardmaze import binding

class HardMaze(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, buf=None, seed=0,
                 linear_speed=4.0, angular_speed=3.141592/30, radar_range=150,
                 range_finder_len=60.0, frame_skip=1, max_ticks=1500):
        obs_size = 2 + 5 + 4 # player pos (2) + range finder readings (5) + radar readings one hot (4)
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(obs_size,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(4) # noop, left, forward, right
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, linear_speed=linear_speed, angular_speed=angular_speed,
            radar_range=radar_range, range_finder_len=range_finder_len, frame_skip=frame_skip, max_ticks=max_ticks)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

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
    N = 4096

    env = HardMaze(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        # obs, rew, term, trunc, info = env.step(actions[i % CACHE])
        # print(obs)
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    sps = int(steps / (time.time() - start))
    print(f'HardMaze SPS: {sps:,}')
