import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.invest import binding

def process_name_to_process_id(process_name: str):
    if process_name == "sin":
        return 0
    if process_name == "fbmm":
        return 1
    

class Invest(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, time_horizon=100,
                 hurst=0.1, process_type="sin", buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(4,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(21)
        self.render_mode = render_mode
        self.num_agents = num_envs

        process_id = process_name_to_process_id(process_type)

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, time_horizon=time_horizon,
            hurst=hurst, process_type=process_id)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 2048
    env = Invest(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(-5, 5, (CACHE, N))

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        # obs, rewards, terminals, truncations, info = env.step(actions[steps % CACHE])
        # print(obs)
        # input("Press enter to continue...")
        steps += 1

    sps = int(env.num_agents * steps / (time.time() - start))
    print(f'Invest SPS: {sps:,}')
