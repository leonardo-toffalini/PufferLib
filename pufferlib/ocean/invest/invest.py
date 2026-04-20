import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.invest import binding

def process_name_to_process_id(process_name: str):
    if process_name == "sin":
        return 0
    if process_name == "fbm":
        return 1
    return 0
    

class Invest(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, time_horizon=100,
                 t_min=None, t_max=None, T_min=None, T_max=None, hurst=0.1, process_type="fbm", liquidate=1,
                 friction_coef=0.00, friction_power=2, liq_type=1, price_window_size=32,
                 prediction_len=0, reward_scale=1e-3, action_low=-10.0, action_high=10.0,
                 normalize_observations=False,
                 obs_norm_alpha=1e-3, obs_norm_eps=1e-8, obs_norm_clip=5.0, buf=None, seed=0):
        action_low = float(action_low)
        action_high = float(action_high)
        if not np.isfinite(action_low) or not np.isfinite(action_high):
            raise ValueError("action_low and action_high must be finite")
        if action_low >= action_high:
            raise ValueError(f"action_low ({action_low}) must be < action_high ({action_high})")

        self.single_observation_space = gymnasium.spaces.Box(
            low=-obs_norm_clip,
            high=obs_norm_clip,
            shape=(5 + price_window_size,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=action_low,
            high=action_high,
            shape=(1,),
            dtype=np.float32,
        )
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.reward_scale = float(reward_scale)
        self.action_low = action_low
        self.action_high = action_high
        self.normalize_observations = bool(normalize_observations)
        self.obs_norm_alpha = float(obs_norm_alpha)
        self.obs_norm_eps = float(obs_norm_eps)
        self.obs_norm_clip = float(obs_norm_clip)
        self._obs_stats_initialized = False

        process_id = process_name_to_process_id(process_type)

        # Prefer lowercase (from configparser), fallback to uppercase
        if t_min is None:
            t_min = T_min
        if t_max is None:
            t_max = T_max
        # Default to fixed horizon if still None
        if t_min is None:
            t_min = time_horizon
        if t_max is None:
            t_max = time_horizon

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            time_horizon=time_horizon,
            t_min=t_min,
            t_max=t_max,
            hurst=hurst,
            process_type=process_id,
            liquidate=liquidate,
            friction_coef=friction_coef,
            friction_power=friction_power,
            liq_type=liq_type,
            price_window_size=price_window_size,
            prediction_len=prediction_len,
            reward_scale=self.reward_scale,
        )
        obs_dim = self.single_observation_space.shape[0]
        self._obs_mean = np.zeros((obs_dim,), dtype=np.float32)
        self._obs_var = np.ones((obs_dim,), dtype=np.float32)

    def _update_obs_stats(self, obs):
        batch_mean = obs.mean(axis=0, dtype=np.float32)
        batch_var = obs.var(axis=0, dtype=np.float32)
        if not self._obs_stats_initialized:
            self._obs_mean[:] = batch_mean
            self._obs_var[:] = np.maximum(batch_var, self.obs_norm_eps)
            self._obs_stats_initialized = True
            return

        alpha = self.obs_norm_alpha
        self._obs_mean[:] = (1.0 - alpha) * self._obs_mean + alpha * batch_mean
        self._obs_var[:] = (1.0 - alpha) * self._obs_var + alpha * batch_var
        self._obs_var[:] = np.maximum(self._obs_var, self.obs_norm_eps)

    def _normalize_observations(self):
        obs = self.observations
        self._update_obs_stats(obs)
        denom = np.sqrt(self._obs_var + self.obs_norm_eps)
        obs[:] = (obs - self._obs_mean) / denom
        np.clip(obs, -self.obs_norm_clip, self.obs_norm_clip, out=obs)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        if self.normalize_observations:
            self._normalize_observations()
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        if self.normalize_observations:
            self._normalize_observations()
        info = [binding.vec_log(self.c_envs)]
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 1
    env = Invest(num_envs=N)
    obs, _ = env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.uniform(env.action_low, env.action_high, (CACHE, N, 1)).astype(np.float32)

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        # obs, rewards, terminals, truncations, info = env.step(actions[steps % CACHE])
        steps += 1

    sps = int(env.num_agents * steps / (time.time() - start))
    print(f'Invest SPS: {sps:,}')
