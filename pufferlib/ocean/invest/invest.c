#include "invest.h"
#include <stdio.h>

int encode_action(int a) {
  return a + 10; // i know this is super simple but it abstracts away the mistery + 10
}

int main() {
  Invest env = {.T = 128,
                .H = 0.1,
                .process_type = 1,
                .liquidate = 1,
                .liq_type = 1,
                .friction_coef = 0.01f,
                .friction_power = 2,
                .price_window_size = 32};
  env.observations = (float *)calloc(37, sizeof(float));
  env.actions = (float *)calloc(1, sizeof(float));
  env.rewards = (float *)calloc(1, sizeof(float));
  env.terminals = (unsigned char *)calloc(1, sizeof(unsigned char));

  c_reset(&env);
  c_render(&env);
  while (!WindowShouldClose()) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
        env.actions[0] = -100.0f;
      } else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
        env.actions[0] = 100.0f;
      } else {
        env.actions[0] = 0.0f;
      }
    } else {
      env.actions[0] = env.prices[env.tick] > 0 ? -10.0f : 10.0f;
      // env.actions[0] = rand() % 21;
    }
    if (IsKeyDown(KEY_SPACE)) {
      env.render_frames_remaining++;
    }
    c_step(&env);
    c_render(&env);
  }
  free(env.observations);
  free(env.actions);
  free(env.rewards);
  free(env.terminals);
  c_close(&env);
}
