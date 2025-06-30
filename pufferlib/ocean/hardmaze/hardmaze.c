#include "hardmaze.h"

int main() {
  HardMaze env = {};
  env.observations = (float *)calloc(2 + 1 + 5, sizeof(float));
  env.actions = (int *)calloc(1, sizeof(int));
  env.rewards = (float *)calloc(1, sizeof(float));
  env.terminals = (unsigned char *)calloc(1, sizeof(unsigned char));

  c_reset(&env);
  c_render(&env);
  while (!WindowShouldClose()) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      env.actions[0] = 0;
      if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
        env.actions[0] = FORWARD;
      if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
        env.actions[0] = LEFT;
      if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
        env.actions[0] = RIGHT;
    } else {
      env.actions[0] = rand() % 4;
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
