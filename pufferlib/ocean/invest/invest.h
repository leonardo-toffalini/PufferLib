#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int MAX_T = 500;

typedef struct {
  float score;
  float n;
} Log;

typedef struct {
  Log log;
  unsigned char *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int T;
  float risky;
  float riskless;
  int tick;
} Invest;

void c_reset(Invest *env) {
  env->risky = 0.0f;
  env->riskless = 0.0f;
  env->tick = 0;
}

void c_step(Invest *env) {
  env->tick += 1;
  env->rewards[0] = 0;
  env->terminals[0] = 0;

  int action = env->actions[0];

  // compute_observations(env);
}

void c_render(Invest *env) {
  if (!IsWindowReady()) {
    InitWindow(1080, 720, "PufferLib Invest");
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    SetTargetFPS(30);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  const float spacing = 1080 / (float)env->T;

  for (int i = 1; i < env->T; i++) {
    DrawLine(spacing * (i - 1), 100, spacing * i, 100, RED);
  }

  BeginDrawing();
  ClearBackground(BLACK);
  EndDrawing();
}

void c_close(Invest *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
