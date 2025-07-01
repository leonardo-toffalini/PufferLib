/*
 DCP: Double Cart Pole
 The same as the original cartpole, but there is an extra arm
 attached to the first one with a revolut joint.

 obs: (float) q_0, q_1, q_2, q_0', q_1', q_2'
 where
 - q_0 is the x displacement of the cart
 - q_1 is the angle between the first arm and the cart
 - q_2 is the angle between the two arms

 actions: (int) -1, 0, 1
 how much force to apply to the cart
*/

#include "raylib.h"
#include <stdlib.h>
#include <string.h>

const unsigned char NOOP = 0;
const unsigned char LEFT = 1;
const unsigned char RIGHT = 2;

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

typedef struct {
  float perf;
  float score;
  float episode_return;
  float episode_length;
  float n;
} Log;

typedef struct {
  Log log;
  float *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int tick;
  float q0;
  float q1;
  float q2;
  float q0_dot;
  float q1_dot;
  float q2_dot;
} Dcp;

void add_log(Dcp *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

// Required function
void c_reset(Dcp *env) {
  memset(env->observations, 0, 6 * sizeof(float));
  env->tick = 0;
  env->q0 = 400.0f;
}

// Required function
void c_step(Dcp *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = 0;

  if (env->terminals[0]) {
    add_log(env);
    c_reset(env);
    return;
  }
}

void draw_rail(Dcp *env) { DrawLine(20, 320, 1220, 320, PUFF_WHITE); }

void draw_cart(Dcp *env) {
  DrawRectangle(env->q0, 320 - 20, 60, 40, PUFF_CYAN);
}

// Required function. Should handle creating the client on first call
void c_render(Dcp *env) {
  if (!IsWindowReady()) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(1240, 640, "PufferLib Dcp");
    SetTargetFPS(30);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground((Color){6, 24, 24, 255});
  draw_rail(env);
  draw_cart(env);

  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Dcp *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
