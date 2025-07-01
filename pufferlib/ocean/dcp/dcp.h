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
#include "raymath.h"
#include <stdlib.h>
#include <string.h>

const unsigned char NOOP = 0;
const unsigned char LEFT = 1;
const unsigned char RIGHT = 2;

const int CART_WIDTH = 60;
const int CART_HEIGHT = 40;

const float POLE_LENGHT = 100.0f;

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

void c_reset(Dcp *env) {
  memset(env->observations, 0, 6 * sizeof(float));
  env->tick = 0;
  env->q0 = 620.0f;
  env->q1 = PI / 4;
  env->q2 = PI / 6;
  env->q0_dot = 0.0f;
  env->q1_dot = 0.0f;
  env->q2_dot = 0.0f;
}

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
  DrawRectangle(env->q0 - CART_WIDTH / 2.0f, 320 - CART_HEIGHT / 2.0f,
                CART_WIDTH, CART_HEIGHT, PUFF_CYAN);

  Vector2 center = (Vector2){env->q0, 320};
  Vector2 right_base = (Vector2){1.0f, 0.0f};
  Vector2 pole1_end = Vector2Add(
      center, Vector2Scale(Vector2Rotate(right_base, env->q1), POLE_LENGHT));
  Vector2 pole2_end = Vector2Add(
      pole1_end,
      Vector2Scale(Vector2Rotate(right_base, env->q1 + env->q2), POLE_LENGHT));
  DrawLineEx(center, pole1_end, 5, GREEN);
  DrawLineEx(pole1_end, pole2_end, 5, YELLOW);
  DrawCircleV(center, 7, PUFF_RED);
  DrawCircleV(pole1_end, 5, PUFF_RED);
}

void draw_stats(Dcp *env) {
  DrawText(TextFormat("Steps: %i", env->tick), 10, 10, 20, PUFF_WHITE);
  DrawText(TextFormat("Position: %.2f", env->q0), 10, 40, 20, PUFF_WHITE);
  DrawText(TextFormat("Angle 1: %.2f", env->q1 * 180.0f / M_PI), 10, 70, 20,
           PUFF_WHITE);
  DrawText(TextFormat("Angle 2: %.2f", env->q2 * 180.0f / M_PI), 10, 100, 20,
           PUFF_WHITE);

  DrawText(TextFormat("Vel: %.2f", env->q0_dot), 10, 550, 20, PUFF_WHITE);
  DrawText(TextFormat("Omega 1: %.2f", env->q1_dot * 180.0f / M_PI), 10, 580,
           20, PUFF_WHITE);
  DrawText(TextFormat("Omega 2: %.2f", env->q2_dot * 180.0f / M_PI), 10, 610,
           20, PUFF_WHITE);
}

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
  draw_stats(env);

  EndDrawing();
}

void c_close(Dcp *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
