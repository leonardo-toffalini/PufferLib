#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// color palette: https://lospec.com/palette-list/bluem0ld
const Color MY_BLACK = (Color){25, 27, 26, 255};
const Color MY_DARK = (Color){41, 66, 87, 255};
const Color MY_MEDIUM = (Color){87, 156, 154, 255};
const Color MY_LIGHT = (Color){153, 201, 179, 255};

// actions
const int LEFT = 0;
const int DOWN = 1;
const int RIGHT = 2;

const float GRAVITY = 9.81f;
const int PLAYER_SIZE = 40;
const float ASPECT_RATIO = 1.6f;
const float ANGLE_SPEED = 3.0f;

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
  int size;
  Vector2 player_pos;
  Vector2 player_vel;
  float player_angle;
  int goal;
} Lander;

void c_reset(Lander *env) {
  env->player_pos =
      (Vector2){ASPECT_RATIO * env->size / 2.0f, env->size / 2.0f};
  env->player_angle = 0.0f;
  env->goal = (rand() % 2 == 0) ? env->size : -env->size;
}

void c_step(Lander *env) {
  env->rewards[0] = 0;
  env->terminals[0] = 0;
  int action = env->actions[0];
  switch (action) {
  case LEFT:
    env->player_angle -= ANGLE_SPEED;
    break;
  case DOWN:
    break;
  case RIGHT:
    env->player_angle += ANGLE_SPEED;
    break;
  default:
    break;
  }
}

void c_render(Lander *env) {
  if (!IsWindowReady()) {
    InitWindow(ASPECT_RATIO * env->size, env->size, "PufferLib Lander");
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    SetTargetFPS(30);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  Rectangle player = {env->player_pos.x, env->player_pos.y, PLAYER_SIZE,
                      PLAYER_SIZE};
  Vector2 origin = {PLAYER_SIZE / 2.0f, PLAYER_SIZE / 2.0f};

  DrawRectanglePro(player, origin, env->player_angle, MY_MEDIUM);
  DrawPixel(env->player_pos.x, env->player_pos.y, RED);

  BeginDrawing();
  ClearBackground(MY_DARK);
  EndDrawing();
}

void c_close(Lander *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
