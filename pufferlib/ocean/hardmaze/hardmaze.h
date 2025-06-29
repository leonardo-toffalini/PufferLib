#include "raylib.h"
#include <stdlib.h>
#include <string.h>

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char TARGET = 2;

#define MAX_WALLS 20

typedef struct {
  float perf;
  float score;
  float episode_return;
  float episode_length;

  float n;
} Log;

typedef struct {
  float ax;
  float ay;
  float bx;
  float by;
} Wall;

typedef struct {
  Log log;
  unsigned char *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int size;
  int tick;
  int r;
  int c;
  Wall walls[MAX_WALLS];
} HardMaze;

void add_log(HardMaze *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

void set_up_walls(HardMaze *env) {
  int wall_idx = 0;
  env->walls[wall_idx++] = (Wall){64, 250, 144, 330};
  env->walls[wall_idx++] = (Wall){64, 200, 244, 180};
  env->walls[wall_idx++] = (Wall){244, 180, 234, 480};
  env->walls[wall_idx++] = (Wall){244, 180, 364, 140};
  env->walls[wall_idx++] = (Wall){239, 330, 434, 480};
  env->walls[wall_idx++] = (Wall){264, 576, 344, 496};
  env->walls[wall_idx++] = (Wall){576, 420, 324, 240};
}

void c_reset(HardMaze *env) { set_up_walls(env); }

void c_step(HardMaze *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = 0;
}

void draw_box() {
  // top
  DrawLine(64, 64, 640 - 64, 64, BLACK);
  // left
  DrawLine(64, 64, 64, 640 - 64, BLACK);
  // right
  DrawLine(640 - 64, 64, 640 - 64, 640 - 64, BLACK);
  // bottom
  DrawLine(64, 640 - 64, 640 - 64, 640 - 64, BLACK);
}

void c_render(HardMaze *env) {
  if (!IsWindowReady()) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(640, 640, "PufferLib HardMaze");
    SetTargetFPS(5);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(RAYWHITE);

  draw_box();

  for (int i = 0; i < MAX_WALLS; i++) {
    Wall w = env->walls[i];
    DrawLine(w.ax, w.ay, w.bx, w.by, BLACK);
  }

  EndDrawing();
}

void c_close(HardMaze *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
