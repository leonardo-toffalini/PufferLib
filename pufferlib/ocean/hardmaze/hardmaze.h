#include "raylib.h"
#include "raymath.h"
#include <stdlib.h>
#include <string.h>

const unsigned char NOOP = 0;
const unsigned char LEFT = 1;
const unsigned char FORWARD = 2;
const unsigned char RIGHT = 3;

const float ANGULAR_SPEED = 0.1f;
const float LINEAR_SPEED = 2.0f;

#define MAX_WALLS 20

typedef struct {
  float perf;
  float score;
  float episode_return;
  float episode_length;

  float n;
} Log;

typedef struct {
  Vector2 pos;
  Vector2 forward;
  float radius;
  float radius_sq;
} Player;

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
  Player player;
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
  // boundaries
  env->walls[wall_idx++] = (Wall){64, 64, 576, 64};
  env->walls[wall_idx++] = (Wall){64, 64, 64, 576};
  env->walls[wall_idx++] = (Wall){576, 64, 576, 576};
  env->walls[wall_idx++] = (Wall){64, 576, 576, 576};

  // obstacles
  env->walls[wall_idx++] = (Wall){64, 250, 144, 330};
  env->walls[wall_idx++] = (Wall){64, 200, 244, 180};
  env->walls[wall_idx++] = (Wall){244, 180, 234, 480};
  env->walls[wall_idx++] = (Wall){244, 180, 364, 140};
  env->walls[wall_idx++] = (Wall){239, 330, 434, 480};
  env->walls[wall_idx++] = (Wall){264, 576, 344, 496};
  env->walls[wall_idx++] = (Wall){576, 420, 324, 240};
}

void c_reset(HardMaze *env) {
  set_up_walls(env);
  Player player = {
      (Vector2){100.0f, 540.0f},
      (Vector2){0.0f, -1.0f},
      10.0f,
      100.0f,
  };
  env->player = player;
}

int check_collisions(HardMaze *env) {
  Wall w;
  Vector2 p1, p2;
  for (int i = 0; i < MAX_WALLS; i++) {
    w = env->walls[i];
    p1 = (Vector2){w.ax, w.ay};
    p2 = (Vector2){w.bx, w.by};
    if (CheckCollisionCircleLine(env->player.pos, env->player.radius, p1, p2))
      return 1;
  }
  return 0;
}

void execute_action(HardMaze *env, int action) {
  Vector2 prev_pos = env->player.pos;

  switch (action) {
  case LEFT:
    env->player.forward = Vector2Rotate(env->player.forward, -ANGULAR_SPEED);
    break;
  case RIGHT:
    env->player.forward = Vector2Rotate(env->player.forward, ANGULAR_SPEED);
    break;
  case FORWARD:
    env->player.pos = Vector2Add(
        env->player.pos, Vector2Scale(env->player.forward, LINEAR_SPEED));
  default:
    break;
  }

  if (check_collisions(env)) {
    env->player.pos = prev_pos;
  }
}

void c_step(HardMaze *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = 0;

  execute_action(env, action);
}

void draw_player(HardMaze *env) {
  Vector2 center = env->player.pos;
  float r = env->player.radius;
  Vector2 forward = env->player.forward;
  DrawRing(center, r - 2, r, 0, 360, 64, BLACK);
  DrawCircleV(center, r - 2, RED);
  DrawLineEx(center, Vector2Add(center, Vector2Scale(forward, 2 * r)), 2, BLUE);
}

void c_render(HardMaze *env) {
  if (!IsWindowReady()) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(640, 640, "PufferLib HardMaze");
    SetTargetFPS(30);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(RAYWHITE);

  for (int i = 0; i < MAX_WALLS; i++) {
    Wall w = env->walls[i];
    DrawLine(w.ax, w.ay, w.bx, w.by, BLACK);
  }

  draw_player(env);

  EndDrawing();
}

void c_close(HardMaze *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
