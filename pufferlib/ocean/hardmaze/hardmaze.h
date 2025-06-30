#include "raylib.h"
#include "raymath.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const unsigned char NOOP = 0;
const unsigned char LEFT = 1;
const unsigned char FORWARD = 2;
const unsigned char RIGHT = 3;

const float ANGULAR_SPEED = PI / 60;
const float LINEAR_SPEED = 2.0f;

#define MAX_WALLS 20
#define NUM_RANGE_FINDERS 5

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
  float angle;
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
  float angle;
  float max_range;
  float distance;
} RangeFinder;

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
  RangeFinder range_finders[NUM_RANGE_FINDERS];
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

void set_up_range_finders(HardMaze *env) {
  float delta_angle = PI / (NUM_RANGE_FINDERS - 1);
  for (int i = 0; i < NUM_RANGE_FINDERS; i++) {
    env->range_finders[i] = (RangeFinder){i * delta_angle, 40.0f, 1.0f};
  }
}

void c_reset(HardMaze *env) {
  set_up_walls(env);
  set_up_range_finders(env);

  Player player = {
      (Vector2){100.0f, 540.0f}, (Vector2){0.0f, -1.0f}, PI / 2, 10.0f, 100.0f,
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

void update_range_finders(HardMaze *env) {
  RangeFinder rf;
  Wall w;
  Vector2 collision_point, direction, end;
  Vector2 base_vec = (Vector2){0.0f, 1.0f};
  Vector2 center = env->player.pos;
  float dist;

  for (int i = 0; i < NUM_RANGE_FINDERS; i++) {
    env->range_finders[i].distance = 1.0f;
    rf = env->range_finders[i];

    for (int j = 0; j < MAX_WALLS; j++) {
      w = env->walls[j];

      direction = Vector2Rotate(base_vec, rf.angle + env->player.angle);

      end = Vector2Add(center, Vector2Scale(direction, rf.max_range));
      if (!CheckCollisionLines(center, end, (Vector2){w.ax, w.ay},
                               (Vector2){w.bx, w.by}, &collision_point))
        continue;

      dist = Vector2Distance(collision_point, center) / rf.max_range;
      if (dist < rf.distance) {
        env->range_finders[i].distance = dist;
      }
    }
  }
}

void execute_action(HardMaze *env, int action) {
  Vector2 prev_pos = env->player.pos;

  switch (action) {
  case LEFT:
    env->player.forward = Vector2Rotate(env->player.forward, -ANGULAR_SPEED);
    env->player.angle -= ANGULAR_SPEED;
    break;
  case RIGHT:
    env->player.forward = Vector2Rotate(env->player.forward, ANGULAR_SPEED);
    env->player.angle += ANGULAR_SPEED;
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

  update_range_finders(env);
}

void c_step(HardMaze *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = 0;

  execute_action(env, action);
}

void draw_walls(HardMaze *env) {
  for (int i = 0; i < MAX_WALLS; i++) {
    Wall w = env->walls[i];
    DrawLine(w.ax, w.ay, w.bx, w.by, BLACK);
  }
}

void draw_player(HardMaze *env) {
  Vector2 center = env->player.pos;
  float r = env->player.radius;
  DrawRing(center, r - 2, r, 0, 360, 64, BLACK);
  DrawCircleV(center, r - 2, RED);
}

void draw_range_finders(HardMaze *env) {
  Vector2 center = env->player.pos;
  Vector2 base_vec = (Vector2){0.0f, 1.0f};
  Vector2 direction;
  RangeFinder rf;
  for (int i = 0; i < NUM_RANGE_FINDERS; i++) {
    rf = env->range_finders[i];
    direction = Vector2Rotate(base_vec, rf.angle + env->player.angle);
    DrawLineEx(
        Vector2Add(center, Vector2Scale(direction, env->player.radius)),
        Vector2Add(center, Vector2Scale(direction, rf.distance * rf.max_range)),
        2, GREEN);
  }
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

  draw_walls(env);
  draw_player(env);
  draw_range_finders(env);

  EndDrawing();
}

void c_close(HardMaze *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
