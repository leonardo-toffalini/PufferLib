/*
 action: Discrete(4)
 - 0 => noop
 - 1 => turn left
 - 2 => go forward
 - 3 => turn right

 observation: Box(0, 1, shape=(2 + 1 + num_range_finders))
 - 2 => for player position normalized
 - 1 => for radar reading, meaning which radar sees the objective (if any)
 - num_range_finders => for the proportional length of each range finder

 rewards:
 +1 for reaching the goal
 -0.1 otherwise for every step
*/

#include "raylib.h"
#include "raymath.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const unsigned char NOOP = 0;
const unsigned char LEFT = 1;
const unsigned char FORWARD = 2;
const unsigned char RIGHT = 3;

const int DEBUG = 0;

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
  float *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int tick;

  // env specific
  // user settable
  float linear_speed;
  float angular_speed;
  float radar_range;
  float range_finder_len;
  int frame_skip;
  int max_ticks;

  int size;
  Wall walls[MAX_WALLS];
  Player player;
  RangeFinder range_finders[NUM_RANGE_FINDERS];
  Vector2 goal;
  Vector2 pois[10];
  int radar_reading;
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
    env->range_finders[i] = (RangeFinder){i * delta_angle, env->range_finder_len, 1.0f};
  }
}

void set_up_pois(HardMaze *env) {
  int idx = 0;

  env->pois[idx++] = (Vector2){320, 450};
  env->pois[idx++] = (Vector2){480, 530};
  env->pois[idx++] = (Vector2){480, 420};
  env->pois[idx++] = (Vector2){300, 300};
  env->pois[idx++] = (Vector2){350, 200};
  env->pois[idx++] = (Vector2){400, 100};
  env->pois[idx++] = (Vector2){250, 115};
}

void c_reset(HardMaze *env) {
  set_up_walls(env);
  set_up_range_finders(env);
  set_up_pois(env);

  Player player = {
      (Vector2){100.0f, 540.0f}, (Vector2){0.0f, -1.0f}, PI / 2, 10.0f, 100.0f,
  };
  env->player = player;
  env->tick = 0;
  env->goal = (Vector2){120, 120};
  env->radar_reading = -1;
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
  Vector2 collision_point, direction, end, rf_base;
  Vector2 base_vec = (Vector2){0.0f, 1.0f};
  Vector2 center = env->player.pos;
  float dist;

  for (int i = 0; i < NUM_RANGE_FINDERS; i++) {
    env->range_finders[i].distance = 1.0f;
    rf = env->range_finders[i];

    for (int j = 0; j < MAX_WALLS; j++) {
      w = env->walls[j];

      direction = Vector2Rotate(base_vec, rf.angle + env->player.angle);
      rf_base = Vector2Add(center, Vector2Scale(direction, env->player.radius));

      end = Vector2Add(rf_base, Vector2Scale(direction, rf.max_range));
      if (!CheckCollisionLines(rf_base, end, (Vector2){w.ax, w.ay},
                               (Vector2){w.bx, w.by}, &collision_point))
        continue;

      dist = Vector2Distance(collision_point, rf_base) / rf.max_range;
      if (dist <= rf.distance) {
        env->range_finders[i].distance = dist;
      }
    }
  }
}

void update_radars(HardMaze *env) {
  env->radar_reading = -1;
  float dist = Vector2Distance(env->player.pos, env->goal);
  if (dist < env->radar_range) {
    float angle = Vector2Angle(env->player.forward,
                               Vector2Subtract(env->goal, env->player.pos));
    if (-3.0f * PI / 4.0f <= angle && angle < -PI / 4.0f)
      env->radar_reading = 1;
    else if (-PI / 4.0f <= angle && angle < PI / 4.0f)
      env->radar_reading = 2;
    else if (PI / 4.0f <= angle && angle < 3.0f * PI / 4.0f)
      env->radar_reading = 3;
    else
      env->radar_reading = 0;
  }
}

void check_reached_goal(HardMaze *env) {
  if (Vector2Distance(env->player.pos, env->goal) < env->player.radius) {
    env->terminals[0] = 1;
    env->rewards[0] = 1;
  }
}

void execute_action(HardMaze *env, int action) {
  Vector2 prev_pos = env->player.pos;

  switch (action) {
  case LEFT:
    env->player.forward = Vector2Rotate(env->player.forward, -env->angular_speed);
    env->player.angle -= env->angular_speed;
    break;
  case RIGHT:
    env->player.forward = Vector2Rotate(env->player.forward, env->angular_speed);
    env->player.angle += env->angular_speed;
    break;
  case FORWARD:
    env->player.pos = Vector2Add(
        env->player.pos, Vector2Scale(env->player.forward, env->linear_speed));
  default:
    break;
  }

  if (check_collisions(env)) {
    env->player.pos = prev_pos;
  }

  update_range_finders(env);
  update_radars(env);
  check_reached_goal(env);
}

void compute_observations(HardMaze *env) {
  int obs_idx = 0;

  // player position normalized
  env->observations[obs_idx++] = env->player.pos.x / 640.0f;
  env->observations[obs_idx++] = env->player.pos.y / 640.0f;

  // range finder readings
  for (int i = 0; i < NUM_RANGE_FINDERS; i++) {
    env->observations[obs_idx++] = env->range_finders[i].distance;
  }

  // radar reading
  for (int i = 0; i < 4; i++) {
    env->observations[obs_idx++] = env->radar_reading == i ? 1 : 0;
  }
}

void print_observations(HardMaze *env) {
  printf("obs = [");
  for (int i = 0; i < 11; i++) {
    printf("%.4f, ", env->observations[i]);
  }
  printf("]\n");
}

void c_step(HardMaze *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = -0.1f; // small penalty every step

  for (int i = 0; i < env->frame_skip; i++)
    execute_action(env, action);

  compute_observations(env);

  if (env->terminals[0] || env->tick > env->max_ticks) {
    add_log(env);
    c_reset(env);
    return;
  }
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
  Vector2 rf_base;
  RangeFinder rf;
  for (int i = 0; i < NUM_RANGE_FINDERS; i++) {
    rf = env->range_finders[i];
    direction = Vector2Rotate(base_vec, rf.angle + env->player.angle);
    rf_base = Vector2Add(center, Vector2Scale(direction, env->player.radius));
    DrawLineEx(
      rf_base,
      Vector2Add(
        rf_base,
        Vector2Scale(
          direction,
          rf.distance * rf.max_range
        )
      ),
      2, GREEN);
  }
}

void draw_radars(HardMaze *env) {
  Vector2 center = env->player.pos;
  Vector2 base_vec = (Vector2){0.0f, -1.0f};
  Vector2 direction;
  float player_angle = 180 * env->player.angle / PI;
  Color c;

  for (int i = 0; i < 4; i++) {
    c = i == env->radar_reading ? ColorAlpha(RED, 0.4f)
                                : ColorAlpha(GRAY, 0.2f);
    DrawRing(center, env->player.radius, env->radar_range,
             player_angle + i * 90.0f - 45.0f,
             player_angle + (i + 1) * 90.0f - 45.0f, 32, c);

    if (DEBUG) {
      direction = Vector2Rotate(base_vec,
                                i * PI / 2.0f + env->player.angle - PI / 4.0f);
      DrawLineEx(
          Vector2Add(center, Vector2Scale(direction, env->player.radius)),
          Vector2Add(center, Vector2Scale(direction, env->radar_range)), 2,
          ColorAlpha(RED, 0.2f));
    }
  }
}

void draw_pois(HardMaze *env) {
  int r = 5;
  for (int i = 0; i < 10; i++) {
    DrawRing(env->pois[i], r - 2, r, 0, 360, 64, BLACK);
    DrawCircleV(env->pois[i], r - 2, BLUE);
  }
}

void draw_goal(HardMaze *env) {
  int r = 5;
  DrawRing(env->goal, r - 2, r, 0, 360, 64, BLACK);
  DrawCircleV(env->goal, r - 2, GREEN);
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

  if (DEBUG)
    print_observations(env);

  draw_walls(env);
  draw_range_finders(env);
  draw_radars(env);
  draw_pois(env);
  draw_goal(env);
  draw_player(env);

  EndDrawing();
}

void c_close(HardMaze *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
