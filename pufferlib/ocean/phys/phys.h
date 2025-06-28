// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/stmio/drone

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "raylib.h"
#include "rlgl.h"

#include "core.h"

// Visualisation properties
#define WIDTH 1080
#define HEIGHT 720
#define TRAIL_LENGTH 50

// Simulation properties
#define GRID_SIZE 10.0f
#define RING_RAD 2.0f
#define RING_MARGIN 4.0f
#define DT 0.02f

// Physical constants for the drone
#define MASS 1.0f       // kg
#define IXX 0.01f       // kgm^2
#define IYY 0.01f       // kgm^2
#define IZZ 0.02f       // kgm^2
#define ARM_LEN 0.1f    // m
#define K_THRUST 3e-5f  // thrust coefficient
#define K_ANG_DAMP 0.2f // angular damping coefficient
#define K_DRAG 1e-6f    // drag (torque) coefficient
#define B_DRAG 0.1f     // linear drag coefficient
#define GRAVITY 9.81f   // m/s^2
#define MAX_RPM 750.0f  // rad/s
#define MAX_VEL 50.0f   // m/s
#define MAX_OMEGA 50.0f // rad/s

const Color PLANE_COLOR = (Color){35, 35, 35, 255};

typedef struct Log Log;
struct Log {
  float episode_return;
  float episode_length;
  float score;
  float perf;
  float n;
};

typedef struct Client Client;
struct Client {
  Camera3D camera;
  float width;
  float height;

  float camera_distance;
  float camera_azimuth;
  float camera_elevation;
  bool is_dragging;
  Vector2 last_mouse_pos;
};

typedef struct Phys Phys;
struct Phys {
  float *observations;
  float *actions;
  float *rewards;
  unsigned char *terminals;

  Log log;
  int tick;
  int score;

  RigidBody body;

  Client *client;
};

void init(Phys *env) {
  env->log = (Log){0};
  env->tick = 0;
  // one extra ring for observation (requires current ring, next ring)
  // max_rings and moves_left are initialised in binding.c
}

void add_log(Phys *env) {
  env->log.score += env->score;
  env->log.episode_length += env->tick;
  env->log.n += 1.0f;
}

void compute_observations(Phys *env) {}

void c_reset(Phys *env) {
  env->tick = 0;
  env->score = 0;
  env->body = create_cuboid();
}

void c_step(Phys *env) {
  env->tick += 1;
  env->rewards[0] = 0;
  env->terminals[0] = 0;

  step_simulation(&env->body);
}

void c_close_client(Client *client) {
  CloseWindow();
  free(client);
}

void c_close(Phys *env) {
  if (env->client != NULL) {
    c_close_client(env->client);
  }
}

static void update_camera_position(Client *c) {
  float r = c->camera_distance;
  float az = c->camera_azimuth;
  float el = c->camera_elevation;

  float x = r * cosf(el) * cosf(az);
  float y = r * cosf(el) * sinf(az);
  float z = r * sinf(el);

  c->camera.position = (Vector3){x, y, z};
  c->camera.target = (Vector3){0, 0, 0};
}

void handle_camera_controls(Client *client) {
  Vector2 mouse_pos = GetMousePosition();

  if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
    client->is_dragging = true;
    client->last_mouse_pos = mouse_pos;
  }

  if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
    client->is_dragging = false;
  }

  if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                           mouse_pos.y - client->last_mouse_pos.y};

    float sensitivity = 0.005f;

    client->camera_azimuth -= mouse_delta.x * sensitivity;

    client->camera_elevation += mouse_delta.y * sensitivity;
    client->camera_elevation =
        Clamp(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

    client->last_mouse_pos = mouse_pos;

    update_camera_position(client);
  }

  float wheel = GetMouseWheelMove();
  if (wheel != 0) {
    client->camera_distance -= wheel * 2.0f;
    client->camera_distance = Clamp(client->camera_distance, 5.0f, 50.0f);
    update_camera_position(client);
  }
}

Client *make_client(Phys *env) {
  Client *client = (Client *)calloc(1, sizeof(Client));

  client->width = WIDTH;
  client->height = HEIGHT;

  SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
  InitWindow(WIDTH, HEIGHT, "PufferLib Phys");

#ifndef __EMSCRIPTEN__
  SetTargetFPS(60);
#endif

  if (!IsWindowReady()) {
    TraceLog(LOG_ERROR, "Window failed to initialize\n");
    free(client);
    return NULL;
  }

  client->camera_distance = 40.0f;
  client->camera_azimuth = 0.0f;
  client->camera_elevation = PI / 10.0f;
  client->is_dragging = false;
  client->last_mouse_pos = (Vector2){0.0f, 0.0f};

  client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
  client->camera.fovy = 45.0f;
  client->camera.projection = CAMERA_PERSPECTIVE;

  update_camera_position(client);

  return client;
}

void c_render(Phys *env) {
  if (env->client == NULL) {
    env->client = make_client(env);
    if (env->client == NULL) {
      TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
      return;
    }
  }

  if (WindowShouldClose()) {
    c_close(env);
    exit(0);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    c_close(env);
    exit(0);
  }

  handle_camera_controls(env->client);

  Client *client = env->client;

  BeginDrawing();
  ClearBackground(BLACK);

  BeginMode3D(client->camera);

  DrawHorizontalPlane((Vector3){0, 0, -10}, (Vector2){40, 40}, PLANE_COLOR);
  DrawGridXY(20, 2.0f, -10 + 0.01f, RAYWHITE);

  draw_body(&env->body);

  // void DrawCubeWires(Vector3 position, float width, float height, float
  // length, Color color);

  EndMode3D();

  DrawText("Left click + drag: Rotate camera", 10, 10, 20, LIGHTGRAY);
  DrawText("Mouse wheel: Zoom in/out", 10, 40, 20, LIGHTGRAY);

  EndDrawing();
}
