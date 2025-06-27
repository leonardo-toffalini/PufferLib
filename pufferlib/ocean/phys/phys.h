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

const Color PLANE_COLOR = (Color){30, 30, 30, 255};

typedef struct Log Log;
struct Log {
  float episode_return;
  float episode_length;
  float score;
  float perf;
  float n;
};

typedef struct {
  float w, x, y, z;
} Quat;

typedef struct {
  float x, y, z;
} Vec3;

static inline float clampf(float v, float min, float max) {
  return fmin(fmax(v, min), max);
}

static inline float rndf(float a, float b) {
  return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline Vec3 add3(Vec3 a, Vec3 b) {
  return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 sub3(Vec3 a, Vec3 b) {
  return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3 scalmul3(Vec3 a, float b) {
  return (Vec3){a.x * b, a.y * b, a.z * b};
}

static inline float dot3(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

static inline Quat quat_mul(Quat q1, Quat q2) {
  Quat out;
  out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
  out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
  out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
  out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
  return out;
}

static inline void quat_normalize(Quat *q) {
  float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
  if (n > 0.0f) {
    q->w /= n;
    q->x /= n;
    q->y /= n;
    q->z /= n;
  }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
  Quat qv = {0.0f, v.x, v.y, v.z};
  Quat tmp = quat_mul(q, qv);
  Quat q_conj = {q.w, -q.x, -q.y, -q.z};
  Quat res = quat_mul(tmp, q_conj);
  return (Vec3){res.x, res.y, res.z};
}

static inline Quat quat_inverse(Quat q) {
  return (Quat){q.w, -q.x, -q.y, -q.z};
}

Quat rndquat() {
  float u1 = rndf(0.0f, 1.0f);
  float u2 = rndf(0.0f, 1.0f);
  float u3 = rndf(0.0f, 1.0f);

  float sqrt_1_minus_u1 = sqrtf(1.0f - u1);
  float sqrt_u1 = sqrtf(u1);

  float pi_2_u2 = 2.0f * M_PI * u2;
  float pi_2_u3 = 2.0f * M_PI * u3;

  Quat q;
  q.w = sqrt_1_minus_u1 * sinf(pi_2_u2);
  q.x = sqrt_1_minus_u1 * cosf(pi_2_u2);
  q.y = sqrt_u1 * sinf(pi_2_u3);
  q.z = sqrt_u1 * cosf(pi_2_u3);

  return q;
}

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

  // Trailing path buffer (for rendering only)
  Vec3 trail[TRAIL_LENGTH];
  int trail_index;
  int trail_count;
};

typedef struct Phys Phys;
struct Phys {
  float *observations;
  float *actions;
  float *rewards;
  unsigned char *terminals;

  Log log;
  int tick;
  int report_interval;
  int score;
  float episodic_return;

  int max_rings;
  int ring_idx;

  int max_moves;
  int moves_left;

  Vec3 pos; // global position (x, y, z)
  Vec3 prev_pos;
  Vec3 vel;   // linear velocity (u, v, w)
  Quat quat;  // roll/pitch/yaw (phi/theta/psi) as a quaternion
  Vec3 omega; // angular velocity (p, q, r)

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
  env->log.episode_return += env->episodic_return;
  env->log.episode_length += env->tick;
  env->log.perf += (float)env->ring_idx / (float)env->max_rings;
  env->log.n += 1.0f;
}

void compute_observations(Phys *env) {}

void c_reset(Phys *env) {
  env->tick = 0;
  env->score = 0;
  env->episodic_return = 0.0f;
}

void c_step(Phys *env) {
  env->tick += 1;
  env->rewards[0] = 0;
  env->terminals[0] = 0;
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
        clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

    client->last_mouse_pos = mouse_pos;

    update_camera_position(client);
  }

  float wheel = GetMouseWheelMove();
  if (wheel != 0) {
    client->camera_distance -= wheel * 2.0f;
    client->camera_distance = clampf(client->camera_distance, 5.0f, 50.0f);
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

  // Initialize trail buffer
  client->trail_index = 0;
  client->trail_count = 0;
  for (int i = 0; i < TRAIL_LENGTH; i++) {
    client->trail[i] = env->pos;
  }

  return client;
}

void DrawHorizontalPlane(Vector3 centerPos, Vector2 size, Color color) {
  rlPushMatrix();
  rlTranslatef(centerPos.x, centerPos.y, centerPos.z);
  rlRotatef(90, 1, 0, 0); // Rotate 90 degrees around X axis
  DrawPlane((Vector3){0, 0, 0}, size, color);
  rlPopMatrix();
}

// Draw a grid centered at (0, 0, z) in the XY plane
void DrawGridXY(int slices, float spacing, float z, Color color) {
  int halfSlices = slices / 2;
  rlBegin(RL_LINES);
  for (int i = -halfSlices; i <= halfSlices; i++) {
    // Optionally use a different color for the center lines
    Color lineColor = color;
    rlColor4ub(lineColor.r, lineColor.g, lineColor.b, lineColor.a);

    // Vertical lines (constant x, varying y)
    rlVertex3f((float)i * spacing, (float)-halfSlices * spacing, z);
    rlVertex3f((float)i * spacing, (float)halfSlices * spacing, z);

    // Horizontal lines (constant y, varying x)
    rlVertex3f((float)-halfSlices * spacing, (float)i * spacing, z);
    rlVertex3f((float)halfSlices * spacing, (float)i * spacing, z);
  }
  rlEnd();
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
  client->trail[client->trail_index] = env->pos;
  client->trail_index = (client->trail_index + 1) % TRAIL_LENGTH;
  if (client->trail_count < TRAIL_LENGTH)
    client->trail_count++;

  BeginDrawing();
  ClearBackground(BLACK);

  BeginMode3D(client->camera);

  DrawHorizontalPlane((Vector3){0, 0, -GRID_SIZE}, (Vector2){40, 40},
                      PLANE_COLOR);
  DrawGridXY(20, 2.0f, -GRID_SIZE + 0.01f, RAYWHITE);

  EndMode3D();

  DrawText("Left click + drag: Rotate camera", 10, 10, 20, LIGHTGRAY);
  DrawText("Mouse wheel: Zoom in/out", 10, 40, 20, LIGHTGRAY);

  EndDrawing();
}
