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
#include <math.h>

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

// ODE for double cart pole (C version of dpc_ode from test.py)
void dcp_ode(float t, const float *x, float *dx, float force) {
    float q0 = x[0];
    float q1 = x[1];
    float q2 = x[2];
    float q0_dot = x[3];
    float q1_dot = x[4];
    float q2_dot = x[5];
    float f = force;

    // Derivatives of positions are velocities
    dx[0] = q0_dot;
    dx[1] = q1_dot;
    dx[2] = q2_dot;

    // Accelerations (copied from Python, using math.h)
    dx[3] = (-4.0f*f*cosf(2.0f*q2)+6.0f*f+
             4.0f*q1_dot*q1_dot*cosf(q1)+
             q1_dot*q1_dot*cosf(q1-q2)-
             q1_dot*q1_dot*cosf(q1+2.0f*q2)+
             2.0f*q1_dot*q2_dot*cosf(q1-q2)+
             q2_dot*q2_dot*cosf(q1-q2)-
             29.43f*sinf(2.0f*q1)+
             9.81f*sinf(2.0f*q1+2.0f*q2)) /
            (3.0f*cosf(2.0f*q1)-22.0f*cosf(2.0f*q2)-
             cosf(2.0f*q1+2.0f*q2)+34.0f);
    dx[4] = (8.0f*f*sinf(q1)-4.0f*f*sinf(q1+2.0f*q2)+
             3.0f*q1_dot*q1_dot*sinf(2.0f*q1)+
             23.0f*q1_dot*q1_dot*sinf(q2)+
             22.0f*q1_dot*q1_dot*sinf(2.0f*q2)+
             q1_dot*q1_dot*sinf(2.0f*q1+q2)+
             46.0f*q1_dot*q2_dot*sinf(q2)+
             2.0f*q1_dot*q2_dot*sinf(2.0f*q1+q2)+
             23.0f*q2_dot*q2_dot*sinf(q2)+
             q2_dot*q2_dot*sinf(2.0f*q1+q2)-
             490.5f*cosf(q1)+
             215.82f*cosf(q1+2.0f*q2)) /
            (3.0f*cosf(2.0f*q1)-
             22.0f*cosf(2.0f*q2)-
             cosf(2.0f*q1+2.0f*q2)+34.0f);
    dx[5] = -((100.0f*q1_dot*q1_dot*sinf(q2)+
               981.0f*cosf(q1+q2)) *
              (-(3.0f*sinf(q1)+
                 sinf(q1+q2))*(3.0f*sinf(q1)+sinf(q1+q2))+ // (-(3*sin(q1)+sin(q1+q2))**2)
               28.0f*cosf(q2)+42.0f)+
             0.5f*(200.0f*q1_dot*q2_dot*sinf(q2)+
                  100.0f*q2_dot*q2_dot*sinf(q2)-
                  2943.0f*cosf(q1)-
                  981.0f*cosf(q1+q2))*
             (25.0f*cosf(q2)+3.0f*cosf(2.0f*q1+q2)+
              cosf(2.0f*q1+2.0f*q2)+13.0f)+
              50.0f*(2.0f*sinf(q1)+3.0f*sinf(q1-q2)-
              2.0f*sinf(q1+q2)-sinf(q1+2.0f*q2))*
             (2.0f*f+3.0f*q1_dot*q1_dot*cosf(q1)+
              q1_dot*q1_dot*cosf(q1+q2)+
              2.0f*q1_dot*q2_dot*cosf(q1+q2)+
              q2_dot*q2_dot*cosf(q1+q2))) /
            (75.0f*cosf(2.0f*q1)-550.0f*cosf(2.0f*q2)-
             25.0f*cosf(2.0f*q1+2.0f*q2)+850.0f);
}

// Symplectic Euler stepper
void symplectic_euler_step(float t, float *x, float dt, float force) {
    float dx[6];
    // Compute derivatives at current state
    dcp_ode(t, x, dx, force);
    // Update velocities
    x[3] += dt * dx[3];
    x[4] += dt * dx[4];
    x[5] += dt * dx[5];
    // Update positions with new velocities
    x[0] += dt * x[3];
    x[1] += dt * x[4];
    x[2] += dt * x[5];
}

void c_step(Dcp *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  env->rewards[0] = 0;

  // Map action to force (example: -1, 0, 1)
  float force = 0.0f;
  // if (action == LEFT) force = -1.0f;
  // else if (action == RIGHT) force = 1.0f;

  // Prepare state vector
  float x[6] = {env->q0, env->q1, env->q2, env->q0_dot, env->q1_dot, env->q2_dot};
  float dt = 0.025f; // Same as Python
  float t = env->tick * dt;
  // Step the system
  symplectic_euler_step(t, x, dt, force);
  // Update env state
  env->q0 = x[0];
  env->q1 = x[1];
  env->q2 = x[2];
  env->q0_dot = x[3];
  env->q1_dot = x[4];
  env->q2_dot = x[5];

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
