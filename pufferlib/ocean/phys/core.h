#pragma once

#include "ray_helpers.h"
#include "raylib.h"
#include "raymath.h"
#include <math.h>

const float DT = 0.01;
const Vector3 IHAT = (Vector3){1.0f, 0.0f, 0.0f};
const Vector3 JHAT = (Vector3){0.0f, 1.0f, 0.0f};
const Vector3 KHAT = (Vector3){0.0f, 0.0f, 1.0f};

typedef struct {
  Vector3 X;
  Vector3 V;
} RigidBody;

RigidBody create_cuboid() {
  Vector3 X = {0.0f, 0.0f, 0.0f};
  Vector3 V = {1.0f, 2.5f, 1.2f};
  return (RigidBody){X, V};
}

void step_simulation(RigidBody *body) {
  body->X = Vector3Add(body->X, Vector3Scale(body->V, 0.01f));
}

void draw_body(RigidBody *body) {
  DrawArrow3D(body->X, Vector3Add(body->X, IHAT), RED);
  DrawArrow3D(body->X, Vector3Add(body->X, JHAT), GREEN);
  DrawArrow3D(body->X, Vector3Add(body->X, KHAT), BLUE);
  DrawArrow3D(body->X, Vector3Add(body->X, body->V), MAGENTA);
  DrawCube(body->X, 1, 2, 3, ColorAlpha(RAYWHITE, 0.5f));
  DrawCubeWires(body->X, 1, 2, 3, RED);
}

/////////////////////////////////////
// Residual from drone.h
/////////////////////////////////////

typedef struct {
  float w, x, y, z;
} Quat;

typedef struct {
  float x, y, z;
} Vec3;

inline float clampf(float v, float min, float max) {
  return fmin(fmax(v, min), max);
}

inline float rndf(float a, float b) {
  return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

inline Vec3 add3(Vec3 a, Vec3 b) {
  return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 sub3(Vec3 a, Vec3 b) {
  return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 scalmul3(Vec3 a, float b) {
  return (Vec3){a.x * b, a.y * b, a.z * b};
}

inline float dot3(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

inline Quat quat_mul(Quat q1, Quat q2) {
  Quat out;
  out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
  out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
  out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
  out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
  return out;
}

inline void quat_normalize(Quat *q) {
  float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
  if (n > 0.0f) {
    q->w /= n;
    q->x /= n;
    q->y /= n;
    q->z /= n;
  }
}

inline Vec3 quat_rotate(Quat q, Vec3 v) {
  Quat qv = {0.0f, v.x, v.y, v.z};
  Quat tmp = quat_mul(q, qv);
  Quat q_conj = {q.w, -q.x, -q.y, -q.z};
  Quat res = quat_mul(tmp, q_conj);
  return (Vec3){res.x, res.y, res.z};
}

inline Quat quat_inverse(Quat q) { return (Quat){q.w, -q.x, -q.y, -q.z}; }

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
