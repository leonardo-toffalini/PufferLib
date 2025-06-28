#pragma once

#include "ray_helpers.h"
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <math.h>

const float DT = 0.001f;
const Vector3 IHAT = (Vector3){1.0f, 0.0f, 0.0f};
const Vector3 JHAT = (Vector3){0.0f, 1.0f, 0.0f};
const Vector3 KHAT = (Vector3){0.0f, 0.0f, 1.0f};

typedef struct {
  Vector3 X;
  Vector3 V;
  Matrix R;
  Vector3 L;

  Vector3 InvI0;
} RigidBody;

static Matrix DiagonalMatrixV(Vector3 v) { return MatrixScale(v.x, v.y, v.z); }
static Matrix DiagonalMatrix(float x, float y, float z) {
  return MatrixScale(x, y, z);
}

static Matrix Cross(Vector3 v) {
  // clang-format off
  Matrix result = {
    0.0f, -v.z, v.y,
    v.z, 0.0f, -v.x,
    -v.y, v.x, 0.0f,
    0.0f, 0.0f, 1.0f,
  };
  return result;
}

static Matrix MatrixFloatMultiply(Matrix m, float c) {
  return MatrixMultiply(m, MatrixScale(c, c, c));
}

static Vector3 inertia_cuboid_density(float a, float b, float c) {
  float V = a * b * c;
  Vector3 D = (Vector3){b * b + c * c, a * a + c * c, a * a + b * b};
  return Vector3Scale(D, V / 12);
}

RigidBody create_cuboid(void) {
  Vector3 X = {0.0f, 0.0f, 0.0f};
  Vector3 V = {0.8f, 0.2f, 0.3f};
  Matrix R = MatrixIdentity();
  Vector3 L = {1.0f, 1.0f, 0.0f};

  Vector3 InvI0 = Vector3Invert(inertia_cuboid_density(1.4f, 0.7f, 2.1f));

  return (RigidBody){X, V, R, L, InvI0};
}

void step_simulation(RigidBody *body) {
  Matrix R = body->R;
  Matrix InvI0 = DiagonalMatrixV(body->InvI0);

  Vector3 omega = Vector3Transform(
    body->L,
    MatrixMultiply(
      MatrixMultiply(R, InvI0),
      MatrixTranspose(R)
    )
  );
  body->R = MatrixAdd(MatrixFloatMultiply(MatrixMultiply(Cross(omega), R), DT), body->R);
  body->X = Vector3Add(body->X, Vector3Scale(body->V, DT));
}

void draw_body(RigidBody *body) {
  Vector3 zero = Vector3Zero();

  rlPushMatrix();
  rlTranslatef(body->X.x, body->X.y, body->X.z);
  rlMultMatrixf((float *)&body->R);
  DrawArrow3D(zero, IHAT, RED);
  DrawArrow3D(zero, JHAT, GREEN);
  DrawArrow3D(zero, KHAT, BLUE);
  DrawArrow3D(zero, body->V, MAGENTA);
  DrawCube(Vector3Zero(), 1, 2, 3, ColorAlpha(RAYWHITE, 0.5f));
  DrawCubeWires(Vector3Zero(), 1, 2, 3, RED);
  rlPopMatrix();
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
