#pragma once

#include "ray_helpers.h"
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include <math.h>
#include <stdio.h>

const float DT = 0.01f;
const Vector3 IHAT = (Vector3){1.0f, 0.0f, 0.0f};
const Vector3 JHAT = (Vector3){0.0f, 1.0f, 0.0f};
const Vector3 KHAT = (Vector3){0.0f, 0.0f, 1.0f};
const float SIDE_A = 1.0f;
const float SIDE_B = 2.0f;
const float SIDE_C = 3.0f;

typedef struct {
  Vector3 X;
  Vector3 V;
  Matrix R;
  Vector3 L;
  Vector3 Omega;
  Vector3 InvI0;
} RigidBody;

static Matrix DiagonalMatrixV(Vector3 v) { return MatrixScale(v.x, v.y, v.z); }
static Matrix DiagonalMatrix(float x, float y, float z) {
  return MatrixScale(x, y, z);
}

static Matrix Cross(Vector3 v) {
  // clang-format off
  Matrix result = {
    0.0f, -v.z, v.y, 0.0f,
    v.z, 0.0f, -v.x, 0.0f,
    -v.y, v.x, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f
  };
  return result;
}

static Matrix MatrixFloatMultiply(Matrix m, float c) {
  return MatrixMultiply(m, MatrixScale(c, c, c));
}

void print_matrix(Matrix mat) {
    printf("[ %f %f %f %f ]\n", mat.m0,  mat.m4,  mat.m8,  mat.m12);
    printf("[ %f %f %f %f ]\n", mat.m1,  mat.m5,  mat.m9,  mat.m13);
    printf("[ %f %f %f %f ]\n", mat.m2,  mat.m6,  mat.m10, mat.m14);
    printf("[ %f %f %f %f ]\n\n", mat.m3,  mat.m7,  mat.m11, mat.m15);
}

static Vector3 inertia_cuboid_density(float a, float b, float c) {
  float V = a * b * c;
  Vector3 D = (Vector3){b * b + c * c, a * a + c * c, a * a + b * b};
  return Vector3Scale(D, V / 12);
}

RigidBody create_cuboid(void) {
  Vector3 X = {0.0f, 0.0f, 0.0f};
  Vector3 V = {0.4f, 0.5f, 0.1f};
  Matrix R = MatrixIdentity();
  Vector3 L = {2.0f, 0.5f, 0.0f};
  Vector3 Omega = {0.0f, 0.0f, 0.0f};
  Vector3 InvI0 = Vector3Invert(inertia_cuboid_density(SIDE_A, SIDE_B, SIDE_C));

  return (RigidBody){X, V, R, L, Omega, InvI0};
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
  // Vector3 omega = body->Omega;
  body->Omega = omega;
  body->R = MatrixAdd(body->R, MatrixFloatMultiply(MatrixMultiply(Cross(omega), R), DT));
  body->X = Vector3Add(body->X, Vector3Scale(body->V, DT));
}

void draw_body(RigidBody *body) {
  Vector3 zero = Vector3Zero();
  print_matrix(body->R);
  printf("det(R) = %f\n", MatrixDeterminant(body->R));

  DrawArrow3D(body->X, Vector3Add(body->X, body->L), YELLOW);
  DrawArrow3D(body->X, Vector3Add(body->X, body->V), MAGENTA);

  rlPushMatrix();
  rlTranslatef(body->X.x, body->X.y, body->X.z);
  rlMultMatrixf((float *)&body->R);
  DrawArrow3D(zero, IHAT, RED);
  DrawArrow3D(zero, JHAT, GREEN);
  DrawArrow3D(zero, KHAT, BLUE);
  DrawCube(zero, 1, 2, 3, ColorAlpha(RAYWHITE, 0.5f));
  DrawCubeWires(Vector3Zero(), SIDE_A, SIDE_B, SIDE_C, RED);
  rlPopMatrix();
}

