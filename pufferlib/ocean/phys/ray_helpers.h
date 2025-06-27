#pragma once

#include "raylib.h"
#include "raymath.h"

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

void DrawArrow3D(Vector3 start, Vector3 end, Color color) {
  Vector3 direction = Vector3Normalize(Vector3Subtract(end, start));
  float arrowLength = Vector3Distance(start, end);
  float shaftLength = arrowLength * 0.8f; // 80% shaft, 20% head
  float shaftRadius = 0.05f;
  float headLength = arrowLength * 0.2f;
  float headRadius = 0.12f;
  Vector3 shaftEnd = Vector3Add(start, Vector3Scale(direction, shaftLength));
  DrawCylinderEx(start, shaftEnd, shaftRadius, shaftRadius, shaftLength, color);
  DrawCylinderEx(shaftEnd, end, headRadius, 0.0f, headLength, color);
}
