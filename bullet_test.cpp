#include <btBulletDynamicsCommon.h>

int main() {
  // Set up the basic components for a Bullet physics world
  btDefaultCollisionConfiguration *collisionConfig =
      new btDefaultCollisionConfiguration();
  btCollisionDispatcher *dispatcher =
      new btCollisionDispatcher(collisionConfig);
  btBroadphaseInterface *broadphase = new btDbvtBroadphase();
  btSequentialImpulseConstraintSolver *solver =
      new btSequentialImpulseConstraintSolver();
  btDiscreteDynamicsWorld *dynamicsWorld = new btDiscreteDynamicsWorld(
      dispatcher, broadphase, solver, collisionConfig);

  // Set gravity (optional)
  dynamicsWorld->setGravity(btVector3(0, -9.81, 0));

  // Clean up
  delete dynamicsWorld;
  delete solver;
  delete broadphase;
  delete dispatcher;
  delete collisionConfig;

  return 0;
}
