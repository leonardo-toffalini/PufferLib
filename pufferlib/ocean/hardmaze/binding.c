#include "hardmaze.h"

#define Env HardMaze
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->linear_speed = unpack(kwargs, "linear_speed");
  env->angular_speed = unpack(kwargs, "angular_speed");
  env->radar_range = unpack(kwargs, "radar_range");
  env->range_finder_len = unpack(kwargs, "range_finder_len");
  env->frame_skip = unpack(kwargs, "frame_skip");
  env->max_ticks = unpack(kwargs, "max_ticks");
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "perf", log->perf);
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
  return 0;
}
