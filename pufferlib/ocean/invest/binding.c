#include "invest.h"

#define Env Invest
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->T = unpack(kwargs, "time_horizon");
  env->H = unpack(kwargs, "hurst");
  env->process_type = unpack(kwargs, "process_type");
  env->liquidate = unpack(kwargs, "liquidate");
  env->friction_coef = unpack(kwargs, "friction_coef");
  env->friction_power = unpack(kwargs, "friction_power");
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
  return 0;
}
