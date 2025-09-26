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
  env->price_window_size = unpack(kwargs, "price_window_size");
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
  assign_to_dict(dict, "terminal_risky", log->terminal_risky);
  assign_to_dict(dict, "terminal_riskless", log->terminal_riskless);
  assign_to_dict(dict, "terminal_price", log->terminal_price);
  assign_to_dict(dict, "pre_terminal_risky", log->pre_terminal_risky);
  assign_to_dict(dict, "pre_terminal_riskless", log->pre_terminal_riskless);
  assign_to_dict(dict, "liquidation_steps", log->liquidation_steps);
  assign_to_dict(dict, "liquidation_cost", log->liquidation_cost);
  assign_to_dict(dict, "liquidation_action", log->liquidation_action);
  return 0;
}
