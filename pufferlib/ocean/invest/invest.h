#include "fbm.h"
#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const float MAX_PRICE = 1.0f;
const float MAX_RISKY = 400.0f;
const float MAX_RISKLESS = 400.0f;

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

  // env specific
  // can be defined
  int T;
  float H;
  int process_type;
  int liquidate;
  int liq_type;
  float friction_coef;
  int friction_power;
  int price_window_size;

  float riskless;
  float risky;
  double *prices;
  float *riskless_history;
  float *risky_history;

  int tick;
} Invest;

void add_log(Invest *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0] > 0 ? 1 : 0;
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

double *sin_process(Invest *env) {
  double *process = malloc((2 * env->T + 1) * sizeof(double));
  for (int i = 0; i < 2 * env->T + 1; i++) {
    process[i] = sin(2 * PI * i / env->T);
  }
  return process;
}

void compute_observations(Invest *env) {
  int obs_idx = 0;
  env->observations[obs_idx++] = (float)env->tick / env->T;
  env->observations[obs_idx++] = env->prices[env->tick] / MAX_PRICE;
  env->observations[obs_idx++] = env->riskless / MAX_RISKLESS;
  env->observations[obs_idx++] = env->risky / MAX_RISKY;

  for (int i = 0; i < env->price_window_size; i++) {
    if (env->tick - i >= 0) {
      env->observations[obs_idx++] = env->prices[env->tick - i] / MAX_PRICE;
    } else {
      env->observations[obs_idx++] = 0;
    }
  }
}

// Required function
void c_reset(Invest *env) {
  env->tick = 0;
  env->riskless = 0;
  env->risky = 0;

  // Set default price_window_size if not already set
  if (env->price_window_size <= 0) {
    env->price_window_size = 32;
  }

  srand(time(NULL));

  // simulate_fBm(env->H, env->T, env->T);
  if (env->process_type == 0)
    env->prices = sin_process(env);
  else
    env->prices = simulate_fBm(env->H, 2 * env->T, 2 * env->T);

  if (env->riskless_history == NULL)
    env->riskless_history = malloc((2 * env->T + 1) * sizeof(float));
  if (env->risky_history == NULL)
    env->risky_history = malloc((2 * env->T + 1) * sizeof(float));

  memset(env->riskless_history, 0, sizeof(*env->riskless_history));
  memset(env->risky_history, 0, sizeof(*env->risky_history));

  compute_observations(env);
}

void execute_action(Invest *env, float action) {
  float price = 10.0f * env->prices[env->tick];
  env->risky += action;
  // env->riskless = env->riskless - action * price;
  env->riskless = env->riskless - action * price -
                  env->friction_coef * pow(fabsf(action), env->friction_power);

  env->riskless_history[env->tick] = env->riskless;
  env->risky_history[env->tick] = env->risky;
  env->rewards[0] = (env->riskless + price * env->riskless) / 100;
  env->tick += 1;
}

void liquidate(Invest *env, int liq_type) {
  float liquidation_step = -env->risky / (env->T + 1);
  switch (liq_type) {
  case 0:
    execute_action(env, -env->risky);
    break;
  case 1:
    while (env->tick <= 2 * env->T) {
      execute_action(env, liquidation_step);
    }
    break;
  default:
    execute_action(env, -env->risky);
    break;
  }
}

void c_step(Invest *env) {
  env->terminals[0] = 0;
  env->rewards[0] = 0;

  int action = env->actions[0] - 1;

  execute_action(env, action);

  if (env->tick >= env->T) {
    if (env->liquidate)
      liquidate(env, env->liq_type);
    // env->rewards[0] = env->riskless;
    env->terminals[0] = 1;
    add_log(env);
    c_reset(env);
  }

  compute_observations(env);
}

// Required function. Should handle creating the client on first call
void c_render(Invest *env) {
  static bool window_initialized = false;
  const int screen_width = 800;
  const int screen_height = 600;
  const int margin = 50;
  const int graph_width = screen_width - 2 * margin;
  const int graph_height = screen_height - 2 * margin;

  if (!window_initialized) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screen_width, screen_height, "Investment Simulation");
    SetTargetFPS(15);
    window_initialized = true;
  }

  if (!IsWindowReady())
    return;

  BeginDrawing();
  Color dark_bg = (Color){20, 20, 20, 255};
  ClearBackground(dark_bg);

  // if (env->rewards[0] != 0) {
  //   printf("reward (terminal riskless): %f\n", env->rewards[0]);
  // }

  // Draw axes
  Color axis_color = RAYWHITE;
  DrawLine(margin, screen_height - margin, screen_width - margin,
           screen_height - margin, axis_color);
  DrawLine(margin, margin, margin, screen_height - margin, axis_color);

  float max_price = 0;
  float max_assets = 0;
  float min_value = 0;
  float min_price = env->prices[0];
  float max_price_val = env->prices[0];
  for (int i = 0; i <= env->tick; i++) {
    max_price = fmaxf(max_price, env->prices[i]);
    max_assets = fmaxf(max_assets,
                       fmaxf(env->riskless_history[i], env->risky_history[i]));
    min_value = fminf(min_value,
                      fminf(env->riskless_history[i], env->risky_history[i]));
    if (env->prices[i] < min_price)
      min_price = env->prices[i];
    if (env->prices[i] > max_price_val)
      max_price_val = env->prices[i];
  }
  float max_value = fmaxf(max_price, max_assets);
  float value_range = max_value - min_value;
  if (value_range == 0)
    value_range = 1;

  // Define colors
  Color price_color = YELLOW;
  Color riskless_color = SKYBLUE;
  Color risky_color = LIME;

  // Draw price line (YELLOW), always in the middle half of the graph
  for (int i = 1; i <= env->tick; i++) {
    float x1 = margin + (i - 1) * graph_width / (float)env->T;
    float x2 = margin + i * graph_width / (float)env->T;

    float norm1 =
        (env->prices[i - 1] - min_price) / (max_price_val - min_price + 1e-8f);
    float norm2 =
        (env->prices[i] - min_price) / (max_price_val - min_price + 1e-8f);

    // Map to 25% - 75% of the graph height
    float y1 = screen_height - margin - (0.25f + 0.5f * norm1) * graph_height;
    float y2 = screen_height - margin - (0.25f + 0.5f * norm2) * graph_height;

    DrawLine(x1, y1, x2, y2, price_color);
  }

  // Draw riskless asset line (SKYBLUE)
  for (int i = 1; i < env->tick; i++) {
    float x1 = margin + (i - 1) * graph_width / (float)env->T;
    float y1 = screen_height - margin -
               ((env->riskless_history[i - 1] - min_value) / value_range) *
                   graph_height;
    float x2 = margin + i * graph_width / (float)env->T;
    float y2 =
        screen_height - margin -
        ((env->riskless_history[i] - min_value) / value_range) * graph_height;
    DrawLine(x1, y1, x2, y2, riskless_color);
  }

  // Draw risky asset line (LIME)
  for (int i = 1; i < env->tick; i++) {
    float x1 = margin + (i - 1) * graph_width / (float)env->T;
    float y1 =
        screen_height - margin -
        ((env->risky_history[i - 1] - min_value) / value_range) * graph_height;
    float x2 = margin + i * graph_width / (float)env->T;
    float y2 =
        screen_height - margin -
        ((env->risky_history[i] - min_value) / value_range) * graph_height;
    DrawLine(x1, y1, x2, y2, risky_color);
  }

  // Draw current values
  DrawText(TextFormat("Price: %.2f", env->prices[env->tick]),
           screen_width - 200, margin - 20, 20, price_color);
  DrawText(TextFormat("Riskless: %.2f", env->riskless), screen_width - 200,
           margin, 20, riskless_color);
  DrawText(TextFormat("Risky: %.2f", env->risky), screen_width - 200,
           margin + 20, 20, risky_color);

  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Invest *env) {
  free(env->prices);
  free(env->riskless_history);
  free(env->risky_history);
  if (IsWindowReady()) {
    CloseWindow();
  }
}
