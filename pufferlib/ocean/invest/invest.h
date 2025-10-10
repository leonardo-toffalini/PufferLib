#include "fbm.h"
#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <assert.h>

const float MAX_PRICE = 1.0f;
const float MAX_RISKY = 400.0f;
const float MAX_RISKLESS = 400.0f;

typedef struct {
  float perf;
  float score;
  float episode_return;
  float episode_length;
  float terminal_risky;
  float terminal_riskless;
  float terminal_price;
  float pre_terminal_risky;
  float pre_terminal_riskless;
  float liquidation_steps;
  float liquidation_cost;
  float liquidation_action;
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

  // For rendering liquidation steps
  int last_episode_length; // Total length including liquidation
  double *last_prices;
  float *last_riskless_history;
  float *last_risky_history;
  int render_frames_remaining;
  int history_capacity;
  int last_prices_capacity;
  int last_history_capacity;
} Invest;

void add_log(Invest *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0] > 0 ? 1 : 0;
  env->log.terminal_risky += env->risky;
  env->log.terminal_riskless += env->riskless;
  int max_idx = 2 * env->T;
  int idx = env->tick;
  if (idx > max_idx)
    idx = max_idx;
  if (idx < 0)
    idx = 0;
  env->log.terminal_price += env->prices[idx];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

void add_pre_terminal_log(Invest *env) {
  env->log.pre_terminal_risky = env->risky;
  env->log.pre_terminal_riskless = env->riskless;
}

// TODO:  try with random phase, random amplitude, and random number of peeks
double *sin_process(Invest *env) {
  double *process = (double *)malloc((2 * env->T + 1) * sizeof(double));
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

  // Initialize rendering fields on first reset
  static int first_reset = 1;
  if (first_reset) {
    env->last_episode_length = 0;
    env->last_prices = NULL;
    env->last_riskless_history = NULL;
    env->last_risky_history = NULL;
    env->render_frames_remaining = 0;
    env->history_capacity = 0;
    env->last_prices_capacity = 0;
    env->last_history_capacity = 0;
    first_reset = 0;
  }

  // Seed RNG only once
  static int rng_seeded = 0;
  if (!rng_seeded) {
    srand(time(NULL));
    rng_seeded = 1;
  }

  // Free old prices array to avoid memory leak
  if (env->prices != NULL) {
    free(env->prices);
    env->prices = NULL;
  }

  // simulate_fBm(env->H, env->T, env->T);
  if (env->process_type == 0)
    env->prices = sin_process(env);
  else
    env->prices = simulate_fBm(env->H, 2 * env->T, 2 * env->T);

  int needed_len = 2 * env->T + 1;
  if (env->riskless_history == NULL || env->risky_history == NULL ||
      env->history_capacity != needed_len) {
    free(env->riskless_history);
    free(env->risky_history);
    env->riskless_history = (float *)malloc(needed_len * sizeof(float));
    env->risky_history = (float *)malloc(needed_len * sizeof(float));
    env->history_capacity = needed_len;
  }

  // Initialize the full arrays
  memset(env->riskless_history, 0, needed_len * sizeof(float));
  memset(env->risky_history, 0, needed_len * sizeof(float));

  compute_observations(env);
}

void execute_action(Invest *env, float action) {
  float price = env->prices[env->tick];
  float prev_risky = env->risky;
  float prev_riskless = env->riskless;

  env->risky += action;
  // to have no friction set friction_coef = 0
  env->riskless = env->riskless - action * price -
                  env->friction_coef * pow(fabsf(action), env->friction_power);

  env->riskless_history[env->tick] = env->riskless;
  env->risky_history[env->tick] = env->risky;

  // env->rewards[0] = price > 0 ? -action : action;
  env->rewards[0] = env->riskless - prev_riskless; // this worked for sin

  // terminal riskless or 0
  // env->reward[0] = env->tick == env->T ? env->riskless : 0;

  env->tick += 1;
}

void liquidate(Invest *env) {
  float initial_riskless = env->riskless;
  float initial_risky = env->risky;
  float liquidation_step = -env->risky / (env->T + 1);
  int liquidation_steps = 0;

  env->log.liquidation_action = liquidation_step;

  switch (env->liq_type) {
  case 0:
    // liquidate entire position in a single step
    execute_action(env, -env->risky);
    liquidation_steps = 1;
    break;
  case 1:
    // liquidate position in env->T steps
    while (env->tick <= 2 * env->T) {
      execute_action(env, liquidation_step);
      liquidation_steps++;
    }
    break;
  default:
    execute_action(env, -env->risky);
    liquidation_steps = 1;
    break;
  }

  env->log.liquidation_steps = liquidation_steps;
  env->log.liquidation_cost = initial_riskless - env->riskless;
}

void c_step(Invest *env) {
  env->terminals[0] = 0;
  env->rewards[0] = 0;

  int action =
      env->actions[0] - 10; // {0, 1, ..., 20} -> {-10, ..., 0, ..., 10}
  // int action = env->actions[0] - 1; // {0, 1, 2} -> {-1, 0, 1}

  execute_action(env, action);

  if (env->tick >= env->T) {
    add_pre_terminal_log(env);
    if (env->liquidate) {
      float pre_liq_reward = env->rewards[0];
      liquidate(env);
      env->rewards[0] = pre_liq_reward;
    }
    // env->rewards[0] = env->riskless;
    env->terminals[0] = 1;
    // Snapshot last episode for rendering before reset
    int episode_len = env->tick; // includes liquidation if any
    if (episode_len > 0) {
      if (env->last_prices == NULL || env->last_prices_capacity < episode_len) {
        free(env->last_prices);
        env->last_prices =
            (double *)malloc(episode_len * sizeof(double));
        env->last_prices_capacity = episode_len;
      }
      if (env->last_riskless_history == NULL ||
          env->last_risky_history == NULL ||
          env->last_history_capacity < episode_len) {
        free(env->last_riskless_history);
        free(env->last_risky_history);
        env->last_riskless_history =
            (float *)malloc(episode_len * sizeof(float));
        env->last_risky_history =
            (float *)malloc(episode_len * sizeof(float));
        env->last_history_capacity = episode_len;
      }
      memcpy(env->last_prices, env->prices, episode_len * sizeof(double));
      memcpy(env->last_riskless_history, env->riskless_history,
             episode_len * sizeof(float));
      memcpy(env->last_risky_history, env->risky_history,
             episode_len * sizeof(float));
      env->last_episode_length = episode_len;
      env->render_frames_remaining = 60; // ~4 seconds at 15 fps
    }
    add_log(env);
    c_reset(env);
  }

  // Avoid recomputing observations immediately after a terminal reset
  if (!env->terminals[0])
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

  // Determine what data to render
  bool render_last_episode = (env->render_frames_remaining > 0);
  double *prices_to_render =
      render_last_episode ? env->last_prices : env->prices;
  float *riskless_to_render =
      render_last_episode ? env->last_riskless_history : env->riskless_history;
  float *risky_to_render =
      render_last_episode ? env->last_risky_history : env->risky_history;
  int episode_length =
      render_last_episode ? env->last_episode_length : env->tick;
  int T_to_use = env->T;

  if (env->render_frames_remaining == 1) c_reset(env);
  
  if (render_last_episode) {
    env->render_frames_remaining--;
  }

  // Draw axes
  Color axis_color = RAYWHITE;
  DrawLine(margin, screen_height - margin, screen_width - margin,
           screen_height - margin, axis_color);
  DrawLine(margin, margin, margin, screen_height - margin, axis_color);

  // Skip rendering if no data available
  if (prices_to_render == NULL || episode_length == 0) {
    EndDrawing();
    return;
  }

  // Calculate min/max values for scaling
  float max_price = 0;
  float max_assets = 0;
  float min_value = 0;
  float min_price = prices_to_render[0];
  float max_price_val = prices_to_render[0];

  for (int i = 0; i < episode_length; i++) {
    max_price = fmaxf(max_price, prices_to_render[i]);
    if (prices_to_render[i] < min_price)
      min_price = prices_to_render[i];
    if (prices_to_render[i] > max_price_val)
      max_price_val = prices_to_render[i];

    max_assets =
        fmaxf(max_assets, fmaxf(riskless_to_render[i], risky_to_render[i]));
    min_value =
        fminf(min_value, fminf(riskless_to_render[i], risky_to_render[i]));
  }

  float max_value = fmaxf(max_price, max_assets);
  float value_range = max_value - min_value;
  if (value_range == 0)
    value_range = 1;

  // Define colors
  Color price_color = YELLOW;
  Color riskless_color = SKYBLUE;
  Color risky_color = LIME;

  // Scale x-axis based on episode length (including liquidation)
  float total_time_scale = (float)episode_length;
  
  // Draw axis ticks and labels
  {
    int x_ticks = 8;
    int y_ticks = 6;
    
    // X-axis ticks (time)
    for (int i = 0; i <= x_ticks; i++) {
      float t = (float)i / x_ticks;
      float x = margin + t * graph_width;
      DrawLine((int)x, screen_height - margin - 5, (int)x, screen_height - margin + 5, axis_color);
      int time_label = (int)roundf(t * total_time_scale);
      DrawText(TextFormat("%d", time_label), (int)(x - 10), screen_height - margin + 8, 12, axis_color);
    }
    DrawText("Time", screen_width / 2 - 20, screen_height - margin + 24, 14, axis_color);
    
    // Y-axis ticks (value)
    for (int i = 0; i <= y_ticks; i++) {
      float t = (float)i / y_ticks;
      float y = screen_height - margin - t * graph_height;
      DrawLine(margin - 5, (int)y, margin + 5, (int)y, axis_color);
      float v = min_value + t * value_range;
      DrawText(TextFormat("%.0f", v), margin - 44, (int)(y - 8), 12, axis_color);
    }
    DrawText("Value", margin - 40, margin - 26, 14, axis_color);
  }
  
  // Draw a faint horizontal line at y = 0 if within range
  if (0.0f >= min_value && 0.0f <= (min_value + value_range)) {
    float y_zero = screen_height - margin - ((0.0f - min_value) / value_range) * graph_height;
    Color zero_color = ColorAlpha(RED, 0.2f);
    DrawLine(margin, (int)y_zero, screen_width - margin, (int)y_zero, zero_color);
  }
  
  // Draw price line (YELLOW)
  for (int i = 1; i < episode_length; i++) {
    float x1 = margin + (i - 1) * graph_width / total_time_scale;
    float x2 = margin + i * graph_width / total_time_scale;

    float norm1 = (prices_to_render[i - 1] - min_price) /
                  (max_price_val - min_price + 1e-8f);
    float norm2 =
        (prices_to_render[i] - min_price) / (max_price_val - min_price + 1e-8f);

    // Map to 25% - 75% of the graph height
    float y1 = screen_height - margin - (0.25f + 0.5f * norm1) * graph_height;
    float y2 = screen_height - margin - (0.25f + 0.5f * norm2) * graph_height;

    DrawLine(x1, y1, x2, y2, price_color);
  }

  // Draw riskless asset line (SKYBLUE)
  for (int i = 1; i < episode_length; i++) {
    float x1 = margin + (i - 1) * graph_width / total_time_scale;
    float y1 =
        screen_height - margin -
        ((riskless_to_render[i - 1] - min_value) / value_range) * graph_height;
    float x2 = margin + i * graph_width / total_time_scale;
    float y2 =
        screen_height - margin -
        ((riskless_to_render[i] - min_value) / value_range) * graph_height;
    DrawLine(x1, y1, x2, y2, riskless_color);
  }

  // Draw risky asset line (LIME)
  for (int i = 1; i < episode_length; i++) {
    float x1 = margin + (i - 1) * graph_width / total_time_scale;
    float y1 =
        screen_height - margin -
        ((risky_to_render[i - 1] - min_value) / value_range) * graph_height;
    float x2 = margin + i * graph_width / total_time_scale;
    float y2 = screen_height - margin -
               ((risky_to_render[i] - min_value) / value_range) * graph_height;
    DrawLine(x1, y1, x2, y2, risky_color);
  }

  // Draw vertical line to indicate end of main episode and start of liquidation
  if (episode_length > T_to_use && env->liquidate) {
    float liquidation_start_x =
        margin + T_to_use * graph_width / total_time_scale;
    DrawLine(liquidation_start_x, margin, liquidation_start_x,
             screen_height - margin, RED);
    DrawText("Liquidation", liquidation_start_x + 5, margin + 10, 16, RED);
  }

  // Draw current values
  if (render_last_episode) {
    DrawText("EPISODE COMPLETE - Showing liquidation", 10, 10, 20, RED);
    DrawText(TextFormat("Frames remaining: %d", env->render_frames_remaining),
             10, 35, 16, RAYWHITE);
  } else {
    DrawText(TextFormat("Price: %.2f", prices_to_render[env->tick]),
             screen_width - 200, margin - 20, 20, price_color);
    DrawText(TextFormat("Riskless: %.2f", env->riskless), screen_width - 200,
             margin, 20, riskless_color);
    DrawText(TextFormat("Risky: %.2f", env->risky), screen_width - 200,
             margin + 20, 20, risky_color);
    DrawText(TextFormat("Tick: %d / %d", env->tick, env->T), screen_width - 200,
             margin + 40, 20, RAYWHITE);
  }

  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Invest *env) {
  free(env->prices);
  free(env->riskless_history);
  free(env->risky_history);
  free(env->last_prices);
  free(env->last_riskless_history);
  free(env->last_risky_history);
  if (IsWindowReady()) {
    CloseWindow();
  }
}
