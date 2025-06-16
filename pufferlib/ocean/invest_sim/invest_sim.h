#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "raylib.h"
#include <unistd.h> // for usleep

#define PI 3.14159265358979323846
#define MAX_TIME_HORIZON 1000  // Define a reasonable maximum time horizon

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    int window_size;
    float n; // Required as the last field 
} Log;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet

    // env specific
    int time_horizon;
    int window_size;
    float riskless_asset;
    float risky_asset;
    float prices[MAX_TIME_HORIZON]; // Fixed-size array for prices
    float riskless_history[MAX_TIME_HORIZON + 1]; // History for riskless asset
    float risky_history[MAX_TIME_HORIZON + 1];    // History for risky asset

    int tick;
} InvestSim;

void add_log(InvestSim* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

float sin_price_function(InvestSim* env) {
    return sin(2 * PI * env->tick / (float)env->time_horizon) + 1;
}

// Required function
void c_reset(InvestSim* env) {
    if (env->time_horizon > MAX_TIME_HORIZON) {
        fprintf(stderr, "Error: Time horizon is greater than MAX_TIME_HORIZON: %d > %d\n", env->time_horizon, MAX_TIME_HORIZON);
        exit(1);
    }

    env->tick = 0;
    env->riskless_asset = 0;
    env->risky_asset = 0;
    
    // Precompute all prices
    for (int i = 0; i <= env->time_horizon; i++) {
        env->tick = i;  // Temporarily set tick to compute price
        env->prices[i] = sin_price_function(env);
    }
    env->tick = 0;  // Reset tick back to 0
    
    // Initialize asset histories
    for (int i = 0; i <= env->time_horizon; i++) {
        env->riskless_history[i] = 0;
        env->risky_history[i] = 0;
    }
    
    env->observations[0] = env->prices[0];
    env->observations[1] = env->riskless_asset;
    env->observations[2] = env->risky_asset;
    env->observations[3] = env->time_horizon - env->tick;
    memset(&env->observations[4], 0, env->window_size * sizeof(float));
}

// Required function
void c_step(InvestSim* env) {
    int action = env->actions[0];  // how much to buy/sell of the risky asset
    action = action - 50;

    env->terminals[0] = 0;
    env->rewards[0] = 0;

    float price = env->prices[env->tick];
    env->risky_asset += action;
    env->riskless_asset -= action * price;
    env->rewards[0] = env->riskless_asset + env->risky_asset * price;

    // Store asset history
    env->riskless_history[env->tick] = env->riskless_asset;
    env->risky_history[env->tick] = env->risky_asset;

    if (env->tick >= env->time_horizon) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }

    env->observations[0] = price;
    env->observations[1] = env->riskless_asset;
    env->observations[2] = env->risky_asset;
    env->observations[3] = env->time_horizon - env->tick;

    memmove(&env->observations[4], &env->observations[4 + 1], (env->window_size - 1) * sizeof(float));
    env->observations[4 + env->window_size - 1] = price;
    env->tick += 1;
}

// Required function. Should handle creating the client on first call
void c_render(InvestSim* env) {
    static bool window_initialized = false;
    const int screen_width = 800;
    const int screen_height = 600;
    const int margin = 50;
    const int graph_width = screen_width - 2 * margin;
    const int graph_height = screen_height - 2 * margin;

    if (!window_initialized) {
        InitWindow(screen_width, screen_height, "Investment Simulation");
        SetTargetFPS(60);
        window_initialized = true;
    }

    if (!IsWindowReady()) return;

    BeginDrawing();
    Color dark_bg = (Color){30, 30, 30, 255};
    ClearBackground(dark_bg);

    // Draw axes
    Color axis_color = RAYWHITE;
    DrawLine(margin, screen_height - margin, screen_width - margin, screen_height - margin, axis_color);
    DrawLine(margin, margin, margin, screen_height - margin, axis_color);

    // Find min and max values for scaling
    float max_price = 0;
    float max_assets = 0;
    float min_value = 0;
    float min_price = env->prices[0];
    float max_price_val = env->prices[0];
    for (int i = 0; i <= env->tick; i++) {
        max_price = fmaxf(max_price, env->prices[i]);
        max_assets = fmaxf(max_assets, fmaxf(env->riskless_history[i], env->risky_history[i]));
        min_value = fminf(min_value, fminf(env->riskless_history[i], env->risky_history[i]));
        if (env->prices[i] < min_price) min_price = env->prices[i];
        if (env->prices[i] > max_price_val) max_price_val = env->prices[i];
    }
    float max_value = fmaxf(max_price, max_assets);
    // To allow negative values, adjust min_value
    float value_range = max_value - min_value;
    if (value_range == 0) value_range = 1;

    // Define colors
    Color price_color = YELLOW;
    Color riskless_color = SKYBLUE;
    Color risky_color = LIME;

    // Draw price line (YELLOW), always in the middle half of the graph
    for (int i = 1; i <= env->tick; i++) {
        float x1 = margin + (i - 1) * graph_width / (float)env->time_horizon;
        float x2 = margin + i * graph_width / (float)env->time_horizon;

        float norm1 = (env->prices[i - 1] - min_price) / (max_price_val - min_price + 1e-8f);
        float norm2 = (env->prices[i] - min_price) / (max_price_val - min_price + 1e-8f);

        // Map to 25% - 75% of the graph height
        float y1 = screen_height - margin - (0.25f + 0.5f * norm1) * graph_height;
        float y2 = screen_height - margin - (0.25f + 0.5f * norm2) * graph_height;

        DrawLine(x1, y1, x2, y2, price_color);
    }

    // Draw riskless asset line (SKYBLUE)
    for (int i = 1; i < env->tick; i++) {
        float x1 = margin + (i - 1) * graph_width / (float)env->time_horizon;
        float y1 = screen_height - margin - ((env->riskless_history[i - 1] - min_value) / value_range) * graph_height;
        float x2 = margin + i * graph_width / (float)env->time_horizon;
        float y2 = screen_height - margin - ((env->riskless_history[i] - min_value) / value_range) * graph_height;
        DrawLine(x1, y1, x2, y2, riskless_color);
    }

    // Draw risky asset line (LIME)
    for (int i = 1; i < env->tick; i++) {
        float x1 = margin + (i - 1) * graph_width / (float)env->time_horizon;
        float y1 = screen_height - margin - ((env->risky_history[i - 1] - min_value) / value_range) * graph_height;
        float x2 = margin + i * graph_width / (float)env->time_horizon;
        float y2 = screen_height - margin - ((env->risky_history[i] - min_value) / value_range) * graph_height;
        DrawLine(x1, y1, x2, y2, risky_color);
    }

    // Draw labels
    DrawText("Price", margin - 40, margin - 20, 20, price_color);
    DrawText("Riskless Asset", margin - 40, margin, 20, riskless_color);
    DrawText("Risky Asset", margin - 40, margin + 20, 20, risky_color);

    // Draw current values
    char value_text[100];
    sprintf(value_text, "Price: %.2f", env->prices[env->tick]);
    DrawText(value_text, screen_width - 200, margin - 20, 20, price_color);
    sprintf(value_text, "Riskless: %.2f", env->riskless_asset);
    DrawText(value_text, screen_width - 200, margin, 20, riskless_color);
    sprintf(value_text, "Risky: %.2f", env->risky_asset);
    DrawText(value_text, screen_width - 200, margin + 20, 20, risky_color);

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(InvestSim* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
