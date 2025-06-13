/* Squared: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "raylib.h"

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    float* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    
    // env specific
    int time_horizon;
    float* prices;
    float riskless_asset;
    float risky_asset;
    int tick;
} InvestSim;

void add_log(InvestSim* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(InvestSim* env) {
    env->prices = (float*)malloc((env->time_horizon + 1) * sizeof(float));
    for (int i = 0; i <= env->time_horizon; i++) {
        env->prices[i] = sin(i / 10.0f) + 1;
    }

    env->riskless_asset = 0;
    env->risky_asset = 0;

    memset(env->observations, 0, 3 * sizeof(float));
    env->observations[0] = env->prices[0];
    env->observations[1] = env->riskless_asset;
    env->observations[2] = env->risky_asset;
    env->tick = 0;
}

// Required function
void c_step(InvestSim* env) {
    env->tick += 1;

    float action = env->actions[0];  // how much to buy/sell of the risky asset

    /* ===== runtime sanity check –– delete after debugging ===== */
    if (!isfinite(action) || action < -100.0001f || action > 100.0001f) {
        fprintf(stderr,
                "[BAD ACTION] tick=%d  raw=%.6f\n",
                env->tick, action);
        fflush(stderr);
    }
    /* ========================================================== */

    if (!isfinite(action))             action = 0.0f;
    action = fminf(fmaxf(action, -100.0f), 100.0f);
    env->actions[0] = action;

    env->terminals[0] = 0;
    env->rewards[0] = 0;

    env->risky_asset += action;
    env->riskless_asset -= action * env->prices[env->tick];

    if (env->tick >= env->time_horizon) {
        env->terminals[0] = 1;
        env->rewards[0] = env->riskless_asset + env->risky_asset * env->prices[env->time_horizon];
        add_log(env);
        c_reset(env);
    }

    // printf("tick: %d, prices: %f, riskless_asset: %f, risky_asset: %f\n", env->tick, env->prices[env->tick], env->riskless_asset, env->risky_asset);
    env->observations[0] = env->prices[env->tick];
    env->observations[1] = env->riskless_asset;
    env->observations[2] = env->risky_asset;
}

// Required function. Should handle creating the client on first call
void c_render(InvestSim* env) {
    if (!IsWindowReady()) {
        InitWindow(800, 600, "Investment Simulation");
        SetTargetFPS(30);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Draw axes
    DrawLine(50, 550, 750, 550, WHITE);  // x-axis
    DrawLine(50, 50, 50, 550, WHITE);    // y-axis

    // Find min and max values for scaling
    float min_val = INFINITY;
    float max_val = -INFINITY;
    for (int i = 0; i <= env->time_horizon; i++) {
        min_val = fminf(min_val, env->prices[i]);
        max_val = fmaxf(max_val, env->prices[i]);
    }
    min_val = fminf(min_val, fminf(env->riskless_asset, env->risky_asset));
    max_val = fmaxf(max_val, fmaxf(env->riskless_asset, env->risky_asset));

    // Add some padding to the range
    float range = max_val - min_val;
    min_val -= range * 0.1f;
    max_val += range * 0.1f;

    // Draw price line (blue)
    for (int i = 0; i < env->time_horizon; i++) {
        int x1 = 50 + (i * 700) / env->time_horizon;
        int y1 = 550 - ((env->prices[i] - min_val) * 500) / (max_val - min_val);
        int x2 = 50 + ((i + 1) * 700) / env->time_horizon;
        int y2 = 550 - ((env->prices[i + 1] - min_val) * 500) / (max_val - min_val);
        DrawLine(x1, y1, x2, y2, BLUE);
    }

    // Draw current risky asset value (green)
    int risky_x = 50 + (env->tick * 700) / env->time_horizon;
    int risky_y = 550 - ((env->risky_asset - min_val) * 500) / (max_val - min_val);
    DrawCircle(risky_x, risky_y, 5, GREEN);

    // Draw current riskless asset value (red)
    int riskless_x = 50 + (env->tick * 700) / env->time_horizon;
    int riskless_y = 550 - ((env->riskless_asset - min_val) * 500) / (max_val - min_val);
    DrawCircle(riskless_x, riskless_y, 5, RED);

    // Draw legend
    DrawText("Price", 60, 30, 20, BLUE);
    DrawText("Risky Asset", 150, 30, 20, GREEN);
    DrawText("Riskless Asset", 280, 30, 20, RED);

    // Draw current values
    char buffer[100];
    sprintf(buffer, "Time: %d/%d", env->tick, env->time_horizon);
    DrawText(buffer, 600, 30, 20, WHITE);

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(InvestSim* env) {
    free(env->prices);
    if (IsWindowReady()) {
        CloseWindow();
    }
}
