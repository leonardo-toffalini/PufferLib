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
#include <unistd.h> // for usleep

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
    float* risky_asset_history;
    float* riskless_asset_history;
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
    env->risky_asset_history = (float*)malloc((env->time_horizon + 1) * sizeof(float));
    env->riskless_asset_history = (float*)malloc((env->time_horizon + 1) * sizeof(float));
    for (int i = 0; i <= env->time_horizon; i++) {
        env->prices[i] = sin(i / 10.0f) + 1;
        env->risky_asset_history[i] = 0;
        env->riskless_asset_history[i] = 0;
    }
    env->riskless_asset = 0;
    env->risky_asset = 0;
    memset(env->observations, 0, 3 * sizeof(float));
    env->observations[0] = env->prices[0];
    env->observations[1] = env->riskless_asset;
    env->observations[2] = env->risky_asset;
    env->tick = 0;
    // Record initial asset values
    env->risky_asset_history[0] = env->risky_asset;
    env->riskless_asset_history[0] = env->riskless_asset;
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

    // Record asset values at this tick
    env->risky_asset_history[env->tick] = env->risky_asset;
    env->riskless_asset_history[env->tick] = env->riskless_asset;

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
void c_render_raylib(InvestSim* env) {
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
    free(env->risky_asset_history);
    free(env->riskless_asset_history);
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void c_render(InvestSim* env) {
    printf("\033[2J\033[H");
    printf("\033[1;36mInvestment Simulation\033[0m\n");
    printf("Time: %d/%d\n\n", env->tick, env->time_horizon);
    printf("\033[1;34mCurrent Price: %.2f\033[0m\n", env->prices[env->tick]);
    printf("\033[1;32mRisky Asset: %.2f\033[0m\n", env->risky_asset);
    printf("\033[1;31mRiskless Asset: %.2f\033[0m\n\n", env->riskless_asset);
    const int width = 120;
    const int height = 40;
    char chart[height][width + 1];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            chart[i][j] = ' ';
        }
        chart[i][width] = '\0';
    }
    float min_val = INFINITY;
    float max_val = -INFINITY;
    for (int i = 0; i <= env->tick; i++) {
        min_val = fminf(min_val, env->prices[i]);
        max_val = fmaxf(max_val, env->prices[i]);
        min_val = fminf(min_val, env->risky_asset_history[i]);
        max_val = fmaxf(max_val, env->risky_asset_history[i]);
        min_val = fminf(min_val, env->riskless_asset_history[i]);
        max_val = fmaxf(max_val, env->riskless_asset_history[i]);
    }
    float range = max_val - min_val;
    min_val -= range * 0.1f;
    max_val += range * 0.1f;
    // Plot price history
    for (int i = 0; i < width; i++) {
        int time_idx = (i * env->tick) / (width - 1);
        float price = env->prices[time_idx];
        int y = (int)((price - min_val) * (height - 1) / (max_val - min_val));
        y = height - 1 - y;
        if (y >= 0 && y < height) {
            chart[y][i] = '*';
        }
    }
    // Plot risky asset history
    for (int i = 0; i < width; i++) {
        int time_idx = (i * env->tick) / (width - 1);
        float risky = env->risky_asset_history[time_idx];
        int y = (int)((risky - min_val) * (height - 1) / (max_val - min_val));
        y = height - 1 - y;
        if (y >= 0 && y < height) {
            chart[y][i] = '^';
        }
    }
    // Plot riskless asset history
    for (int i = 0; i < width; i++) {
        int time_idx = (i * env->tick) / (width - 1);
        float riskless = env->riskless_asset_history[time_idx];
        int y = (int)((riskless - min_val) * (height - 1) / (max_val - min_val));
        y = height - 1 - y;
        if (y >= 0 && y < height) {
            chart[y][i] = 'v';
        }
    }
    printf("\033[1;34mPrice: ●\033[0m  \033[1;32mRisky: ●\033[0m  \033[1;31mRiskless: ●\033[0m\n");
    printf("┌");
    for (int i = 0; i < width; i++) printf("─");
    printf("┐\n");
    for (int i = 0; i < height; i++) {
        printf("│");
        for (int j = 0; j < width; j++) {
            char c = chart[i][j];
            if (c == '*') printf("\033[1;34m●\033[0m");
            else if (c == '^') printf("\033[1;32m●\033[0m");
            else if (c == 'v') printf("\033[1;31m●\033[0m");
            else printf(" ");
        }
        printf("│\n");
    }
    printf("└");
    for (int i = 0; i < width; i++) printf("─");
    printf("┘\n");
    printf("Min: %.2f  Max: %.2f\n", min_val, max_val);
    fflush(stdout);
    usleep(100000); // Sleep for 0.1 seconds (100,000 microseconds)
}
