#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char OPPONENT = 2;

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
    unsigned char* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int size;
    int tick;
    int r;
    int c;
} Hexapawn;

typedef struct {
    int r;
    int c;
} Position;

typedef struct {
    Position from;
    Position to;
} Move;

Move decode_action(Hexapawn* env, int action) {
    // Action space is size*size*6 where 6 is number of move types
    // Move types are: N, S, NW, NE, SW, SE
    // 10 = 1*6 + 4 => second indexed cell, fourth move type
    // 50 = 8*6 + 2 => ninth indexed cell, second move type
    int pos = action / 6;
    int move_type = action % 6;

    Move m;
    m.from.r = pos / env->size;
    m.from.c = pos % env->size;
    m.to.r = m.from.r;
    m.to.c = m.from.c;
    
    // Decode move type into target position
    switch(move_type) {
        case 0: // N
            m.to.r = m.from.r - 1;
            m.to.c = m.from.c;
            break;
        case 1: // S  
            m.to.r = m.from.r + 1;
            m.to.c = m.from.c;
            break;
        case 2: // NW
            m.to.r = m.from.r - 1;
            m.to.c = m.from.c - 1;
            break;
        case 3: // NE
            m.to.r = m.from.r - 1;
            m.to.c = m.from.c + 1;
            break;
        case 4: // SW
            m.to.r = m.from.r + 1;
            m.to.c = m.from.c - 1;
            break;
        case 5: // SE
            m.to.r = m.from.r + 1;
            m.to.c = m.from.c + 1;
            break;
    }
    
    return m;
}

int p2i(Hexapawn* env, Position p) {
    // position to index
    return p.r * env->size + p.c;
}

int is_valid_move(Hexapawn* env, Move m) {
    if (env->observations[p2i(env, m.from)] != AGENT) return 0;
    if (m.from.r > m.to.r) return 0;

    if (m.from.r != m.to.r && m.from.c != m.to.c) {
        return env->observations[p2i(env, m.to)] == OPPONENT ? 1 : 0;
    } else {
        return env->observations[p2i(env, m.to)] == EMPTY ? 1 : 0;
    }
    return 0;
}

int legal_moves(Hexapawn* env) {
    int moves = 0;
    int total_possible_moves = env->size * env->size * 6;
    for (int i = 0; i < total_possible_moves; i++) {
        Move m = decode_action(env, i);
        if (is_valid_move(env, m)) moves++;
    }
    return moves;
}

int game_over(Hexapawn* env) {
    for (int i = 0; i < env->size; i++) {
        if (env->observations[i] == OPPONENT) return 1;
    }
    for (int i = 0; i < env->size; i++) {
        if (env->observations[(env->size-1)*env->size + i] == AGENT) return 1;
    }
    if (legal_moves(env) == 0) return 1;
    return 0;
}

void add_log(Hexapawn* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->rewards[0];
    env->log.n++;
}

// Required function
void c_reset(Hexapawn* env) {
    int tiles = env->size*env->size;
    memset(env->observations, 0, tiles*sizeof(unsigned char));
    for (int i = 0; i < env->size; i++) env->observations[i] = AGENT;
    for (int i = 0; i < env->size; i++) env->observations[(env->size-1)*env->size + i] = OPPONENT;
    env->r = 0;
    env->c = 0;
    env->tick = 0;
}

void opponent_move(Hexapawn* env) {
    // Find all opponent pieces
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++) {
            if (env->observations[i*env->size + j] != OPPONENT) continue;
            
            // Try moves in order: forward, capture left, capture right
            Position from = {i, j};
            
            // Try forward move
            Position to = {i-1, j};
            if (i > 0 && env->observations[p2i(env, to)] == EMPTY) {
                env->observations[p2i(env, from)] = EMPTY;
                env->observations[p2i(env, to)] = OPPONENT;
                return;
            }
            
            // Try capture left
            to = (Position){i-1, j-1};
            if (i > 0 && j > 0 && env->observations[p2i(env, to)] == AGENT) {
                env->observations[p2i(env, from)] = EMPTY;
                env->observations[p2i(env, to)] = OPPONENT;
                return;
            }
            
            // Try capture right
            to = (Position){i-1, j+1};
            if (i > 0 && j < env->size-1 && env->observations[p2i(env, to)] == AGENT) {
                env->observations[p2i(env, from)] = EMPTY;
                env->observations[p2i(env, to)] = OPPONENT;
                return;
            }
        }
    }
}

// Required function
void c_step(Hexapawn* env) {
    env->tick += 1;

    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    Move m = decode_action(env, action);
    
    // Check if move is valid
    if (!is_valid_move(env, m)) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;  // Invalid move means loss
        add_log(env);
        c_reset(env);
        return;
    }

    // Execute the agent's move
    env->observations[p2i(env, m.from)] = EMPTY;
    env->observations[p2i(env, m.to)] = AGENT;

    // Check for game over after agent's move
    if (game_over(env)) {
        env->terminals[0] = 1;
        
        // Check if agent won (reached opponent's side)
        int agent_won = 0;
        for (int i = 0; i < env->size; i++) {
            if (env->observations[(env->size-1)*env->size + i] == AGENT) {
                agent_won = 1;
                break;
            }
        }
        
        // Check if opponent won (reached agent's side)
        int opponent_won = 0;
        for (int i = 0; i < env->size; i++) {
            if (env->observations[i] == OPPONENT) {
                opponent_won = 1;
                break;
            }
        }

        // Assign rewards based on outcome
        if (agent_won) {
            env->rewards[0] = 1.0;  // Agent won
        } else if (opponent_won) {
            env->rewards[0] = -1.0;  // Opponent won
        } else {
            env->rewards[0] = 0.0;  // No legal moves left
        }
        
        add_log(env);
        c_reset(env);
        return;
    }

    // Opponent's turn
    opponent_move(env);

    // Check for game over after opponent's move
    if (game_over(env)) {
        env->terminals[0] = 1;
        
        // Check if agent won (reached opponent's side)
        int agent_won = 0;
        for (int i = 0; i < env->size; i++) {
            if (env->observations[(env->size-1)*env->size + i] == AGENT) {
                agent_won = 1;
                break;
            }
        }
        
        // Check if opponent won (reached agent's side)
        int opponent_won = 0;
        for (int i = 0; i < env->size; i++) {
            if (env->observations[i] == OPPONENT) {
                opponent_won = 1;
                break;
            }
        }

        // Assign rewards based on outcome
        if (agent_won) {
            env->rewards[0] = 1.0;  // Agent won
        } else if (opponent_won) {
            env->rewards[0] = -1.0;  // Opponent won
        } else {
            env->rewards[0] = 0.0;  // No legal moves left
        }
        
        add_log(env);
        c_reset(env);
        return;
    }
}

// Required function. Should handle creating the client on first call
void c_render(Hexapawn* env) {
    if (!IsWindowReady()) {
        InitWindow(64*env->size, 64*env->size, "PufferLib Hexapawn");
        SetTargetFPS(5);
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(BLACK);

    int cell_size = 64;
    int board_size = env->size * cell_size;

    // Draw grid lines
    for (int i = 0; i <= env->size; i++) {
        // Vertical lines
        DrawLine(i * cell_size, 0, i * cell_size, board_size, GRAY);
        // Horizontal lines
        DrawLine(0, i * cell_size, board_size, i * cell_size, GRAY);
    }

    // Draw pieces
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++) {
            int piece = env->observations[i*env->size + j];
            if (piece == EMPTY) continue;

            Color piece_color = (piece == AGENT) ? BLUE : RED;
            int center_x = j * cell_size + cell_size/2;
            int center_y = i * cell_size + cell_size/2;
            int radius = cell_size/3;

            DrawCircle(center_x, center_y, radius, piece_color);
        }
    }

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Hexapawn* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
