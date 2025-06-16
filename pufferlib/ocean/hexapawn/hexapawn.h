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
    int reward_move_valid;
    int reward_move_invalid;
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

int is_valid_move(Hexapawn* env, Move m, int player) {
  if (env->observations[p2i(env, m.from)] != player) return 0;
  // illegal to move backwards
  if (player == AGENT) {
    if (m.from.r > m.to.r) return 0;
  } else if (player == OPPONENT) {
    if (m.from.r < m.to.r) return 0;
  }

  int other_player = player == AGENT ? OPPONENT : AGENT;

  if (m.from.r != m.to.r && m.from.c != m.to.c) {
    // diagonal moves must be captures
    return env->observations[p2i(env, m.to)] == other_player ? 1 : 0;
  } else {
    // can only go to empty cell forwards
    return env->observations[p2i(env, m.to)] == EMPTY ? 1 : 0;
  }
  return 0;
}

int num_valid_moves(Hexapawn* env, int player) {
  int n = 0;
  int num_total_moves = env->size * env->size * 6;
  for (int i = 0; i < num_total_moves; i++) {
    Move move = decode_action(env, i);
    if (is_valid_move(env, move, OPPONENT)) n++;
  }
  return n;
}

int reached_other_side(Hexapawn* env, int player) {
  if (player == AGENT) {
    for (int i = 0; i < env->size; i++) {
      if (env->observations[env->size*(env->size-1) + i] == AGENT) {
        return 1;
      }
    }
  } else if (player == OPPONENT) {
    for (int i = 0; i < env->size; i++) {
      if (env->observations[i] == OPPONENT) {
        return 1;
      }
    }
  } else {
    return 0;
  }
}

int make_move(Hexapawn* env, Move move, int player) {
  if (!is_valid_move(env, move, player)) return 0;

  env->observations[p2i(env, move.from)] = EMPTY;
  env->observations[p2i(env, move.to)] = player;

  // for (int i = 0; i < env->size; i++) {
  //   for (int j = 0; j < env->size; j++) {
  //     int piece = env->observations[i*env->size + j];
  //     printf("%d ", piece);
  //   }
  //   printf("\n");
  // }
  
  if (reached_other_side(env, player)) {
    env->terminals[0] = 1;
  }

  // check if opponent has valid moves left
  int other_player = player == AGENT ? OPPONENT : AGENT;
  if (num_valid_moves(env, other_player) == 0) {
    env->terminals[0] = 1;
  }

  if (env->terminals[0] == 1) {
    if (player == AGENT) {
      env->rewards[0] = 1;
    } else if (player == OPPONENT) {
      env->rewards[0] = -1;
    }
  }

  return 1;
}

void scripted_opponent(Hexapawn* env) {
  // do first legal move
  int num_total_moves = env->size * env->size * 6;
  for (int i = 0; i < num_total_moves; i++) {
    Move move = decode_action(env, i);
    printf("AAAAAA\n");
    if (make_move(env, move, OPPONENT)) {
      break;
    } 
  }
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
    env->terminals[0] = 0;
}

// Required function
void c_step(Hexapawn* env) {
  env->tick += 1;
  env->rewards[0] = 0.0;
  int action = (int)env->actions[0];
  Move move = decode_action(env, action);

  if (make_move(env, move, AGENT)) {
    env->rewards[0] += env->reward_move_valid;
    env->log.episode_return += env->reward_move_valid;
    if (env->terminals[0] != 1) {
      printf("BBBBBBBBBBB\n");
      scripted_opponent(env);
    }
  } else {
    env->rewards[0] += env->reward_move_invalid;
    env->log.episode_return += env->reward_move_invalid;
  }

  if(env->rewards[0] > 1){
    env->rewards[0] = 1;
  } 
  if(env->rewards[0] < -1){
    env->rewards[0] = -1;
  }

  if (env->terminals[0] == 1) {
    add_log(env);
    c_reset(env);
    return;
  }
}

// Required function. Should handle creating the client on first call
void c_render(Hexapawn* env) {
  const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

  int cell_size = 128;
  if (!IsWindowReady()) {
    InitWindow(cell_size * env->size, cell_size * env->size, "PufferLib Hexapawn");
    SetTargetFPS(60);
  }

  // Standard across our envs so exiting is always the same
  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground(PUFF_BACKGROUND);

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

