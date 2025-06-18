#include "raylib.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char OPPONENT = 2;

// Required struct. Only use floats!
typedef struct {
  float perf;  // Recommended 0-1 normalized single real number perf metric
  float score; // Recommended unnormalized single real number perf metric
  float episode_return; // Recommended metric: sum of agent rewards over episode
  float episode_length; // Recommended metric: number of steps of agent episode
  // Any extra fields you add here may be exported to Python in binding.c
  float n; // Required as the last field
} Log;

typedef struct Client {
  Texture2D black_pawn;
  Texture2D white_pawn;
} Client;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
  Client *client;
  Log log; // Required field. Env binding code uses this to aggregate logs
  unsigned char *observations; // Required. You can use any obs type, but make
                               // sure it matches in Python!
  int *actions;   // Required. int* for discrete/multidiscrete, float* for box
  float *rewards; // Required
  unsigned char
      *terminals; // Required. We don't yet have truncations as standard yet
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

Move decode_action(Hexapawn *env, int action) {
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
  switch (move_type) {
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

int p2i(Hexapawn *env, Position p) {
  // position to index
  return p.r * env->size + p.c;
}

int is_valid_move(Hexapawn *env, Move m, int player) {
  // Check if move is within board bounds
  if (m.from.r < 0 || m.from.r >= env->size || m.from.c < 0 ||
      m.from.c >= env->size || m.to.r < 0 || m.to.r >= env->size ||
      m.to.c < 0 || m.to.c >= env->size) {
    return 0;
  }

  if (env->observations[p2i(env, m.from)] != player) {
    return 0;
  }
  // illegal to move backwards
  if (player == AGENT) {
    if (m.from.r > m.to.r)
      return 0;
  } else if (player == OPPONENT) {
    if (m.from.r < m.to.r) {
      return 0;
    }
  }

  int other_player = player == AGENT ? OPPONENT : AGENT;

  if (m.from.r != m.to.r && m.from.c != m.to.c) {
    // diagonal moves must be captures
    return env->observations[p2i(env, m.to)] == other_player ? 1 : 0;
  } else {
    // can only go to empty cell forwards
    return env->observations[p2i(env, m.to)] == EMPTY ? 1 : 0;
  }
}

int num_valid_moves(Hexapawn *env, int player) {
  int n = 0;
  int num_total_moves = env->size * env->size * 6;
  for (int i = 0; i < num_total_moves; i++) {
    Move move = decode_action(env, i);
    if (is_valid_move(env, move, OPPONENT))
      n++;
  }
  return n;
}

int reached_other_side(Hexapawn *env, int player) {
  switch (player) {
  case AGENT:
    for (int i = 0; i < env->size; i++) {
      if (env->observations[env->size * (env->size - 1) + i] == AGENT) {
        return 1;
      }
    }
  case OPPONENT:
    for (int i = 0; i < env->size; i++) {
      if (env->observations[i] == OPPONENT) {
        return 1;
      }
    }
  default:
    return 0;
  }
}

int count_pieces(Hexapawn *env, int player) {
  int num_opp_pieces = 0;
  for (int i = 0; i < env->size; i++) {
    for (int j = 0; j < env->size; j++) {
      int piece = env->observations[i * env->size + j];
      if (piece == player)
        num_opp_pieces++;
    }
  }
  return num_opp_pieces;
}

int make_move(Hexapawn *env, Move move, int player) {
  if (!is_valid_move(env, move, player))
    return 0;

  env->observations[p2i(env, move.from)] = EMPTY;
  env->observations[p2i(env, move.to)] = player;

  int other_player = player == AGENT ? OPPONENT : AGENT;
  int opponent_has_moves = num_valid_moves(env, other_player) > 0;

  // First check if current player won by reaching other side
  if (reached_other_side(env, player)) {
    env->terminals[0] = 1;
    env->rewards[0] = player == AGENT ? 1 : -1;
    return 1;
  }
  // Then check if opponent has no moves (only if current player hasn't won)
  if (!opponent_has_moves) {
    env->terminals[0] = 1;
    env->rewards[0] = player == AGENT ? 1 : -1;
    return 1;
  }
  // Then check if opponent has no pieces left
  int num_opp_pieces = count_pieces(env, other_player);
  if (num_opp_pieces == 0) {
    env->terminals[0] = 1;
    env->rewards[0] = player == AGENT ? 1 : -1;
    return 1;
  }

  return 1;
}

void first_move(Hexapawn *env) {
  // Try moves in sequence until finding a valid one
  Move move;
  int num_total_moves = env->size * env->size * 6;
  for (int i = 0; i < num_total_moves; i++) {
    move = decode_action(env, i);
    if (make_move(env, move, OPPONENT)) {
      break;
    }
  }
}

void random_move(Hexapawn *env) {
  // Try moves in sequence until finding a valid one
  srand(time(NULL));
  int num_total_moves = env->size * env->size * 6;
  int move_id;
  Move move;
  while (true) {
    move_id = (int)(rand() % num_total_moves);
    move = decode_action(env, move_id);
    if (make_move(env, move, OPPONENT)) {
      break;
    }
  }
}

void scripted_opponent(Hexapawn *env, int difficulty) {
  switch (difficulty) {
  case 0:
    first_move(env);
    break;
  case 1:
    random_move(env);
    break;
  default:
    first_move(env);
    break;
  }
}

void add_log(Hexapawn *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

// Required function
void c_reset(Hexapawn *env) {
  int tiles = env->size * env->size;
  for (int i = 0; i < tiles; i++)
    env->observations[i] = EMPTY;
  for (int i = 0; i < env->size; i++)
    env->observations[i] = AGENT;
  for (int i = 0; i < env->size; i++)
    env->observations[(env->size - 1) * env->size + i] = OPPONENT;
  env->r = 0;
  env->c = 0;
  env->tick = 0;
  env->terminals[0] = 0;
}

// Required function
void c_step(Hexapawn *env) {
  env->tick += 1;
  env->rewards[0] = 0.0;
  int action = (int)env->actions[0];
  Move move = decode_action(env, action);

  // Agent's move
  if (make_move(env, move, AGENT)) {
    env->rewards[0] += 0.0; // env->reward_move_valid;
    // check if game ended after agent move
    if (env->terminals[0] != 1) {
      scripted_opponent(env, 1);
    }
  } else {
    env->rewards[0] = env->reward_move_invalid;
  }

  if (env->rewards[0] > 1) {
    env->rewards[0] = 1;
  }
  if (env->rewards[0] < -1) {
    env->rewards[0] = -1;
  }

  if (env->terminals[0] == 1) {
    add_log(env);
    c_reset(env);
    return;
  }
}

// Required function. Should handle creating the client on first call
void c_render(Hexapawn *env) {
  const Color BG1 = (Color){27, 27, 27, 255};
  const Color BG2 = (Color){13, 13, 13, 255};

  int cell_size = 128;
  int window_width = cell_size * env->size;
  int window_height = cell_size * env->size;

  if (!IsWindowReady()) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(window_width, window_height, "Puffer Hexapawn");
    SetTargetFPS(30);
  } else if (GetScreenWidth() != window_width ||
             GetScreenHeight() != window_height) {
    SetWindowSize(window_width, window_height);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    CloseWindow();
    exit(0);
    return;
  }

  BeginDrawing();
  ClearBackground(BG1);

  for (int i = 0; i < env->size; i++) {
    for (int j = 0; j < env->size; j++) {
      int piece = env->observations[i * env->size + j];
      if ((i + j) % 2 == 0)
        DrawRectangle(j * cell_size - 1, i * cell_size - 1, cell_size + 1,
                      cell_size + 1, BG2);
      if (piece == EMPTY)
        continue;

      Color piece_color = (piece == AGENT) ? BLUE : RED;
      int center_x = j * cell_size + cell_size / 2;
      int center_y = i * cell_size + cell_size / 2;
      int radius = cell_size / 3;

      DrawCircle(center_x, center_y, radius, piece_color);
      DrawCircleGradient(center_x - radius / 3, center_y - radius / 3,
                         radius / 3, (Color){255, 255, 255, 80},
                         (Color){255, 255, 255, 10});
      DrawCircleGradient(center_x, center_y, radius, (Color){255, 255, 255, 50},
                         (Color){255, 255, 255, 5});
    }
  }

  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Hexapawn *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
