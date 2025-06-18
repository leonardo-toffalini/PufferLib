#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// NOTE: actions and observations could take up half the space, bc checkers is
// only played on every other cell

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char OPPONENT = 3;
const unsigned char AGENT_PAWN = 1;
const unsigned char AGENT_KING = 2;
const unsigned char OPPONENT_PAWN = 3;
const unsigned char OPPONENT_KING = 4;

float clamp(float x, float low, float high) {
  return fminf(high, fmaxf(low, x));
}

// Required struct. Only use floats!
typedef struct {
  float perf;  // Recommended 0-1 normalized single real number perf metric
  float score; // Recommended unnormalized single real number perf metric
  float episode_return; // Recommended metric: sum of agent rewards over episode
  float episode_length; // Recommended metric: number of steps of agent episode
  // Any extra fields you add here may be exported to Python in binding.c
  float n; // Required as the last field
} Log;

typedef struct {
  unsigned char current_player;
  // ...
} GameState;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
  Log log; // Required field. Env binding code uses this to aggregate logs
  unsigned char *observations; // Required. You can use any obs type, but make
                               // sure it matches in Python!
  int *actions;   // Required. int* for discrete/multidiscrete, float* for box
  float *rewards; // Required
  unsigned char
      *terminals; // Required. We don't yet have truncations as standard yet
  int size;
  int tick;
  GameState *game_state;
} Checkers;

typedef struct {
  int r;
  int c;
} Position;

typedef struct {
  Position from;
  Position to;
} Move;

Move decode_action(Checkers *env, int action) {
  // Action space is size*size*8 where 8 is number of move types
  // Move types are: NW, NE, SW, SE, 2*NW, 2*NE, 2*SW, 2*SE,
  // 4 = 0*8 + 3 => zeroth cell, second move type
  // 10 = 1*8 + 2 => first cell, second move type
  // 50 = 6*8 + 2 => sixth cell, second move type
  int num_move_types = 8;
  int pos = action / num_move_types;
  int move_type = action % num_move_types;

  Move m;
  m.from.r = pos / env->size;
  m.from.c = pos % env->size;
  m.to.r = m.from.r;
  m.to.c = m.from.c;

  // Decode move type into target position
  switch (move_type) {
  case 0: // NW
    m.to.r = m.from.r - 1;
    m.to.c = m.from.c - 1;
    break;
  case 1: // NE
    m.to.r = m.from.r - 1;
    m.to.c = m.from.c + 1;
    break;
  case 2: // SW
    m.to.r = m.from.r + 1;
    m.to.c = m.from.c - 1;
    break;
  case 3: // SE
    m.to.r = m.from.r + 1;
    m.to.c = m.from.c + 1;
    break;
  case 4: // 2*NW
    m.to.r = m.from.r - 2;
    m.to.c = m.from.c - 2;
    break;
  case 5: // 2*NE
    m.to.r = m.from.r - 2;
    m.to.c = m.from.c + 2;
    break;
  case 6: // 2*SW
    m.to.r = m.from.r + 2;
    m.to.c = m.from.c - 2;
    break;
  case 7: // 2*SE
    m.to.r = m.from.r + 2;
    m.to.c = m.from.c + 2;
    break;
  }

  return m;
}

int p2i(Checkers *env, Position p) {
  // position to index
  return p.r * env->size + p.c;
}

int get_piece(Checkers *env, Position p) {
  return env->observations[p2i(env, p)];
}

int get_piece_type(Checkers *env, Position p) {
  int piece = get_piece(env, p);
  if (piece == AGENT_PAWN || piece == AGENT_KING)
    return AGENT;
  if (piece == OPPONENT_PAWN || piece == OPPONENT_KING)
    return OPPONENT;
  return EMPTY;
}

int check_in_bounds(Checkers *env, Position p) {
  return 0 <= p.r && p.r < env->size && 0 <= p.c && p.c < env->size;
}

int get_move_direction(Checkers *env, Move m) {
  // return +1 if the move is visually downwards, -1 otherwise
  return m.to.r > m.from.r ? 1 : -1;
}

int valid_move_direction(Checkers *env, Move m) {
  int piece = get_piece(env, m.from);
  if (piece == AGENT_PAWN)
    return get_move_direction(env, m) == 1 ? 1 : 0;
  if (piece == OPPONENT_PAWN)
    return get_move_direction(env, m) == -1 ? 1 : 0;
  return 1; // kings can move in any direction
}

int is_diagonal_move(Move m) { return m.to.r - m.from.r == m.to.c - m.from.c; }
int move_size(Move m) { return abs(m.from.r - m.to.r); }

int is_valid_move(Checkers *env, Move m) {
  // checks:
  // - m.from and m.to are in bounds
  // - m.from is a piece of the same type as the player on turn
  // - m.to is empty
  // - move direction is forwards for a pawn
  // - move is diagonal
  // - move size is 1 or 2
  // - if its not a capture only move one diagonal
  // - if its a capture move two diagonal and theres a piece in between
  if (!check_in_bounds(env, m.from) || !check_in_bounds(env, m.to))
    return 0;

  if (get_piece_type(env, m.from) != env->game_state->current_player)
    return 0;

  if (get_piece(env, m.to) != EMPTY)
    return 0;

  if (!valid_move_direction(env, m))
    return 0;

  if (!is_diagonal_move(m))
    return 0;

  if (move_size(m) != 1 && move_size(m) != 2)
    return 0;

  if (move_size(m) == 2) {
    int other_player =
        env->game_state->current_player == AGENT ? OPPONENT : AGENT;
    Position between_pos =
        (Position){(m.from.r + m.to.r) / 2, (m.from.c + m.to.c) / 2};
    if (get_piece_type(env, between_pos) != other_player)
      return 0;
  }

  return 1;
}

void make_move(Checkers *env, int action) {
  Move m = decode_action(env, action);
  if (!is_valid_move(env, m)) {
    env->rewards[0] += 0.0f; // illegal move penality
    return;                  // nothing happens if an illegal move is made
  }
  int moving_piece = get_piece(env, m.from);
  env->observations[p2i(env, m.from)] = EMPTY;
  env->observations[p2i(env, m.to)] = moving_piece;
  if (move_size(m) == 2) {
    Position between_pos =
        (Position){(m.from.r + m.to.r) / 2, (m.from.c + m.to.c) / 2};
    env->observations[p2i(env, between_pos)] = EMPTY;
    return;
  }
  int other_player =
      env->game_state->current_player == AGENT ? OPPONENT : AGENT;
  env->game_state->current_player = other_player;
}

void add_log(Checkers *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

// Required function
void c_reset(Checkers *env) {
  int tiles = env->size * env->size;
  for (int i = 0; i < tiles; i++)
    env->observations[i] = EMPTY;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < env->size; j++) {
      if ((i + j) % 2)
        env->observations[i * env->size + j] = AGENT_PAWN;
    }
  }
  for (int i = env->size - 3; i < env->size; i++) {
    for (int j = 0; j < env->size; j++) {
      if ((i + j) % 2)
        env->observations[i * env->size + j] = OPPONENT_PAWN;
    }
  }
  env->tick = 0;
  env->terminals[0] = 0;
  env->game_state = (GameState *)malloc(sizeof(GameState));
  env->game_state->current_player = AGENT;
}

// Required function
void c_step(Checkers *env) {
  env->tick += 1;
  env->rewards[0] = 0.0f;
  env->terminals[0] = 0;
  int action = env->actions[0];
  make_move(env, action);

  env->rewards[0] = clamp(env->rewards[0], -1.0f, 1.0f);
  if (env->terminals[0] == 1) {
    add_log(env);
    c_reset(env);
    return;
  }
}

// Required function. Should handle creating the client on first call
void c_render(Checkers *env) {
  const Color BG1 = (Color){27, 27, 27, 255};
  const Color BG2 = (Color){13, 13, 13, 255};

  int cell_size = 128;
  int window_width = cell_size * env->size;
  int window_height = cell_size * env->size;
  int radius = cell_size / 3;
  int king_offset = 14;

  if (!IsWindowReady()) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(window_width, window_height, "Puffer Checkers");
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

  Color piece_color;
  for (int i = 0; i < env->size; i++) {
    for (int j = 0; j < env->size; j++) {
      int piece = env->observations[i * env->size + j];
      if ((i + j) % 2 == 0)
        DrawRectangle(j * cell_size - 1, i * cell_size - 1, cell_size + 1,
                      cell_size + 1, BG2);
      if (piece == EMPTY)
        continue;

      int center_x = j * cell_size + cell_size / 2;
      int center_y = i * cell_size + cell_size / 2;

      switch (piece) {
      case AGENT_PAWN:
        piece_color = BLUE;
        DrawCircle(center_x, center_y, radius, piece_color);
        DrawCircleGradient(center_x - radius / 3, center_y - radius / 3,
                           radius / 3, (Color){255, 255, 255, 80},
                           (Color){255, 255, 255, 10});
        DrawCircleGradient(center_x, center_y, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});
        break;

      case AGENT_KING:
        piece_color = BLUE;
        DrawCircle(center_x, center_y, radius, piece_color);
        DrawCircleGradient(center_x, center_y, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});

        DrawCircleGradient(center_x, center_y - king_offset / 2, radius,
                           (Color){20, 20, 20, 60}, (Color){20, 20, 20, 30});
        DrawCircle(center_x, center_y - king_offset, radius, piece_color);
        DrawCircleGradient(
            center_x - radius / 3, center_y - radius / 3 - king_offset,
            radius / 3, (Color){255, 255, 255, 80}, (Color){255, 255, 255, 10});
        DrawCircleGradient(center_x, center_y - king_offset, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});
        break;

      case OPPONENT_PAWN:
        piece_color = RED;
        DrawCircle(center_x, center_y, radius, piece_color);
        DrawCircleGradient(center_x - radius / 3, center_y - radius / 3,
                           radius / 3, (Color){255, 255, 255, 80},
                           (Color){255, 255, 255, 10});
        DrawCircleGradient(center_x, center_y, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});
        break;

      case OPPONENT_KING:
        piece_color = RED;
        DrawCircle(center_x, center_y, radius, piece_color);
        DrawCircleGradient(center_x, center_y, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});

        DrawCircleGradient(center_x, center_y - king_offset / 2, radius,
                           (Color){20, 20, 20, 60}, (Color){20, 20, 20, 30});
        DrawCircle(center_x, center_y - king_offset, radius, piece_color);
        DrawCircleGradient(
            center_x - radius / 3, center_y - radius / 3 - king_offset,
            radius / 3, (Color){255, 255, 255, 80}, (Color){255, 255, 255, 10});
        DrawCircleGradient(center_x, center_y - king_offset, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});
        break;

      default:
        break;
      }
    }
  }

  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Checkers *env) {
  free(env->game_state);
  if (IsWindowReady()) {
    CloseWindow();
  }
}
