#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
  float perf;
  float score;
  float episode_return;
  float episode_length;
  float n;
} Log;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
  Log log;
  unsigned char *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int size;
  int tick;
  int current_player;
  int agent_pieces;
  int opponent_pieces;
  int capture_available_cache;
  int capture_available_valid;
  int game_over_cache;
  int game_over_valid;
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
  int num_move_types = 8;
  int pos = action / num_move_types;
  int move_type = action % num_move_types;

  Move m;
  m.from.r = pos / env->size;
  m.from.c = pos % env->size;
  m.to.r = m.from.r;
  m.to.c = m.from.c;

  switch (move_type) {
  case 0:
    m.to.r = m.from.r - 1;
    m.to.c = m.from.c - 1;
    break;
  case 1:
    m.to.r = m.from.r - 1;
    m.to.c = m.from.c + 1;
    break;
  case 2:
    m.to.r = m.from.r + 1;
    m.to.c = m.from.c - 1;
    break;
  case 3:
    m.to.r = m.from.r + 1;
    m.to.c = m.from.c + 1;
    break;
  case 4:
    m.to.r = m.from.r - 2;
    m.to.c = m.from.c - 2;
    break;
  case 5:
    m.to.r = m.from.r - 2;
    m.to.c = m.from.c + 2;
    break;
  case 6:
    m.to.r = m.from.r + 2;
    m.to.c = m.from.c - 2;
    break;
  case 7:
    m.to.r = m.from.r + 2;
    m.to.c = m.from.c + 2;
    break;
  }

  return m;
}

int p2i(Checkers *env, Position p) { return p.r * env->size + p.c; }

int get_piece(Checkers *env, Position p) {
  if (!check_in_bounds(env, p)) {
    return EMPTY;
  }
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
  return m.to.r > m.from.r ? 1 : -1;
}

int valid_move_direction(Checkers *env, Move m) {
  int piece = get_piece(env, m.from);
  if (piece == AGENT_PAWN)
    return get_move_direction(env, m) == 1 ? 1 : 0;
  if (piece == OPPONENT_PAWN)
    return get_move_direction(env, m) == -1 ? 1 : 0;
  return 1;
}

int is_diagonal_move(Move m) {
  int dr = m.to.r - m.from.r;
  int dc = m.to.c - m.from.c;
  return (dr == dc) || (dr == -dc);
}
int move_size(Move m) { return abs(m.from.r - m.to.r); }

int is_valid_move_no_capture(Checkers *env, Move m) {
  if (m.from.r < 0 || m.from.c < 0 || m.to.r < 0 || m.to.c < 0) {
    return 0;
  }

  if (!check_in_bounds(env, m.from) || !check_in_bounds(env, m.to))
    return 0;

  if (get_piece_type(env, m.from) != env->current_player)
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
    int other_player = env->current_player == AGENT ? OPPONENT : AGENT;
    Position between_pos =
        (Position){(m.from.r + m.to.r) / 2, (m.from.c + m.to.c) / 2};
    if (get_piece_type(env, between_pos) != other_player)
      return 0;
  }

  return 1;
}

int capture_available(Checkers *env) {
  if (env->capture_available_valid) {
    return env->capture_available_cache;
  }

  int current_pawn = env->current_player == AGENT ? AGENT_PAWN : OPPONENT_PAWN;
  int current_king = env->current_player == AGENT ? AGENT_KING : OPPONENT_KING;

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece != current_pawn && piece != current_king)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    int directions[4][2] = {{-2, -2}, {-2, 2}, {2, -2}, {2, 2}};
    for (int d = 0; d < 4; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r < 0 || new_r >= env->size || new_c < 0 || new_c >= env->size)
        continue;

      if (env->observations[new_r * env->size + new_c] != EMPTY)
        continue;

      int mid_r = r + directions[d][0] / 2;
      int mid_c = c + directions[d][1] / 2;
      int mid_piece = env->observations[mid_r * env->size + mid_c];

      int opponent_pawn =
          env->current_player == AGENT ? OPPONENT_PAWN : AGENT_PAWN;
      int opponent_king =
          env->current_player == AGENT ? OPPONENT_KING : AGENT_KING;

      if (mid_piece == opponent_pawn || mid_piece == opponent_king) {
        if (piece == current_pawn) {
          int move_dir = directions[d][0] > 0 ? 1 : -1;
          int valid_dir = env->current_player == AGENT ? 1 : -1;
          if (move_dir != valid_dir)
            continue;
        }

        env->capture_available_cache = 1;
        env->capture_available_valid = 1;
        return 1;
      }
    }
  }

  env->capture_available_cache = 0;
  env->capture_available_valid = 1;
  return 0;
}

int is_valid_move(Checkers *env, Move m) {
  if (capture_available(env) && move_size(m) != 2)
    return 0;
  return is_valid_move_no_capture(env, m);
}

int num_legal_moves(Checkers *env) {
  int res = 0;
  int current_pawn = env->current_player == AGENT ? AGENT_PAWN : OPPONENT_PAWN;
  int current_king = env->current_player == AGENT ? AGENT_KING : OPPONENT_KING;
  int has_captures = capture_available(env);

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece != current_pawn && piece != current_king)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    int directions[8][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1},
                            {-2, -2}, {-2, 2}, {2, -2}, {2, 2}};

    for (int d = 0; d < 8; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r < 0 || new_r >= env->size || new_c < 0 || new_c >= env->size)
        continue;

      if (env->observations[new_r * env->size + new_c] != EMPTY)
        continue;

      int move_size = abs(directions[d][0]);

      if (has_captures && move_size != 2)
        continue;

      if (piece == current_pawn) {
        int move_dir = directions[d][0] > 0 ? 1 : -1;
        int valid_dir = env->current_player == AGENT ? 1 : -1;
        if (move_dir != valid_dir)
          continue;
      }

      if (move_size == 2) {
        int mid_r = r + directions[d][0] / 2;
        int mid_c = c + directions[d][1] / 2;
        int mid_piece = env->observations[mid_r * env->size + mid_c];

        int opponent_pawn =
            env->current_player == AGENT ? OPPONENT_PAWN : AGENT_PAWN;
        int opponent_king =
            env->current_player == AGENT ? OPPONENT_KING : AGENT_KING;

        if (mid_piece != opponent_pawn && mid_piece != opponent_king)
          continue;
      }

      res++;
    }
  }

  return res;
}

int num_pieces_by_player(Checkers *env, int player) {
  if (player == AGENT) {
    return env->agent_pieces;
  } else {
    return env->opponent_pieces;
  }
}

int try_make_king(Checkers *env) {
  int promoted = 0;

  for (int i = 0; i < env->size; i++) {
    if (env->observations[i] == OPPONENT_PAWN) {
      env->observations[i] = OPPONENT_KING;
      promoted = 1;
    }
  }
  for (int i = 0; i < env->size; i++) {
    if (env->observations[env->size * (env->size - 1) + i] == AGENT_PAWN) {
      env->observations[env->size * (env->size - 1) + i] = AGENT_KING;
      promoted = 1;
    }
  }

  if (promoted) {
    env->capture_available_valid = 0;
    env->game_over_valid = 0;
  }

  return promoted;
}

int is_game_over(Checkers *env) {
  if (env->game_over_valid) {
    return env->game_over_cache;
  }

  int current_player_pieces = num_pieces_by_player(env, env->current_player);
  int other_player = env->current_player == AGENT ? OPPONENT : AGENT;
  int other_player_pieces = num_pieces_by_player(env, other_player);

  if (current_player_pieces == 0 || other_player_pieces == 0) {
    env->game_over_cache = 1;
    env->game_over_valid = 1;
    return 1;
  }

  int has_captures = capture_available(env);
  if (has_captures) {
    env->game_over_cache = 0;
    env->game_over_valid = 1;
    return 0;
  }

  int current_pawn = env->current_player == AGENT ? AGENT_PAWN : OPPONENT_PAWN;
  int current_king = env->current_player == AGENT ? AGENT_KING : OPPONENT_KING;

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece != current_pawn && piece != current_king)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    int directions[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    for (int d = 0; d < 4; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r < 0 || new_r >= env->size || new_c < 0 || new_c >= env->size)
        continue;
      if (env->observations[new_r * env->size + new_c] != EMPTY)
        continue;

      if (piece == current_pawn) {
        int move_dir = directions[d][0] > 0 ? 1 : -1;
        int valid_dir = env->current_player == AGENT ? 1 : -1;
        if (move_dir != valid_dir)
          continue;
      }

      env->game_over_cache = 0;
      env->game_over_valid = 1;
      return 0;
    }
  }

  env->game_over_cache = 1;
  env->game_over_valid = 1;
  return 1;
}

// Helper function to determine who won the game
int get_winner(Checkers *env) {
  int agent_pieces = num_pieces_by_player(env, AGENT);
  int opponent_pieces = num_pieces_by_player(env, OPPONENT);

  if (agent_pieces == 0) {
    return OPPONENT;
  }

  if (opponent_pieces == 0) {
    return AGENT;
  }

  if (is_game_over(env)) {
    return env->current_player == AGENT ? OPPONENT : AGENT;
  }

  return EMPTY;
}

void make_move(Checkers *env, int action) {
  Move m = decode_action(env, action);
  if (!is_valid_move(env, m)) {
    env->rewards[0] = -1.0f; // Penalty for invalid move
    return;
  }

  int moving_piece = get_piece(env, m.from);
  env->observations[p2i(env, m.from)] = EMPTY;
  env->observations[p2i(env, m.to)] = moving_piece;

  // Track if a capture occurred for intermediate reward
  int capture_occurred = 0;
  float reward = 0.0f; // Initialize reward accumulator

  if (move_size(m) == 2) {
    Position between_pos =
        (Position){(m.from.r + m.to.r) / 2, (m.from.c + m.to.c) / 2};
    int captured_piece = env->observations[p2i(env, between_pos)];
    env->observations[p2i(env, between_pos)] = EMPTY;
    capture_occurred = 1;

    if (captured_piece == AGENT_PAWN || captured_piece == AGENT_KING) {
      env->agent_pieces--;
      reward -= 0.05f; // Small negative reward for losing pieces
    } else if (captured_piece == OPPONENT_PAWN ||
               captured_piece == OPPONENT_KING) {
      env->opponent_pieces--;
    }
  }

  env->capture_available_valid = 0;
  env->game_over_valid = 0;

  // Track if promotion occurred for intermediate reward
  int promotion_occurred = try_make_king(env);

  if (move_size(m) == 1 || !capture_available(env)) {
    int other_player = env->current_player == AGENT ? OPPONENT : AGENT;
    env->current_player = other_player;
  }

  // Assign intermediate rewards
  if (capture_occurred && env->current_player == OPPONENT) {
    // Agent just made a capture, give reward
    reward += 0.1f; // Small positive reward for capturing
  } else if (env->current_player == OPPONENT) {
    // Agent made a successful move (no capture)
    reward += 0.01f; // Very small positive reward for successful moves
  }

  if (promotion_occurred) {
    // Check if agent was promoted
    for (int i = 0; i < env->size; i++) {
      if (env->observations[env->size * (env->size - 1) + i] == AGENT_KING) {
        reward += 0.05f; // Small reward for promotion
        break;
      }
    }
  }

  if (is_game_over(env)) {
    env->terminals[0] = 1;
    int winner = get_winner(env);
    reward = winner == AGENT
                 ? 1.0f
                 : -1.0f; // Game over rewards override intermediate rewards
  }

  // Ensure reward stays within bounds
  env->rewards[0] = clamp(reward, -1.0f, 1.0f);
}

void scripted_first_move(Checkers *env) {
  int current_pawn = env->current_player == AGENT ? AGENT_PAWN : OPPONENT_PAWN;
  int current_king = env->current_player == AGENT ? AGENT_KING : OPPONENT_KING;
  int has_captures = capture_available(env);

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece != current_pawn && piece != current_king)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    int directions[8][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1},
                            {-2, -2}, {-2, 2}, {2, -2}, {2, 2}};

    for (int d = 0; d < 8; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r < 0 || new_r >= env->size || new_c < 0 || new_c >= env->size)
        continue;
      if (env->observations[new_r * env->size + new_c] != EMPTY)
        continue;

      int move_size = abs(directions[d][0]);

      if (has_captures && move_size != 2)
        continue;

      if (piece == current_pawn) {
        int move_dir = directions[d][0] > 0 ? 1 : -1;
        int valid_dir = env->current_player == AGENT ? 1 : -1;
        if (move_dir != valid_dir)
          continue;
      }

      if (move_size == 2) {
        int mid_r = r + directions[d][0] / 2;
        int mid_c = c + directions[d][1] / 2;
        int mid_piece = env->observations[mid_r * env->size + mid_c];

        int opponent_pawn =
            env->current_player == AGENT ? OPPONENT_PAWN : AGENT_PAWN;
        int opponent_king =
            env->current_player == AGENT ? OPPONENT_KING : AGENT_KING;

        if (mid_piece != opponent_pawn && mid_piece != opponent_king)
          continue;
      }

      int action = i * 8 + d;
      make_move(env, action);
      return;
    }
  }
}

// Helper function to evaluate position value
float evaluate_position(Checkers *env) {
  float score = 0.0f;

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    int r = i / env->size;
    int c = i % env->size;

    if (piece == AGENT_PAWN) {
      score += 1.0f + (r * 0.1f); // Pawns are worth more as they advance
    } else if (piece == AGENT_KING) {
      score += 2.0f; // Kings are worth more
    } else if (piece == OPPONENT_PAWN) {
      score -= 1.0f + ((env->size - 1 - r) * 0.1f);
    } else if (piece == OPPONENT_KING) {
      score -= 2.0f;
    }
  }

  return score;
}

// Helper function to check if a move leads to immediate capture opportunity for
// opponent
int move_leads_to_capture(Checkers *env, Move m) {
  // Temporarily make the move
  int moving_piece = get_piece(env, m.from);
  env->observations[p2i(env, m.from)] = EMPTY;
  env->observations[p2i(env, m.to)] = moving_piece;

  if (move_size(m) == 2) {
    Position between_pos =
        (Position){(m.from.r + m.to.r) / 2, (m.from.c + m.to.c) / 2};
    env->observations[p2i(env, between_pos)] = EMPTY;
  }

  // Check if opponent can capture this piece
  int current_player_backup = env->current_player;
  env->current_player = env->current_player == AGENT ? OPPONENT : AGENT;

  int can_be_captured = 0;
  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece == EMPTY)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    // Check if this piece can capture the moved piece
    int directions[4][2] = {{-2, -2}, {-2, 2}, {2, -2}, {2, 2}};
    for (int d = 0; d < 4; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r == m.to.r && new_c == m.to.c) {
        int mid_r = r + directions[d][0] / 2;
        int mid_c = c + directions[d][1] / 2;
        int mid_piece = env->observations[mid_r * env->size + mid_c];

        if (mid_piece == moving_piece) {
          can_be_captured = 1;
          break;
        }
      }
    }
    if (can_be_captured)
      break;
  }

  // Restore the board
  env->observations[p2i(env, m.from)] = moving_piece;
  env->observations[p2i(env, m.to)] = EMPTY;

  if (move_size(m) == 2) {
    Position between_pos =
        (Position){(m.from.r + m.to.r) / 2, (m.from.c + m.to.c) / 2};
    int captured_piece =
        env->current_player == AGENT ? OPPONENT_PAWN : AGENT_PAWN;
    env->observations[p2i(env, between_pos)] = captured_piece;
  }

  env->current_player = current_player_backup;
  return can_be_captured;
}

// Helper function to get all valid moves for current player
typedef struct {
  Move moves[100];
  int count;
} MoveList;

MoveList get_all_valid_moves(Checkers *env) {
  MoveList moves;
  moves.count = 0;

  int current_pawn = env->current_player == AGENT ? AGENT_PAWN : OPPONENT_PAWN;
  int current_king = env->current_player == AGENT ? AGENT_KING : OPPONENT_KING;
  int has_captures = capture_available(env);

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece != current_pawn && piece != current_king)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    int directions[8][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1},
                            {-2, -2}, {-2, 2}, {2, -2}, {2, 2}};

    for (int d = 0; d < 8; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r < 0 || new_r >= env->size || new_c < 0 || new_c >= env->size)
        continue;
      if (env->observations[new_r * env->size + new_c] != EMPTY)
        continue;

      int move_size = abs(directions[d][0]);

      if (has_captures && move_size != 2)
        continue;

      if (piece == current_pawn) {
        int move_dir = directions[d][0] > 0 ? 1 : -1;
        int valid_dir = env->current_player == AGENT ? 1 : -1;
        if (move_dir != valid_dir)
          continue;
      }

      if (move_size == 2) {
        int mid_r = r + directions[d][0] / 2;
        int mid_c = c + directions[d][1] / 2;
        int mid_piece = env->observations[mid_r * env->size + mid_c];

        int opponent_pawn =
            env->current_player == AGENT ? OPPONENT_PAWN : AGENT_PAWN;
        int opponent_king =
            env->current_player == AGENT ? OPPONENT_KING : AGENT_KING;

        if (mid_piece != opponent_pawn && mid_piece != opponent_king)
          continue;
      }

      Move m;
      m.from.r = r;
      m.from.c = c;
      m.to.r = new_r;
      m.to.c = new_c;

      if (moves.count < 100) {
        moves.moves[moves.count] = m;
        moves.count++;
      }
    }
  }

  return moves;
}

// Helper function to evaluate a move
float evaluate_move(Checkers *env, Move m) {
  float score = 0.0f;

  // Prioritize captures
  if (move_size(m) == 2) {
    score += 10.0f;
  }

  // Prioritize king moves (they're more valuable)
  int piece = get_piece(env, m.from);
  if (piece == AGENT_KING || piece == OPPONENT_KING) {
    score += 2.0f;
  }

  // Prefer moves that advance pawns toward promotion
  if (piece == AGENT_PAWN && env->current_player == AGENT) {
    score += (m.to.r - m.from.r) * 0.5f; // Moving forward is good
  } else if (piece == OPPONENT_PAWN && env->current_player == OPPONENT) {
    score += (m.from.r - m.to.r) * 0.5f; // Moving forward is good
  }

  // Prefer center control
  int center_distance_from =
      abs(m.from.r - env->size / 2) + abs(m.from.c - env->size / 2);
  int center_distance_to =
      abs(m.to.r - env->size / 2) + abs(m.to.c - env->size / 2);
  score += (center_distance_from - center_distance_to) * 0.1f;

  // Avoid moves that lead to immediate capture
  if (move_leads_to_capture(env, m)) {
    score -= 5.0f;
  }

  // Prefer moves that protect pieces
  if (piece == AGENT_PAWN || piece == AGENT_KING) {
    // Check if this move protects other pieces
    int directions[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    for (int d = 0; d < 4; d++) {
      int protect_r = m.to.r + directions[d][0];
      int protect_c = m.to.c + directions[d][1];

      if (protect_r >= 0 && protect_r < env->size && protect_c >= 0 &&
          protect_c < env->size) {
        int protected_piece =
            env->observations[protect_r * env->size + protect_c];
        if (protected_piece == AGENT_PAWN || protected_piece == AGENT_KING) {
          score += 0.5f;
        }
      }
    }
  }

  return score;
}

void scripted_strong_move(Checkers *env) {
  MoveList moves = get_all_valid_moves(env);

  if (moves.count == 0)
    return;

  // Find the best move
  float best_score = -1000.0f;
  Move best_move = moves.moves[0];

  for (int i = 0; i < moves.count; i++) {
    float score = evaluate_move(env, moves.moves[i]);
    if (score > best_score) {
      best_score = score;
      best_move = moves.moves[i];
    }
  }

  // Convert move to action
  int from_pos = p2i(env, best_move.from);
  int to_pos = p2i(env, best_move.to);

  int dr = best_move.to.r - best_move.from.r;
  int dc = best_move.to.c - best_move.from.c;

  int move_type = -1;
  if (dr == -1 && dc == -1)
    move_type = 0;
  else if (dr == -1 && dc == 1)
    move_type = 1;
  else if (dr == 1 && dc == -1)
    move_type = 2;
  else if (dr == 1 && dc == 1)
    move_type = 3;
  else if (dr == -2 && dc == -2)
    move_type = 4;
  else if (dr == -2 && dc == 2)
    move_type = 5;
  else if (dr == 2 && dc == -2)
    move_type = 6;
  else if (dr == 2 && dc == 2)
    move_type = 7;

  if (move_type >= 0) {
    int action = from_pos * 8 + move_type;
    make_move(env, action);
  }
}

void scripted_expert_move(Checkers *env) {
  // Expert level: look ahead one move and consider opponent's best response
  MoveList moves = get_all_valid_moves(env);

  if (moves.count == 0)
    return;

  float best_score = -1000.0f;
  Move best_move = moves.moves[0];

  for (int i = 0; i < moves.count; i++) {
    // Temporarily make this move
    int moving_piece = get_piece(env, moves.moves[i].from);
    env->observations[p2i(env, moves.moves[i].from)] = EMPTY;
    env->observations[p2i(env, moves.moves[i].to)] = moving_piece;

    if (move_size(moves.moves[i]) == 2) {
      Position between_pos =
          (Position){(moves.moves[i].from.r + moves.moves[i].to.r) / 2,
                     (moves.moves[i].from.c + moves.moves[i].to.c) / 2};
      env->observations[p2i(env, between_pos)] = EMPTY;
    }

    // Switch to opponent's turn
    int current_player_backup = env->current_player;
    env->current_player = env->current_player == AGENT ? OPPONENT : AGENT;

    // Find opponent's best move
    MoveList opponent_moves = get_all_valid_moves(env);
    float opponent_best_score = -1000.0f;

    for (int j = 0; j < opponent_moves.count; j++) {
      float score = evaluate_move(env, opponent_moves.moves[j]);
      if (score > opponent_best_score) {
        opponent_best_score = score;
      }
    }

    // Restore board
    env->observations[p2i(env, moves.moves[i].from)] = moving_piece;
    env->observations[p2i(env, moves.moves[i].to)] = EMPTY;

    if (move_size(moves.moves[i]) == 2) {
      Position between_pos =
          (Position){(moves.moves[i].from.r + moves.moves[i].to.r) / 2,
                     (moves.moves[i].from.c + moves.moves[i].to.c) / 2};
      int captured_piece =
          current_player_backup == AGENT ? OPPONENT_PAWN : AGENT_PAWN;
      env->observations[p2i(env, between_pos)] = captured_piece;
    }

    env->current_player = current_player_backup;

    // Score this move based on position after opponent's best response
    float move_score =
        evaluate_move(env, moves.moves[i]) - opponent_best_score * 0.5f;

    if (move_score > best_score) {
      best_score = move_score;
      best_move = moves.moves[i];
    }
  }

  // Convert move to action
  int from_pos = p2i(env, best_move.from);
  int dr = best_move.to.r - best_move.from.r;
  int dc = best_move.to.c - best_move.from.c;

  int move_type = -1;
  if (dr == -1 && dc == -1)
    move_type = 0;
  else if (dr == -1 && dc == 1)
    move_type = 1;
  else if (dr == 1 && dc == -1)
    move_type = 2;
  else if (dr == 1 && dc == 1)
    move_type = 3;
  else if (dr == -2 && dc == -2)
    move_type = 4;
  else if (dr == -2 && dc == 2)
    move_type = 5;
  else if (dr == 2 && dc == -2)
    move_type = 6;
  else if (dr == 2 && dc == 2)
    move_type = 7;

  if (move_type >= 0) {
    int action = from_pos * 8 + move_type;
    make_move(env, action);
  }
}

void scripted_random_move(Checkers *env) { scripted_first_move(env); }

void scripted_step(Checkers *env, int difficulty) {
  switch (difficulty) {
  case 0:
    scripted_first_move(env);
    break;
  case 1:
    scripted_random_move(env);
    break;
  default:
    scripted_random_move(env);
    break;
  }
}

void update_piece_counts(Checkers *env) {
  env->agent_pieces = 0;
  env->opponent_pieces = 0;

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece == AGENT_PAWN || piece == AGENT_KING) {
      env->agent_pieces++;
    } else if (piece == OPPONENT_PAWN || piece == OPPONENT_KING) {
      env->opponent_pieces++;
    }
  }

  env->capture_available_valid = 0;
  env->game_over_valid = 0;
}

void add_log(Checkers *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n += 1;
}

// Required function
void c_reset(Checkers *env) {
  env->tick = 0;
  env->terminals[0] = 0;
  env->rewards[0] = 0.0f;

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

  env->current_player = AGENT;

  update_piece_counts(env);
}

// Required function
void c_step(Checkers *env) {
  env->tick += 1;
  int action = env->actions[0];
  env->rewards[0] = 0.0f;
  env->terminals[0] = 0;

  make_move(env, action);

  env->rewards[0] = clamp(env->rewards[0], -1.0f, 1.0f);
  if (env->terminals[0] == 1) {
    add_log(env);
    c_reset(env);
    return;
  }

  scripted_step(env, 1);
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

  int cell_size = 64;
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
  if (IsWindowReady()) {
    CloseWindow();
  }
}
