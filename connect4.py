import streamlit as st
import numpy as np
import random

# Game setup
ROWS = 10
COLS = 12
PLAYER = 1
AI = 2

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1

# Enlarged emoji chips
EMPTY = "⚪"
PLAYER_CHIP = "🔴"
AI_CHIP = "🟡"
LARGE_EMPTY = EMPTY + "\uFE0F"
LARGE_PLAYER_CHIP = PLAYER_CHIP + "\uFE0F" + "\u20DE"
LARGE_AI_CHIP = AI_CHIP + "\uFE0F" + "\u20DE"

def create_board():
    return np.zeros((ROWS, COLS), dtype=int)

def is_valid_location(board, col):
    return board[ROWS-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == 0:
            return r

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def winning_move(board, piece):
    # Check horizontal locations
    for c in range(COLS-3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations
    for c in range(COLS):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(COLS-3):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(COLS-3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

def render_board(board):
    rendered = ""
    for row in range(ROWS-1, -1, -1):
        rendered += "|"
        for col in range(COLS):
            if board[row][col] == 0:
                rendered += f" {LARGE_EMPTY} |"
            elif board[row][col] == 1:
                rendered += f" {LARGE_PLAYER_CHIP} |"
            else:
                rendered += f" {LARGE_AI_CHIP} |"
        rendered += "\n"
    rendered += "+" + "-----+" * COLS + "\n"
    rendered += "|" + "|".join([f" {i+1:2d} " for i in range(COLS)]) + "|\n"
    return rendered

class QLearningAI:
    def __init__(self):
        self.q_table = {}

    def get_state_key(self, board):
        return str(board.flatten())

    def get_action(self, board, epsilon):
        if random.random() < epsilon:
            return random.choice([col for col in range(COLS) if is_valid_location(board, col)])
        else:
            state_key = self.get_state_key(board)
            if state_key not in self.q_table:
                return random.choice([col for col in range(COLS) if is_valid_location(board, col)])
            return max(range(COLS), key=lambda col: self.q_table[state_key].get(col, 0) if is_valid_location(board, col) else float('-inf'))

    def update_q_table(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {}

        current_q = self.q_table[state_key].get(action, 0)
        next_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0

        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - current_q)
        self.q_table[state_key][action] = new_q

def play_game(ai, train=False):
    board = create_board()
    game_over = False
    turn = 0

    while not game_over:
        if turn % 2 == 0:
            col = ai.get_action(board, EPSILON if train else 0)
        else:
            col = ai.get_action(board, EPSILON if train else 0)

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, (turn % 2) + 1)

            if winning_move(board, (turn % 2) + 1):
                reward = 1 if turn % 2 == 0 else -1
                game_over = True
            elif np.all(board != 0):
                reward = 0
                game_over = True
            else:
                reward = 0

            if train:
                next_state = board.copy()
                ai.update_q_table(board, col, reward, next_state)

            turn += 1

    return board

# Streamlit app
st.title("Connect 4 with Q-Learning AI")

if 'ai' not in st.session_state:
    st.session_state.ai = QLearningAI()

if 'board' not in st.session_state:
    st.session_state.board = create_board()
    st.session_state.game_over = False

board_placeholder = st.empty()
board_placeholder.text(render_board(st.session_state.board))

cols = st.columns(COLS)
for i in range(COLS):
    if cols[i].button(str(i+1), key=f"col_{i}"):
        if not st.session_state.game_over and is_valid_location(st.session_state.board, i):
            row = get_next_open_row(st.session_state.board, i)
            drop_piece(st.session_state.board, row, i, PLAYER)
            board_placeholder.text(render_board(st.session_state.board))
            
            if winning_move(st.session_state.board, PLAYER):
                st.write("Player wins!")
                st.session_state.game_over = True
            elif np.all(st.session_state.board != 0):
                st.write("It's a tie!")
                st.session_state.game_over = True
            else:
                ai_col = st.session_state.ai.get_action(st.session_state.board, 0)
                ai_row = get_next_open_row(st.session_state.board, ai_col)
                drop_piece(st.session_state.board, ai_row, ai_col, AI)
                board_placeholder.text(render_board(st.session_state.board))

                if winning_move(st.session_state.board, AI):
                    st.write("AI wins!")
                    st.session_state.game_over = True
                elif np.all(st.session_state.board != 0):
                    st.write("It's a tie!")
                    st.session_state.game_over = True
            
            st.experimental_rerun()

if st.button("Reset Game"):
    st.session_state.board = create_board()
    st.session_state.game_over = False
    st.experimental_rerun()

if st.button("Train AI"):
    progress_bar = st.progress(0)
    for i in range(1000):
        play_game(st.session_state.ai, train=True)
        progress_bar.progress((i + 1) / 1000)
    st.write("AI training completed!")

if st.button("AI vs AI Game"):
    board = play_game(st.session_state.ai, train=False)
    st.text(render_board(board))
    if winning_move(board, PLAYER):
        st.write("AI 1 wins!")
    elif winning_move(board, AI):
        st.write("AI 2 wins!")
    else:
        st.write("It's a tie!")