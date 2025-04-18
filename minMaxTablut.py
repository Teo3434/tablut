import numpy as np
import struct
import math
import socket
import json
from enum import Enum
import random
import os
import sys
import time
#from google.colab import files
# Define Pawn and Turn enums
class Pawn(Enum):
    EMPTY = "O"
    WHITE = "W"
    BLACK = "B"
    THRONE = "T"
    KING = "K"

    @staticmethod
    def from_string(s):
        mapping = {"O": Pawn.EMPTY, "W": Pawn.WHITE, "B": Pawn.BLACK, "T": Pawn.THRONE, "K": Pawn.KING}
        return mapping.get(s, Pawn.EMPTY)

class Turn(Enum):
    WHITE = "W"
    BLACK = "B"
    WHITEWIN = "WW"
    BLACKWIN = "BW"
    DRAW = "D"

    def __str__(self):
        return self.value

class State:
    def __init__(self, board=None, turn=Turn.WHITE, history=None, repeated_moves_allowed=3):
        if board is None:
            self.board = self._initialize_board()
        else:
            self.board = board
        self.turn = turn
        self.history = history if history is not None else []
        self.repeated_moves_allowed = repeated_moves_allowed
        self.moves_without_capture = 0
        self._legal_moves = None

    def _initialize_board(self):
        board = np.full((9, 9), Pawn.EMPTY)
        # Throne
        board[4, 4] = Pawn.THRONE
        # King
        board[4, 4] = Pawn.KING
        # White defenders
        for r, c in [(2, 4), (4 , 2),(4, 3), (4, 5), (4, 6), (3, 4), (5, 4), (6, 4)]:
            board[r, c] = Pawn.WHITE
        # Black attackers
        for r, c in [(0, 3), (0, 4), (0, 5),(1, 4), (8, 3), (8, 4), (8, 5), (7, 4),
                     (3, 0), (4, 0), (5, 0), (4, 1), (3, 8), (4, 8), (5, 8), (4, 7)]:
            board[r, c] = Pawn.BLACK
        return board

    def get_board(self):
        return self.board

    def get_turn(self):
        if self.is_game_over():
            return self.is_game_over()
        return self.turn

    def set_turn(self, turn):
        self.turn = turn

    def board_string(self):
        return "\n".join("".join(pawn.value for pawn in row) for row in self.board)

    def get_pawn(self, row, col):
        return self.board[row, col]

    def clone(self):
        return State(np.copy(self.board), self.turn, self.history[:], self.repeated_moves_allowed)

    def is_draw(self):
        if len(self.history) < 2 * self.repeated_moves_allowed:
            return False  # Non ci sono abbastanza stati per un pareggio

        # Controlla se gli ultimi 'repeated_moves_allowed' stati si ripetono
        last_states = self.history[-self.repeated_moves_allowed:]
        for state in last_states:
            if self.history.count(state) > self.repeated_moves_allowed:
                return True  # Pareggio trovato

        return False  # Nessun pareggio trovato

    def get_legal_moves(self):
        if self._legal_moves is None:
            moves = []
            player = self.turn
            pawn_positions = np.where(
                (self.board == Pawn.WHITE) | (self.board == Pawn.KING)
                if player == Turn.WHITE
                else (self.board == Pawn.BLACK)
            )
            rows, cols = pawn_positions

            citadels = {(0, 3), (0, 4), (0, 5), (1, 4), (3, 0), (4, 0), (5, 0), (4, 1),
                        (3, 8), (4, 8), (5, 8), (4, 7), (8, 3), (8, 4), (8, 5), (7, 4)}
            throne = (4, 4)

            for idx in range(len(rows)):
                row, col = int(rows[idx]), int(cols[idx])
                pawn = self.board[row, col]
                is_from_citadel = (row, col) in citadels

                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

                for dr, dc in directions:
                    r, c = row, col
                    while True:
                        r, c = r + dr, c + dc
                        if not (0 <= r < 9 and 0 <= c < 9) or self.board[r, c] != Pawn.EMPTY:
                            break
                        if (r, c) == throne:
                            continue
                        if (r, c) in citadels and pawn in [Pawn.WHITE, Pawn.KING]:
                            break
                        # Impedisci l'ingresso in una cittadella da una non-citadella
                        if (r, c) in citadels and not is_from_citadel:
                            break
                        # Limita la distanza tra due cittadelle a 5
                        if (r, c) in citadels and is_from_citadel:
                            distance = abs(r - row) if dr != 0 else abs(c - col)
                            if distance > 5:
                                break
                        moves.append((row, col, r, c))
            self._legal_moves = moves
        return self._legal_moves

    def apply_move(self, move,real=False):
        from_row, from_col, to_row, to_col = move
        piece = self.board[from_row, from_col]
        self.board[from_row, from_col] = Pawn.EMPTY if (from_row, from_col) != (4, 4) else Pawn.THRONE
        self.board[to_row, to_col] = piece
        if real==True:
            self._handle_captures(to_row, to_col,real=True)
        else:
            self._handle_captures(to_row, to_col,real=False)
        self.turn = Turn.BLACK if self.turn == Turn.WHITE else Turn.WHITE
        self.history.append(self.board.tobytes())
        self._legal_moves = None

    def _handle_captures(self, row, col,real=True):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        moved_piece = self.board[row, col]
        citadels = {(0, 3), (0, 4), (0, 5), (1, 4), (3, 0), (4, 0), (5, 0), (4, 1),
                    (3, 8), (4, 8), (5, 8), (4, 7), (8, 3), (8, 4), (8, 5), (7, 4)}
        excluded_citadels = {(4, 8), (0, 4), (8, 4), (4, 0)}
        throne = (4, 4)

        for dr, dc in directions:
            r1, c1 = row + dr, col + dc
            r2, c2 = row + 2 * dr, col + 2 * dc

            if not (0 <= r1 < 9 and 0 <= c1 < 9):
                continue

            target = self.board[r1, c1]
            
            if moved_piece in [Pawn.WHITE, Pawn.KING] and target == Pawn.BLACK:  # White captures black
                if (0 <= r2 < 9 and 0 <= c2 < 9) and (self.board[r2, c2] in [Pawn.WHITE, Pawn.KING, Pawn.THRONE] or \
                        ((r2, c2) in citadels and (r2,c2) not in excluded_citadels)):
                    if real:
                        print(self.board_string())
                        print(f"Elimino in posizione {r1},{c1}")
                    self.board[r1, c1] = Pawn.EMPTY if (r1, c1) != throne else Pawn.THRONE
                    self.moves_without_capture = 0

            elif moved_piece == Pawn.BLACK and target == Pawn.WHITE:  # Black captures white only
                if (0 <= r2 < 9 and 0 <= c2 < 9) and (self.board[r2, c2] in [Pawn.BLACK, Pawn.THRONE] or \
                (r2, c2) in citadels or (r2,c2)==throne):
                    if real==True:
                        print(self.board_string())
                        print(f"Elimino in posizione {r1},{c1}")
                    self.board[r1, c1] = Pawn.EMPTY if (r1, c1) != throne else Pawn.THRONE
                    self.moves_without_capture = 0

            elif moved_piece == Pawn.BLACK and target == Pawn.KING:
                king_row, king_col = r1, c1
                # Controlli specifici per direzione
                if dr == 0 and dc == -1:  # Sinistra
                    self._check_capture_king_left(king_row, king_col, real)
                elif dr == 0 and dc == 1:  # Destra
                    self._check_capture_king_right(king_row, king_col, real)
                elif dr == 1 and dc == 0:  # Sotto
                    self._check_capture_king_down(king_row, king_col, real)
                elif dr == -1 and dc == 0:  # Sopra
                    self._check_capture_king_up(king_row, king_col, real)

        self.moves_without_capture += 1

    def _check_capture_king_left(self, king_row, king_col, real):
        citadels = {(0, 3), (0, 4), (0, 5), (1, 4), (3, 0), (4, 0), (5, 0), (4, 1),
                    (3, 8), (4, 8), (5, 8), (4, 7), (8, 3), (8, 4), (8, 5), (7, 4)}
        if king_col > 0:
            if (king_row, king_col) == (4, 4):  # Trono
                if (self.board[3, 4] == Pawn.BLACK and self.board[4, 3] == Pawn.BLACK and 
                    self.board[5, 4] == Pawn.BLACK):
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (3, 4):  # e4
                if self.board[2, 4] == Pawn.BLACK and self.board[3, 3] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (4, 5):  # f5
                if self.board[5, 5] == Pawn.BLACK and self.board[3, 5] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (5, 4):  # e6
                if self.board[6, 4] == Pawn.BLACK and self.board[5, 3] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            else:  # Posizione normale
                if (king_col > 1 and (self.board[king_row, king_col - 1] == Pawn.BLACK or 
                    (king_row, king_col - 1) in citadels)):
                    self._capture_king(king_row, king_col, real)

    def _check_capture_king_right(self, king_row, king_col, real):
        citadels = {(0, 3), (0, 4), (0, 5), (1, 4), (3, 0), (4, 0), (5, 0), (4, 1),
                    (3, 8), (4, 8), (5, 8), (4, 7), (8, 3), (8, 4), (8, 5), (7, 4)}
        if king_col < 8:
            if (king_row, king_col) == (4, 4):  # Trono
                if (self.board[3, 4] == Pawn.BLACK and self.board[4, 5] == Pawn.BLACK and 
                    self.board[5, 4] == Pawn.BLACK):
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (3, 4):  # e4
                if self.board[2, 4] == Pawn.BLACK and self.board[3, 5] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (5, 4):  # e6
                if self.board[5, 5] == Pawn.BLACK and self.board[6, 4] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (4, 3):  # d5
                if self.board[3, 3] == Pawn.BLACK and self.board[5, 3] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            else:  # Posizione normale
                if (king_col < 7 and (self.board[king_row, king_col + 1] == Pawn.BLACK or 
                    (king_row, king_col + 1) in citadels)):
                    self._capture_king(king_row, king_col, real)

    def _check_capture_king_down(self, king_row, king_col, real):
        citadels = {(0, 3), (0, 4), (0, 5), (1, 4), (3, 0), (4, 0), (5, 0), (4, 1),
                    (3, 8), (4, 8), (5, 8), (4, 7), (8, 3), (8, 4), (8, 5), (7, 4)}
        if king_row < 8:
            if (king_row, king_col) == (4, 4):  # Trono
                if (self.board[5, 4] == Pawn.BLACK and self.board[4, 5] == Pawn.BLACK and 
                    self.board[4, 3] == Pawn.BLACK):
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (3, 4):  # e4
                if self.board[3, 3] == Pawn.BLACK and self.board[3, 5] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (4, 3):  # d5
                if self.board[4, 2] == Pawn.BLACK and self.board[5, 3] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (4, 5):  # f5
                if self.board[4, 6] == Pawn.BLACK and self.board[5, 5] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            else:  # Posizione normale
                if (king_row < 7 and (self.board[king_row + 1, king_col] == Pawn.BLACK or 
                    (king_row + 1, king_col) in citadels)):
                    self._capture_king(king_row, king_col, real)

    def _check_capture_king_up(self, king_row, king_col, real):
        citadels = {(0, 3), (0, 4), (0, 5), (1, 4), (3, 0), (4, 0), (5, 0), (4, 1),
                    (3, 8), (4, 8), (5, 8), (4, 7), (8, 3), (8, 4), (8, 5), (7, 4)}
        if king_row > 0:
            if (king_row, king_col) == (4, 4):  # Trono
                if (self.board[3, 4] == Pawn.BLACK and self.board[4, 5] == Pawn.BLACK and 
                    self.board[4, 3] == Pawn.BLACK):
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (5, 4):  # e6
                if self.board[5, 3] == Pawn.BLACK and self.board[5, 5] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (4, 3):  # d5
                if self.board[4, 2] == Pawn.BLACK and self.board[3, 3] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            elif (king_row, king_col) == (4, 5):  # f5
                if self.board[4, 6] == Pawn.BLACK and self.board[3, 5] == Pawn.BLACK:
                    self._capture_king(king_row, king_col, real)
            else:  # Posizione normale
                if (king_row > 1 and (self.board[king_row - 1, king_col] == Pawn.BLACK or 
                    (king_row - 1, king_col) in citadels)):
                    self._capture_king(king_row, king_col, real)

    def _capture_king(self, king_row, king_col, real):
        if real:
            print(self.board_string())
            print(f"Re catturato in posizione {king_row},{king_col}")
        self.board[king_row, king_col] = Pawn.EMPTY
        self.turn = Turn.BLACKWIN
        self.moves_without_capture = 0
    
    def is_game_over(self):
        # Check for draw first (assuming is_draw() is efficient)
        if self.is_draw():
            return Turn.DRAW
        
        # Pre-compute set of victory positions for O(1) lookup
        victory_positions = {(1,0), (2,0), (0,1), (0,2), (0,6), (0,7), (6,0), (7,0),
                            (1,8), (2,8), (6,8), (7,8), (8,1), (8,2), (8,6), (8,7)}
        
        # Find king position
        king_pos = np.where(self.board == Pawn.KING)
        
        # Check if king has been captured
        if len(king_pos[0]) == 0:
            return Turn.BLACKWIN
        
        # Get king coordinates
        king_row, king_col = king_pos[0][0], king_pos[1][0]
        
        # Check victory conditions for white
        if (king_row, king_col) in victory_positions:
            return Turn.WHITEWIN
        
        # Check if no black pieces remain
        if np.sum(self.board == Pawn.BLACK) == 0:
            return Turn.WHITEWIN
        
        # Game continues
        return None
    
def parse_state(json_data):
    # Decodifica la stringa JSON in un dizionario Python
    state_dict = json.loads(json_data)
    # Estrai le informazioni dalla struttura
    board_data = state_dict["board"]
    turn = Turn[state_dict["turn"]]
    print(turn)
    # Converti il board in un formato che può essere utilizzato dalla classe State
    board = np.array([[Pawn[cell] for cell in row] for row in board_data])
    # Crea e ritorna l'oggetto State
    return State(board=board, turn=turn)


def is_captured(old_board, new_board, row, col, player_turn):
    # Perform initial check to avoid unnecessary work
    if old_board[row, col] == Pawn.EMPTY or new_board[row, col] != Pawn.EMPTY:
        return False
    
    # Determine player and opponent pieces
    pawn_player = Pawn.WHITE if player_turn == Turn.WHITE else Pawn.BLACK
    opponent = Pawn.BLACK if player_turn == Turn.WHITE else Pawn.WHITE
    
    # Check if the piece at (row, col) was an opponent's piece
    if old_board[row, col] != opponent:
        return False
    
    # Convert lists to sets for O(1) lookups
    citadels = {(0, 3), (0, 4), (0, 5), (1, 4), (3, 0), (4, 0), (5, 0), (4, 1),
               (3, 8), (4, 8), (5, 8), (4, 7), (8, 3), (8, 4), (8, 5), (7, 4)}
    excluded_citadels = {(4, 8), (0, 4), (8, 4), (4, 0)}
    throne = (4, 4)
    
    # Check vertical and horizontal capture pairs
    for (dr1, dc1), (dr2, dc2) in [((-1, 0), (1, 0)), ((0, -1), (0, 1))]:
        r1, c1 = row + dr1, col + dc1  # First side
        r2, c2 = row + dr2, col + dc2  # Opposite side
        
        # Check if both sides are hostile (causing capture)
        if (is_hostile_side(r1, c1, pawn_player, new_board, citadels, excluded_citadels, throne) and
            is_hostile_side(r2, c2, pawn_player, new_board, citadels, excluded_citadels, throne)):
            return True
    
    return False

def is_hostile_side(r, c, pawn_player, board, citadels, excluded_citadels, throne):
    
    piece = board[r, c]
    
    # Pawn player is the one who moved
    if piece == pawn_player:
        return True
    
    # Throne is hostile
    if piece == Pawn.THRONE:
        return True
    
    # King is hostile for black player since i am moving white
    if pawn_player == Pawn.WHITE and piece == Pawn.KING:
        return True
    
    # King on throne is hostile for white player
    if pawn_player == Pawn.BLACK and piece == Pawn.KING and (r, c) == throne:
        return True
    
    # Citadel logic
    if (r, c) in citadels:
        if pawn_player == Pawn.BLACK:
            return True
        elif pawn_player == Pawn.WHITE and (r, c) not in excluded_citadels:
            return True
    
    return False

def less_pawns(old_board, new_board, player_turn):
    # Verifica cattura tramite conteggio delle pedine
    old_count = 0
    new_count = 0

    target_pawn = Pawn.WHITE if player_turn == Turn.BLACK else Pawn.BLACK

    old_count=count_pawns(old_board,target_pawn)
    new_count=count_pawns(new_board,target_pawn)

    if new_count < old_count:
        return True
    else:
        return False

def count_captures(old_board, new_board, player_turn, print_captures=False):
    # Find all positions where pieces existed in old_board but are empty in new_board
    capture_mask = (old_board != Pawn.EMPTY) & (new_board == Pawn.EMPTY)
    
    # Get coordinates of potential captures
    capture_positions = np.where(capture_mask)
    count = 0
    
    # Check each potential capture
    for i in range(len(capture_positions[0])):
        r, c = capture_positions[0][i], capture_positions[1][i]
        if is_captured(old_board, new_board, r, c, player_turn):
            count += 1
            if print_captures:
                print(f"Capture detected! Player: {player_turn}, Position: ({r}, {c})")
    
    return count

# Networking functions
def connect_to_server(host='localhost', port=9999):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(30)
    try:
        client_socket.connect((host, port))
        print("Connected to server!")
        return client_socket
    except Exception as e:
        print(f"Connection error: {e}")
        return None

def receive_data(client_socket):
    try:
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
            print("Connection closed by server.")
            return None
        return json.loads(data)
    except socket.timeout:
        print("Receive timeout.")
        return None
    except Exception as e:
        print(f"Receive error: {e}")
        return None

def send_move(client_socket, move_json):
    try:
        json_data = json.dumps(move_json)
        client_socket.sendall(json_data.encode('utf-8'))
        print("Move sent to server:", json_data)
    except Exception as e:
        print(f"Send error: {e}")

def move_to_json(move):
    if move is None:
        return {"error": "No legal moves available"}
    from_row, from_col, to_row, to_col = move
    return {"from": {"row": from_row, "col": from_col}, "to": {"row": to_row, "col": to_col}}

'''def get_user_preferences():
    print("Choose mode:")
    print("1. Interactive console play")
    print("2. Connect to server")
    print("3. Train only")
    print("4. Play vs minmax")
    print("5. Train vs minmax")
    mode_choice = input("Enter 1, 2, 3, 4 or 5: ").strip()

    if mode_choice == "3":
        player_color = None  # No color needed for train-only mode
        train_first = True   # Impostiamo direttamente a True per "Train only"
    elif mode_choice == "5":
        player_color=None
        train_first= False
    else:
        print("Choose color to play as:")
        print("1. White (W)")
        print("2. Black (B)")
        color_choice = input("Enter 1 or 2: ").strip()
        player_color = Turn.WHITE if color_choice == "1" else Turn.BLACK

        train_first=False
        if mode_choice == "1" or mode_choice == "2":
            print("Train the network first?")
            print("1. Yes")
            print("2. No")
            train_choice = input("Enter 1 or 2: ").strip()
            train_first = train_choice == "1"

    return mode_choice, player_color, train_first'''
def parse_move_input(input_str):
    try:
        parts = input_str.strip('()').split(',')
        if len(parts) != 4:
            return None
        from_row, from_col, to_row, to_col = map(int, parts)
        # Le coordinate sono già 0-based come richiesto da State
        if not (0 <= from_row <= 8 and 0 <= from_col <= 8 and 0 <= to_row <= 8 and 0 <= to_col <= 8):
            return None
        return (from_row, from_col, to_row, to_col)
    except ValueError:
        return None

def move_to_str(move):
    if move is None:
        return "None"
    return f"({move[0]},{move[1]},{move[2]},{move[3]})"

def count_escape_routes(board, king_pos):
    x, y = king_pos
    escape_routes = 0
    
    # Pre-compute the almost-edge positions as a set for O(1) lookups
    almost_edge_positions = {
        (1,0), (2,0), (0,1), (0,2), (0,6), (0,7), (6,0), (7,0),
        (1,8), (2,8), (6,8), (7,8), (8,1), (8,2), (8,6), (8,7)
    }
    
    # Check each of the four directions
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x + dx, y + dy
        
        # Follow the path until we hit a piece or the edge
        while 0 <= nx < 9 and 0 <= ny < 9:
            if board[nx][ny] == Pawn.EMPTY:
                # Check if we've reached an almost-edge position
                if (nx, ny) in almost_edge_positions:
                    escape_routes += 1
                    break
                # Continue in the same direction
                nx += dx
                ny += dy
            else:
                # Path is blocked by a piece
                break
    
    return escape_routes

def evaluate_board(board):
    # Find king position efficiently
    king_indices = np.where(board == Pawn.KING)
    if king_indices[0].size == 0:
        return 1000  # King captured → attackers win
    
    # Extract position once
    x, y = king_indices[0][0], king_indices[1][0]
    
    # Check if king is at an almost-edge position (use a pre-computed set)
    almost_edge_positions = {(1,0), (2,0), (0,1), (0,2), (0,6), (0,7), (6,0), (7,0),
                          (1,8), (2,8), (6,8), (7,8), (8,1), (8,2), (8,6), (8,7)}
    if (x, y) in almost_edge_positions:
        return -1000
    
    # Count pieces directly using np.sum
    attackers = np.sum(board == Pawn.BLACK)
    defenders = np.sum(board == Pawn.WHITE)
    
    # Calculate king distance from edges more efficiently
    king_distance = min(x, y, 8-x, 8-y)
    
    # Calculate score components
    score = 0
    score -= king_distance * 10  # Closer to edges is better for defenders
    score += (attackers - defenders) * 2  # Numerical advantage
    
    # Count surrounded positions more efficiently
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    surrounded = sum(1 for dx, dy in directions 
                   if 0 <= x+dx < 9 and 0 <= y+dy < 9 and board[x+dx][y+dy] == Pawn.BLACK)
    score += surrounded * 20
    
    # Calculate escape routes
    escapes = count_escape_routes(board, (x, y))
    score -= escapes * 50
    
    return score

def minimax(state, depth, maximizing_player, is_attacker_turn):
    if depth == 0 or state.is_game_over():
        return evaluate_board(state.get_board()), None

    statecopy=state.clone()
    best_move = None
    if maximizing_player:
        max_eval = -math.inf
        for move in statecopy.get_legal_moves():
            statecopy.apply_move(move, real=False)
            eval, _ = minimax(statecopy, depth - 1, False, not is_attacker_turn)
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in statecopy.get_legal_moves():
            statecopy.apply_move(move,real=False)
            eval, _ = minimax(statecopy, depth - 1, True, not is_attacker_turn)
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move
    
def minimaxbeta(state, depth, alpha, beta, maximizing_player, is_attacker_turn, start_time, max_time):
    if depth == 0 or state.is_game_over():
        return evaluate_board(state.get_board()), None

    best_move = None
    for move in state.get_legal_moves():
        
        # Controlla se il tempo è scaduto
        if time.time() - start_time >= max_time:
            break

        statecopy = state.clone()
        statecopy.apply_move(move, real=False)

        eval,_ = minimaxbeta(
            statecopy,
            depth - 1,
            alpha,
            beta,
            not maximizing_player,
            not is_attacker_turn,
            start_time,
            max_time
        )

        if maximizing_player:
            if eval > alpha:
                alpha = eval
                best_move = move
            if beta <= alpha:
                break  # 🔪 Beta cut
        else:
            if eval < beta:
                beta = eval
                best_move = move
            if beta <= alpha:
                break  # 🔪 Alpha cut

    return (alpha, best_move) if maximizing_player else (beta, best_move)

# Funzione iterative deepening che utilizza minimaxbeta con potatura alpha-beta.
def iterative_deepening(state, max_time, maximizing_player, is_attacker_turn):
    """
    Esegue un'iterated deepening con minimax-beta fino a raggiungere il tempo massimo.
    
    Args:
        state: lo stato corrente del gioco.
        max_time: tempo massimo in secondi (es. 58).
        maximizing_player: se True, il nodo corrente è MAX.
        is_attacker_turn: indica se il giocatore corrente è attaccante.
    
    Returns:
        (best_move, best_eval, best_depth) dove:
          - best_move è la mossa migliore trovata,
          - best_eval è il punteggio associato,
          - best_depth è la massima profondità raggiunta.
    """
    start_time = time.time()
    best_move = None
    best_eval = -math.inf if maximizing_player else math.inf
    current_depth = 3

    # Continua la ricerca finché abbiamo tempo disponibile
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_time:
            break  # Interrompi se abbiamo superato il limite
        try:
            eval_current, move_current = minimaxbeta(
                state, current_depth, 
                -math.inf, math.inf, 
                maximizing_player, is_attacker_turn,
                start_time, max_time
            )
        except Exception as e:
            # In caso di errori (ad es. eccezioni dovute al tempo), interrompiamo la ricerca.
            print(f"Errore alla profondità {current_depth}: {e}")
            break

        # Aggiorna il miglior risultato se è migliore di quello precedente
        if (maximizing_player and eval_current > best_eval) or (not maximizing_player and eval_current < best_eval):
            best_eval = eval_current
            best_move = move_current
        # Se il valore è abbastanza alto (o basso per MIN) e non si prevede un cambiamento, potremmo interrompere qui
        # (questa è una possibile ottimizzazione aggiuntiva)
        current_depth += 1

    return best_move, best_eval, current_depth - 1


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def convert_move_for_server(move, color):
    # mossa è una tupla (riga da, colonna da, riga a, colonna a)
    from_row, from_col, to_row, to_col = move
    
    # Convertiamo le righe in notazione scacchistica
    from_square = f"{chr(97 + from_col)}{from_row + 1}"
    to_square = f"{chr(97 + to_col)}{to_row + 1}"
    
    # Creiamo il dizionario
    return {"from": from_square, "to": to_square, "turn": color}
    
# Main function
def main():
    player_json = json.dumps("MinMax & Relax")
    state = State()
    if(len(sys.argv)==4):
            color= sys.argv[1].lower()
            timeout=int(sys.argv[2])-2

            server_add=sys.argv[3]
    else:
        print("il numero di parametri deve essere 4")
        return(-1)
    
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if color == 'white':
            # Connect the socket to the port where the server is listening
            server_address = (server_add, 5800)
        elif color == 'black':
            # Connect the socket to the port where the server is listening
            server_address = (server_add, 5801)
        else:
            raise Exception("Se giochi o sei bianco oppure sei nero")
        color_move = color.upper()
        color = color[0].upper()
        sock.connect(server_address)
        sock.send(struct.pack('>i', len(player_json)))
        sock.send(player_json.encode())
        while True:
            len_bytes = struct.unpack('>i', recvall(sock, 4))[0]
            current_state_server_bytes = sock.recv(len_bytes)
            state = parse_state(current_state_server_bytes.decode())
            state.get_turn
            print(state.board_string())
            json_current_state_server = json.loads(current_state_server_bytes)
            print(state.get_turn())
            if(str(state.get_turn())=="BW"):
                if(str(color)=="B"):
                    print("Hai Vinto!!!!! Complimenti!!!!")
                if(str(color)=="W"):
                    print("Hai Perso!!!!!!!")
                return
            
            if(str(state.get_turn())=="WW"):
                if(str(color)=="B"):
                    print("Hai Perso!!!!!!!")
                if(str(color)=="W"):
                    print("Hai Vinto!!!!! Complimenti!!!!")
                return
            
            if(str(state.get_turn())=="D"):
                print("Game finito in pareggio")
                return
            
            print(color)
            if(str(state.get_turn())==str(color)):
                print('sono qui')
                is_attacker_turn=False
                maximizing_player=False
                alpha=math.inf
                beta=-math.inf
                max_time=timeout-2
                if(state.get_turn()==Turn.BLACK):
                    maximizing_player=True
                    is_attacker_turn=True
                    alpha=-math.inf
                    beta=math.inf
                move,_,_ = iterative_deepening(state, timeout, maximizing_player, is_attacker_turn)
                print(move)
                move_for_server = convert_move_for_server(move, color_move)
                print(move_for_server)
                move_for_server = json.dumps(move_for_server)
                print(move)
                sock.send(struct.pack('>i', len(move_for_server)))
                sock.send(move_for_server.encode())

if __name__ == "__main__":
    main()
