import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers, models
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
                if (self.board[king_row, king_col - 1] == Pawn.BLACK or 
                    (king_row, king_col - 1) in citadels):
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
                if (self.board[king_row, king_col + 1] == Pawn.BLACK or 
                    (king_row, king_col + 1) in citadels):
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
                if (self.board[king_row + 1, king_col] == Pawn.BLACK or 
                    (king_row + 1, king_col) in citadels):
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
                if (self.board[king_row - 1, king_col] == Pawn.BLACK or 
                    (king_row - 1, king_col) in citadels):
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

# Neural network with training capability
class AlphaZeroNetwork:
    def __init__(self, weights_path='pesi.weights.h5'):
        self.weights_path = weights_path
        self.model = self._build_model()
        if os.path.exists(weights_path):
            self.load_weights()

    def _build_model(self):
        inputs = layers.Input(shape=(9, 9, 6))  # 5 pawn types + 1 for turn
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        for _ in range(4):
            shortcut = x
            x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(64, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([shortcut, x])
            x = layers.Activation('relu')(x)
        policy = layers.Conv2D(2, (1, 1), activation='relu')(x)
        policy = layers.Flatten()(policy)
        policy = layers.Dense(81 * 81, activation='softmax', name='policy')(policy)
        value = layers.Conv2D(1, (1, 1), activation='relu')(x)
        value = layers.Flatten()(value)
        value = layers.Dense(256, activation='relu')(value)
        value = layers.Dense(1, activation='tanh', name='value')(value)
        model = models.Model(inputs=inputs, outputs=[policy, value])
        model.compile(optimizer='adam', loss={'policy': 'categorical_crossentropy', 'value': 'mse'})
        return model

    def predict(self, state):
        board = self._encode_state(state)
        policy, value = self.model.predict(board[np.newaxis, ...], verbose=0)
        legal_moves = state.get_legal_moves()
        # Mappa le mosse legali agli indici di 81*81 (es. (r1, c1, r2, c2) -> r1*81*9 + c1*81 + r2*9 + c2)
        move_to_index = {m: m[0] * 81 * 9 + m[1] * 81 + m[2] * 9 + m[3] for m in legal_moves}
        legal_policy = np.zeros(len(legal_moves))
        for i, move in enumerate(legal_moves):
            legal_policy[i] = policy[0][move_to_index[move]]
        return legal_policy / legal_policy.sum(), value[0][0]  # Normalizza
    
    def predict_batch(self, states):
        # Codifica gli stati come in predict
        boards = np.array([self._encode_state(state) for state in states])
        # Predizione batch
        raw_policies, values = self.model.predict(boards, verbose=0)
        
        # Lista per le policy normalizzate
        normalized_policies = []
        for i, state in enumerate(states):
            # Ottieni le mosse legali per lo stato corrente
            legal_moves = state.get_legal_moves()
            # Mappa le mosse legali agli indici
            move_to_index = {m: m[0] * 81 * 9 + m[1] * 81 + m[2] * 9 + m[3] for m in legal_moves}
            # Filtra la policy grezza per le mosse legali
            legal_policy = np.zeros(len(legal_moves))
            for j, move in enumerate(legal_moves):
                idx = move_to_index[move]
                if idx < len(raw_policies[i]):
                    legal_policy[j] = raw_policies[i][idx]
            # Normalizza la policy
            policy_sum = legal_policy.sum()
            if policy_sum > 0:
                legal_policy /= policy_sum
            else:
                legal_policy = np.ones(len(legal_moves)) / len(legal_moves)  # Uniforme se somma = 0
            normalized_policies.append(legal_policy)
        
        return normalized_policies, values

    def train(self, experiences):
        states, policies, values = [], [], []
        for state, policy, value in experiences:
            encoded_state = self._encode_state(state)
            legal_moves = state.get_legal_moves()
            full_policy = np.zeros(81 * 81)
            move_to_index = {m: m[0] * 81 * 9 + m[1] * 81 + m[2] * 9 + m[3] for m in legal_moves}
            for idx, prob in enumerate(policy):
                full_policy[move_to_index[legal_moves[idx]]] = prob
            states.append(encoded_state)
            policies.append(full_policy)
            values.append(value)
        states = np.array(states)
        policies = np.array(policies)
        values = np.array(values)
        self.model.fit(states, {'policy': policies, 'value': values}, epochs=1, batch_size=32, verbose=0)

    def _encode_state(self, state):
        board = state.get_board()
        turn = state.get_turn()
        encoding = np.zeros((9, 9, 6))
        for r in range(9):
            for c in range(9):
                pawn = board[r, c]
                if pawn == Pawn.EMPTY:
                    encoding[r, c, 0] = 1
                elif pawn == Pawn.WHITE:
                    encoding[r, c, 1] = 1
                elif pawn == Pawn.BLACK:
                    encoding[r, c, 2] = 1
                elif pawn == Pawn.THRONE:
                    encoding[r, c, 3] = 1
                elif pawn == Pawn.KING:
                    encoding[r, c, 4] = 1
                encoding[r, c, 5] = 1 if turn == Turn.WHITE else 0
        return encoding

    
    def save_weights(self):
        self.model.save_weights(self.weights_path)
        print(f"Weights saved to {self.weights_path}")

    def load_weights(self):
        self.model.load_weights(self.weights_path)
        print(f"Weights loaded from {self.weights_path}")

# Monte Carlo Tree Search
class MCTSNode:
    def __init__(self, state, parent=None, move=None, prior=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_moves = state.get_legal_moves()
        self.prior = prior  

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def select_child(self):
        c = 1.414  # Costante di esplorazione, regolabile
        return max(self.children.items(),
                key=lambda item: item[1].value / (item[1].visits + 1e-6) +
                c * item[1].prior * (self.visits ** 0.5) / (1 + item[1].visits))[1]

'''class MCTS:
    def __init__(self, network, simulations=800):
        self.network = network
        self.simulations = simulations

    def search(self, initial_state):
        root = MCTSNode(initial_state.clone())
        legal_moves = initial_state.get_legal_moves()
        move_to_index = {move: i for i, move in enumerate(legal_moves)}
        policy, _ = self.network.predict(initial_state)
        for _ in range(self.simulations):
            node = self._select(root)
            if not node.state.is_game_over():
                node = self._expand(node)
            value = self._simulate(node)
            self._backpropagate(node, value)
        best_move = max(root.children.items(), key=lambda item: item[1].visits)[0]
        if best_move not in legal_moves:
            best_move = random.choice(legal_moves) if legal_moves else None
        return best_move
    
    def search(self, initial_state):
        root = MCTSNode(initial_state.clone())
        for _ in range(self.simulations):
            node = self._select(root)
            if not node.state.is_game_over():
                node = self._expand(node)
            value = self._simulate(node)
            self._backpropagate(node, value)
        return root  # Restituisci il nodo radice invece della mossa

    def _select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.select_child()
            if node.state.is_game_over():
                break
        return node

    def _expand(self, node):
        if not node.untried_moves:
            return node
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        new_state = node.state.clone()
        new_state.apply_move(move)
        child = MCTSNode(new_state, parent=node, move=move)
        node.children[move] = child
        return child

    def _simulate(self, node):
        state = node.state.clone()
        captures_white = 0
        captures_black = 0
        initial_turn = state.get_turn()  # Salva il turno iniziale per il calcolo finale
        
        while True:
            result = state.is_game_over()
            if result:
                # Calcola captures e losses dal punto di vista del giocatore iniziale
                if initial_turn == "WHITE":
                    captures = captures_white
                    losses = captures_black
                else:
                    captures = captures_black
                    losses = captures_white
                return calculate_reward(result, initial_turn, captures, losses)
            
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                return 0.0  # Ricompensa neutra
            
            move = random.choice(legal_moves)
            old_board = state.get_board().copy()
            current_turn = state.get_turn()
            state.apply_move(move)
            new_board = state.get_board()
            
            # Conta le catture del giocatore corrente
            new_captures = count_captures(old_board, new_board, current_turn, print_captures=False)
            if current_turn == "WHITE":
                captures_white += new_captures
            else:
                captures_black += new_captures
        

    def _backpropagate(self, node, value):
        while node:
            node.visits += 1
            # If there's no parent (root node), just add the value directly
            if node.parent is None:
                node.value += value
            else:
                # Adjust value based on turn perspective
                node.value += value if node.state.get_turn() == node.parent.state.get_turn() else -value
            node = node.parent'''
class MCTS:
    def __init__(self, network, simulations=50,batch_size=32):
        """
        Inizializza l'algoritmo Monte Carlo Tree Search (MCTS).
        
        Args:
            network: La rete neurale che fornisce predizioni di policy e valore.
            simulations: Numero di simulazioni da eseguire per ogni ricerca (default: 50).
        """
        self.network = network
        self.simulations = simulations
        self.prediction_cache = {}  # Cache per memorizzare le predizioni della rete
        self.batch_size=batch_size
        self.recent_states = []  # Lista degli stati recenti
        self.max_recent_states = 10  # Limite per evitare troppa memoria

    def search1(self, initial_state):
        """Esegue le simulazioni MCTS e restituisce il nodo radice."""
        root = MCTSNode(initial_state.clone())
        for _ in range(self.simulations):
            node = self._select(root)
            if node.state.is_game_over():
                value = self._evaluate_game_over(node.state)
            else:
                state_key = node.state.board.tobytes()  # Chiave univoca per lo stato
                if state_key not in self.prediction_cache:
                    node_policy, value = self.network.predict(node.state)
                    self.prediction_cache[state_key] = (node_policy, value)
                else:
                    node_policy, value = self.prediction_cache[state_key]
                if not node.is_fully_expanded():
                    node = self._expand(node, node_policy)
            self._backpropagate(node, value)
        return root

    def search(self, initial_state):
        self.prediction_cache = {}
        root = MCTSNode(initial_state.clone())
        legal_moves = initial_state.get_legal_moves()
        policy, root_value = self.network.predict(initial_state)
        # Aggiungi Dirichlet noise alla policy della radice
        if len(legal_moves) > 0:
            alpha = [0.3 / len(legal_moves)] * len(legal_moves)   # Parametro per il rumore
            noise = np.random.dirichlet(alpha)
            epsilon = 0.25  # Peso del rumore
            policy = (1 - epsilon) * policy + epsilon * noise

            '''# Penalizza mosse che riportano a stati recenti
            for i, move in enumerate(legal_moves):
                next_state = initial_state.clone()
                next_state.apply_move(move, real=False)
                state_key = next_state.board.tobytes()
                if state_key in self.recent_states:
                    policy[i] *= 0.001  # Penalità forte per stati ripetuti'''

         # Esegui simulazioni in batch
        for _ in range(self.simulations // self.batch_size):
            batch = []
            for _ in range(self.batch_size):
                node = self._select(root)

                # Se il nodo non ha figli, espandilo immediatamente
                if not node.is_fully_expanded():
                    policy, _ = self.network.predict(node.state)
                    node = self._expand(node, policy)

                if node.state.is_game_over():
                    value = self._evaluate_game_over(node.state)
                    self._backpropagate(node, value)
                else:
                    batch.append(node)
            
            if batch:
                # Raccogli gli stati del batch
                states = [node.state for node in batch]
                state_keys = [state.board.tobytes() for state in states]
                
                # Filtra gli stati non ancora in cache
                to_predict = [state for state, key in zip(states, state_keys) 
                            if key not in self.prediction_cache]
                if to_predict:
                    policies, values = self.network.predict_batch(to_predict)
                    for state, policy, value in zip(to_predict, policies, values):
                        self.prediction_cache[state.board.tobytes()] = (policy, value)
                
                # Processa ogni nodo nel batch
                for node in batch:
                    state_key = node.state.board.tobytes()
                    node_policy, value = self.prediction_cache[state_key]
                    if not node.is_fully_expanded():
                        node = self._expand(node, node_policy)
                    self._backpropagate(node, value)
        
        # Gestisci eventuali simulazioni rimanenti (se simulations non è multiplo di batch_size)
        remaining = self.simulations % self.batch_size
        if remaining:
            batch = []
            for _ in range(remaining):
                node = self._select(root)

                if not node.is_fully_expanded():
                    policy, _ = self.network.predict(node.state)
                    node = self._expand(node, policy)

                if node.state.is_game_over():
                    value = self._evaluate_game_over(node.state)
                    self._backpropagate(node, value)
                else:
                    batch.append(node)
            
            if batch:
                states = [node.state for node in batch]
                state_keys = [state.board.tobytes() for state in states]
                to_predict = [state for state, key in zip(states, state_keys) 
                            if key not in self.prediction_cache]
                if to_predict:
                    policies, values = self.network.predict_batch(to_predict)
                    for state, policy, value in zip(to_predict, policies, values):
                        self.prediction_cache[state.board.tobytes()] = (policy, value)
                
                for node in batch:
                    state_key = node.state.board.tobytes()
                    node_policy, value = self.prediction_cache[state_key]
                    if not node.is_fully_expanded():
                        node = self._expand(node, node_policy)
                    self._backpropagate(node, value)

            # Aggiorna la lista degli stati recenti
        '''current_state_key = initial_state.board.tobytes()
        if len(self.recent_states) >= self.max_recent_states:
            self.recent_states.pop(0)
        if current_state_key not in self.recent_states:  # Aggiungi solo se non già presente
            self.recent_states.append(current_state_key)'''

        return root

    def _select(self, node):
        """
        Seleziona il nodo da esplorare seguendo la strategia UCB.
        
        Args:
            node: Il nodo corrente da cui partire.
        
        Returns:
            Il nodo selezionato per l'espansione o la valutazione.
        """
        while node.is_fully_expanded() and node.children:
            node = node.select_child()
            if node.state.is_game_over():
                break
        return node

    def _expand(self, node, node_policy):
        current_legal_moves = node.state.get_legal_moves()
        current_move_to_index = {m: i for i, m in enumerate(current_legal_moves)}
        valid_untried_moves = [m for m in node.untried_moves if m in current_legal_moves]
        
        if not valid_untried_moves:
            return node
        
        # Seleziona la mossa con la policy più alta
        best_move = max(valid_untried_moves, key=lambda m: node_policy[current_move_to_index[m]])
        node.untried_moves.remove(best_move)
        new_state = node.state.clone()
        new_state.apply_move(best_move)
        
        # Assegna il prior al nuovo nodo figlio
        move_index = current_move_to_index[best_move]
        prior = node_policy[move_index]
        child = MCTSNode(new_state, parent=node, move=best_move, prior=prior)
        node.children[best_move] = child
        return child

    def _evaluate_game_over(self, state):
        """
        Valuta uno stato finale del gioco (vittoria, sconfitta o pareggio).
        
        Args:
            state: Lo stato del gioco da valutare.
        
        Returns:
            Il valore dello stato: 1.0 per vittoria, -1.0 per sconfitta, 0.0 per pareggio.
        """
        result = state.is_game_over()
        if result == Turn.WHITEWIN:
            return 1.0 if state.get_turn() == Turn.WHITE else -1.0
        elif result == Turn.BLACKWIN:
            return 1.0 if state.get_turn() == Turn.BLACK else -1.0
        elif result == Turn.DRAW:
            return 0.0
        return 0.0

    def _backpropagate(self, node, value):
        """
        Propaga il valore calcolato indietro attraverso l'albero.
        
        Args:
            node: Il nodo da cui iniziare la retropropagazione.
            value: Il valore da propagare.
        """
        while node:
            node.visits += 1
            if node.parent is None:
                node.value += value
            else:
                node.value += value if node.state.get_turn() == node.parent.state.get_turn() else -value
            node = node.parent

def choose_move(root, move_number):
    """
    Sceglie una mossa basata sul numero di visite dei figli della radice.
    
    Args:
        root: Il nodo radice dell'albero MCTS.
        move_number: Il numero della mossa corrente nel gioco.
    
    Returns:
        La mossa scelta.
    """
    if move_number < 10:  # Prime 15 mosse con esplorazione
        tau = 1.0
    else:  # Successivamente più deterministico
        tau = 0.1
    
    # Ottiene le mosse e le visite
    moves = list(root.children.keys())  # Lista delle mosse (es. tuple)
    visits = np.array([child.visits for child in root.children.values()])
    
    # Calcola le probabilità
    probs = visits ** (1 / tau)
    probs /= probs.sum()  # Normalizza
    
    # Sceglie un indice in base alle probabilità
    chosen_index = np.random.choice(len(moves), p=probs)
    
    # Restituisce la mossa corrispondente
    return moves[chosen_index]

def calculate_reward(result, player_color, captures=0, losses=0, capture_reward=0.00, loss_penalty=-0.00):
    reward = 0.0
    if result == Turn.WHITEWIN:
        reward = 1.0 if player_color == Turn.WHITE else -1.0
    elif result == Turn.BLACKWIN:
        reward = 1.0 if player_color == Turn.BLACK else -1.0
    elif result == Turn.DRAW:
        reward = -0.5
    reward += captures * capture_reward + losses * loss_penalty
    return reward

def count_pawns(board, pawn):
    #conta i pawns
    count = 0

    for r in range(9):
        for c in range(9):
            if board[r, c] == pawn or (pawn==Pawn.WHITE and board[r,c]==Pawn.KING):
                count += 1
            
    return count

def self_play(network, mcts, episodes=1):
    experiences = []
    for episode in range(episodes):
        print(f"Starting episode {episode + 1}/{episodes}")
        state = State()  # Assumo che State sia la tua classe per lo stato del gioco
        game_experiences = []
        captures_white = 0  # Catture totali del bianco (pedine nere rimosse)
        captures_black = 0  # Catture totali del nero (pedine bianche rimosse)
        move_number=0

        while True:
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                print("No legal moves available, ending game.")
                break
            
            # Esegui MCTS e scegli una mossa
            root = mcts.search(state)
            move = choose_move(root,move_number)
            print(f"Self-play - Chosen move: {move}")
            
            # Salva lo stato prima della mossa
            old_state = state.clone()
            old_board = old_state.get_board()
            
            # Applica la mossa
            state.apply_move(move, real=True)
            new_board = state.get_board()
            move_number+=1
            # Conta le catture del giocatore corrente
            current_turn = old_state.get_turn()
            new_captures = count_captures(old_board, new_board, current_turn, print_captures=False)
            if current_turn == Turn.WHITE:
                captures_white += new_captures
            else:
                captures_black += new_captures
            
            # Stampa informazioni sulle catture, se presenti
            if new_captures > 0:
                print(f"Self-play - State BEFORE capture (Turn: {current_turn}):")
                print(old_state.board_string())
                print(f"Self-play - State AFTER capture (Turn: {state.get_turn()}):")
                print(state.board_string())
                print(f"Captures - White: {captures_white}, Black: {captures_black}")
            
            # Calcola la policy basata sulle visite di MCTS
            policy = np.zeros(len(legal_moves))
            total_visits = sum(child.visits for child in root.children.values())
            move_to_index = {m: i for i, m in enumerate(legal_moves)}
            for m, child in root.children.items():
                if m in move_to_index:
                    policy[move_to_index[m]] = child.visits / total_visits if total_visits > 0 else 1.0 / len(legal_moves)
            
            # Salva l'esperienza (stato, policy, valore da calcolare dopo)
            game_experiences.append((old_state.clone(), policy, None))
            
            # Controlla se il gioco è finito
            result = state.is_game_over()
            if result:
                print(f"Game over: {result}")
                print(f"Self-play - Final state:")
                print(state.board_string())
                
                # Assegna le ricompense agli stati salvati
                for exp_state, exp_policy, _ in game_experiences:
                    turn = exp_state.get_turn()
                    if turn == Turn.WHITE:
                        captures = captures_white      # Catture fatte dal bianco
                        losses = captures_black        # Perdite del bianco (catture del nero)
                    else:
                        captures = captures_black      # Catture fatte dal nero
                        losses = captures_white        # Perdite del nero (catture del bianco)
                    exp_value = calculate_reward(result, turn, captures, losses)
                    experiences.append((exp_state, exp_policy, exp_value))
                break
        
        print(f"Episode {episode + 1} completed. Captures - White: {captures_white}, Black: {captures_black}")
    
    network.train(experiences)
    network.save_weights()
    print("Training completed, weights saved.")
    return experiences
    # Scarica il file
    #files.download('alphazero_weights.h5')
    #print(f"File save_path creato e pronto per il download.")


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

# JSON parsing and formatting
def parse_state(data):
    try:
        board_str = data['board'].replace('\n', '')
        if len(board_str) != 81:
            raise ValueError("Board string must be 81 characters.")
        board = np.array([Pawn.from_string(c) for c in board_str]).reshape(9, 9)
        turn = Turn(data['turn'])
        return State(board, turn)
    except Exception as e:
        print(f"Error parsing state: {e}")
        return None

def move_to_json(move):
    if move is None:
        return {"error": "No legal moves available"}
    from_row, from_col, to_row, to_col = move
    return {"from": {"row": from_row, "col": from_col}, "to": {"row": to_row, "col": to_col}}

def get_user_preferences():
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

    return mode_choice, player_color, train_first
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

def evaluate_board(board,initial_depth,depth):
    # Find king position efficiently
    king_indices = np.where(board == Pawn.KING)
    if king_indices[0].size == 0:
        return 1000-(initial_depth-depth)  # King captured → attackers win
    
    # Extract position once
    x, y = king_indices[0][0], king_indices[1][0]
    
    # Check if king is at an almost-edge position (use a pre-computed set)
    almost_edge_positions = {(1,0), (2,0), (0,1), (0,2), (0,6), (0,7), (6,0), (7,0),
                          (1,8), (2,8), (6,8), (7,8), (8,1), (8,2), (8,6), (8,7)}
    if (x, y) in almost_edge_positions:
        return -1000+(initial_depth-depth)  # King in almost-edge position → defenders win
    
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
    
def minimaxbeta(state, initial_depth,depth, alpha, beta, maximizing_player, is_attacker_turn, start_time, max_time):
    if depth == 0 or state.is_game_over():
        return evaluate_board(state.get_board(),initial_depth,depth), None

    best_move = None
    for move in state.get_legal_moves():
        
        # Controlla se il tempo è scaduto
        if time.time() - start_time >= max_time:
            break

        statecopy = state.clone()
        statecopy.apply_move(move, real=False)

        eval,_ = minimaxbeta(
            statecopy,
            initial_depth,
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
                state,current_depth, current_depth, 
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
# Main function
def main():
    mode_choice, player_color, train_first = get_user_preferences()
    network = AlphaZeroNetwork()
    mcts = MCTS(network, simulations=32,batch_size=32)

    # Train the network first if requested
    if train_first:
        print("Training network via self-play...")
        print("Pesi iniziali:", network.model.get_weights()[0][0][0][:5])  # Stampa i primi pesi
        self_play(network, mcts, episodes=100)
        print("Pesi dopo self-play:", network.model.get_weights()[0][0][0][:5])
        

    # Mode 3: Train only and exit
    if mode_choice == "3":
        print("Training completed. Exiting.")
        return

    
    # Modalità 1: Gioco interattivo da console
    if mode_choice == "1":
        while True:
            print(f"Stai giocando come {player_color} in modalità console interattiva.")
            state = State()  # Stato iniziale del gioco
            game_data = []  # Lista per salvare i dati della partita (stato, policy, turno)
            captures_white = 0  # Contatore delle catture del Bianco
            captures_black = 0  # Contatore delle catture del Nero
            move_number=0
            while True:
                # Stampa la scacchiera corrente
                print(state.board_string())
                result = state.is_game_over()
                if result:
                    print(f"Partita terminata: {result}")
                    break

                legal_moves = state.get_legal_moves()
                if not legal_moves:
                    print("Nessuna mossa legale disponibile, partita terminata.")
                    result = state.get_result()  # Ottieni il risultato finale
                    break

                # Esegui MCTS per ottenere la policy (probabilità delle mosse)
                root = mcts.search(state)
                policy = np.zeros(len(legal_moves))
                total_visits = sum(child.visits for child in root.children.values())
                move_to_index = {m: i for i, m in enumerate(legal_moves)}
                for m, child in root.children.items():
                    if m in move_to_index:
                        policy[move_to_index[m]] = child.visits / total_visits if total_visits > 0 else 1.0 / len(legal_moves)
                game_data.append((state.clone(), policy, state.get_turn()))

                if state.get_turn() == player_color:
                    # Turno dell'utente
                    while True:
                        move_str = input("Inserisci la tua mossa (es. (6,8,1,8)): ").strip()
                        move = parse_move_input(move_str)  # Converte la stringa in una mossa
                        if move and move in legal_moves:
                            break
                        print("Mossa non valida. Riprova.")
                    old_board = state.get_board().copy()
                    state.apply_move(move,real=True)
                    move_number+=1
                    new_captures = count_captures(old_board, state.get_board(), player_color, print_captures=True)
                    if player_color == Turn.WHITE:
                        captures_white += new_captures
                    else:
                        captures_black += new_captures
                else:
                    # Turno dell'agente
                    best_move = choose_move(root,move_number)
                    old_board = state.get_board().copy()
                    agent_turn=state.get_turn()
                    state.apply_move(best_move,real=True)
                    move_number+=1
                    new_captures = count_captures(old_board, state.get_board(), agent_turn, print_captures=True)
                    if agent_turn == Turn.WHITE:
                        captures_white += new_captures
                    else:
                        captures_black += new_captures
                    print(f"L'agente ha mosso: {move_to_str(best_move)}")
                
                print(f"Scacchiera aggiornata (Turno: {state.get_turn()}):")
                print(state.board_string())
                print(f"Catture - Bianco: {captures_white}, Nero: {captures_black}")     

            # Addestra la rete con i dati della partita
            if result is not None:
                training_data = []
                for s, p, pt in game_data:
                    if pt == Turn.WHITE:
                        captures = captures_white
                        losses = captures_black
                    else:
                        captures = captures_black
                        losses = captures_white
                    value = calculate_reward(result, pt, captures, losses)
                    training_data.append((s, p, value))
                network.train(training_data)
                network.save_weights()
                print("Rete addestrata con i dati della partita.")
            c=input("Vuoi continuare a giocare? S o N")
            if(c=='N' or c=='n' or c==None):
                break

    # Mode 2: Connect to server (mostly unchanged)
    elif mode_choice == "2":
        client_socket = connect_to_server()
        if not client_socket:
            return
        while True:
            data = receive_data(client_socket)
            if data is None:
                break
            state = parse_state(data)
            if state is None:
                continue
            print(f"Received state:\n{state.board_string()}\nTurn: {state.get_turn()}")
            if state.get_turn() != player_color:
                print(f"Not my turn (playing as {player_color}). Waiting for next state.")
                continue
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                print("No legal moves available.")
                move_json = move_to_json(None)
            else:
                best_move = mcts.search(state)
                if best_move not in legal_moves:
                    print("Invalid move selected, choosing random legal move.")
                    best_move = random.choice(legal_moves)
                move_json = move_to_json(best_move)
            old_board = state.get_board().copy()
            state.apply_move(best_move)
            count_captures(old_board, state.get_board(), player_color, print_captures=True)
            send_move(client_socket, move_json)
            result = state.is_game_over()
            if result:
                print(f"Game over: {result}")
                break
        client_socket.close()

    elif mode_choice == "4":
        while True:
            print(f"Stai giocando come {player_color} in modalità console interattiva.")
            state = State()  # Stato iniziale del gioco
            game_data = []  # Lista per salvare i dati della partita (stato, policy, turno)
            captures_white = 0  # Contatore delle catture del Bianco
            captures_black = 0  # Contatore delle catture del Nero
            move_number=0
            while True:
                # Stampa la scacchiera corrente
                print(state.board_string())
                result = state.is_game_over()
                if result:
                    print(f"Partita terminata: {result}")
                    break

                legal_moves = state.get_legal_moves()
                if not legal_moves:
                    print("Nessuna mossa legale disponibile, partita terminata.")
                    result = state.get_result()  # Ottieni il risultato finale
                    break  

                if state.get_turn() == player_color:
                    # Turno dell'utente
                    while True:
                        move_str = input("Inserisci la tua mossa (es. (6,8,1,8)): ").strip()
                        move = parse_move_input(move_str)  # Converte la stringa in una mossa
                        if move and move in legal_moves:
                            break
                        print("Mossa non valida. Riprova.")
                    old_board = state.get_board().copy()
                    state.apply_move(move,real=True)
                    move_number+=1
                    new_captures = count_captures(old_board, state.get_board(), player_color, print_captures=True)
                    if player_color == Turn.WHITE:
                        captures_white += new_captures
                    else:
                        captures_black += new_captures
                else:
                    # Turno dell'agente
                    is_attacker_turn=False
                    maximizing_player=False
                    depth=4
                    alpha=math.inf
                    beta=-math.inf
                    if(state.get_turn()==Turn.BLACK):
                        maximizing_player=True
                        is_attacker_turn=True
                        alpha=-math.inf
                        beta=math.inf
                    #_,best_move = minimaxbeta(state,depth,alpha, beta, maximizing_player ,is_attacker_turn)
                    max_time=58
                    best_move,_, best_depth = iterative_deepening(state,max_time,maximizing_player,is_attacker_turn)
                    print(f"Massima profondità: {best_depth}")
                    old_board = state.get_board().copy()
                    agent_turn=state.get_turn()
                    state.apply_move(best_move,real=True)
                    move_number+=1
                    new_captures = count_captures(old_board, state.get_board(), agent_turn, print_captures=True)
                    if agent_turn == Turn.WHITE:
                        captures_white += new_captures
                    else:
                        captures_black += new_captures
                    print(f"L'agente ha mosso: {move_to_str(best_move)}")
                
                print(f"Scacchiera aggiornata (Turno: {state.get_turn()}):")
                print(state.board_string())
                print(f"Catture - Bianco: {captures_white}, Nero: {captures_black}")
    
    elif mode_choice=="5":
        episodes=5
        experiences = []
        for episode in range(episodes):
            print(f"Starting episode {episode + 1}/{episodes}")
            state = State()  # Assumo che State sia la tua classe per lo stato del gioco
            game_experiences = []
            captures_white = 0  # Catture totali del bianco (pedine nere rimosse)
            captures_black = 0  # Catture totali del nero (pedine bianche rimosse)
            move_number=0
            player_color=random.choice([Turn.WHITE,Turn.BLACK])
            while True:
                
                legal_moves = state.get_legal_moves()
                if not legal_moves:
                    print("No legal moves available, ending game.")
                    break
                
                old_state = state.clone()  # Salva lo stato prima della mossa
                old_board = old_state.get_board()

                if state.get_turn()==player_color:
                    # Esegui MCTS e scegli una mossa
                    root = mcts.search(state)
                    move = choose_move(root,move_number)
                    print(f"Self-play - AI Chosen move: {move}")
                      
                    # Applica la mossa
                    state.apply_move(move, real=True)
                    new_board = state.get_board()
                    move_number+=1
                    # Conta le catture del giocatore corrente
                    current_turn = old_state.get_turn()
                    new_captures = count_captures(old_board, new_board, current_turn, print_captures=False)
                    if current_turn == Turn.WHITE:
                        captures_white += new_captures
                    else:
                        captures_black += new_captures
                    
                    # Stampa informazioni sulle catture, se presenti
                    if new_captures > 0:
                        print(f"Self-play - State BEFORE capture (Turn: {current_turn}):")
                        print(old_state.board_string())
                        print(f"Self-play - State AFTER capture (Turn: {state.get_turn()}):")
                        print(state.board_string())
                        print(f"Captures - White: {captures_white}, Black: {captures_black}")
                    
                    # Calcola la policy basata sulle visite di MCTS
                    policy = np.zeros(len(legal_moves))
                    total_visits = sum(child.visits for child in root.children.values())
                    move_to_index = {m: i for i, m in enumerate(legal_moves)}
                    for m, child in root.children.items():
                        if m in move_to_index:
                            policy[move_to_index[m]] = child.visits / total_visits if total_visits > 0 else 1.0 / len(legal_moves)

                else:
                    is_attacker_turn=False
                    maximizing_player=False
                    depth=4
                    alpha=math.inf
                    beta=-math.inf
                    if(state.get_turn()==Turn.BLACK):
                        maximizing_player=True
                        is_attacker_turn=True
                        alpha=-math.inf
                        beta=math.inf
                    #_,best_move = minimaxbeta(state,depth,alpha, beta, maximizing_player ,is_attacker_turn)
                    max_time=7
                    best_move,_, best_depth = iterative_deepening(state,max_time,maximizing_player,is_attacker_turn)
                    print(f"Massima profondità: {best_depth}")
                    
                    # Esegui MCTS per calcolare la policy
                    root = mcts.search(old_state)  # Usa lo stato prima della mossa
                    policy = np.zeros(len(legal_moves))
                    total_visits = sum(child.visits for child in root.children.values())
                    move_to_index = {m: i for i, m in enumerate(legal_moves)}
                    for m, child in root.children.items():
                        if m in move_to_index:
                            policy[move_to_index[m]] = child.visits / total_visits if total_visits > 0 else 1.0 / len(legal_moves)

                    agent_turn=state.get_turn()
                    state.apply_move(best_move,real=True)
                    move_number+=1
                    new_captures = count_captures(old_board, state.get_board(), agent_turn, print_captures=True)
                    if agent_turn == Turn.WHITE:
                        captures_white += new_captures
                    else:
                        captures_black += new_captures
                    print(f"MinMax ha mosso: {move_to_str(best_move)}")
                
                print(state.board_string())
                # Salva l'esperienza (stato, policy, valore da calcolare dopo)
                game_experiences.append((old_state.clone(), policy, None))
                # Controlla se il gioco è finito
                result = state.is_game_over()
                if result:
                    print(f"Game over: {result}")
                    print(f"Self-play - Final state:")
                    print(state.board_string())
                    
                    # Assegna le ricompense agli stati salvati
                    for exp_state, exp_policy, _ in game_experiences:
                        turn = exp_state.get_turn()
                        if turn == Turn.WHITE:
                            captures = captures_white      # Catture fatte dal bianco
                            losses = captures_black        # Perdite del bianco (catture del nero)
                        else:
                            captures = captures_black      # Catture fatte dal nero
                            losses = captures_white        # Perdite del nero (catture del bianco)
                        exp_value = calculate_reward(result, turn, captures, losses)
                        experiences.append((exp_state, exp_policy, exp_value))
                    break
            
            print(f"Episode {episode + 1} completed. Captures - White: {captures_white}, Black: {captures_black}")
        
        network.train(experiences)
        network.save_weights()
        print("Training completed, weights saved.")
        return experiences
        # Scarica il file
        #files.download('alphazero_weights.h5')
        #print(f"File save_path creato e pronto per il download.")
if __name__ == "__main__":
    main()