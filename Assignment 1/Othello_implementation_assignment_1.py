import numpy as np
import random
import time




def possible_moves(board, player):
    opponent = "B" if player == -1 else "W"
    playerC = "B" if player == 1 else "W"
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    possible_moves = set()
    
    for r in range(8):
        for c in range(8):
            if board[r][c] == playerC:
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    found_opponent = False
                    
                    while 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == opponent:
                        found_opponent = True
                        nr += dr
                        nc += dc
                    
                    if found_opponent and 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == " ":
                        possible_moves.add((nr, nc))
    
    return possible_moves

def no_moves(board):
    return len(possible_moves(board, 1)) == 0 and len(possible_moves(board, -1)) == 0

def has_won(board, player):
    player_count = np.count_nonzero(board == ("B" if player == 1 else "W"))
    opponent_count = np.count_nonzero(board == ("W" if player == 1 else "B"))

    if np.count_nonzero(board == ' ') == 0 or no_moves(board):
        if player_count > opponent_count:
            return 1
        elif player_count < opponent_count:
            return -1
        else:
            return 0 #draw
    return 2  # Game is not yet finished

def update_board(board, move, player):
    opponent = "B" if player == -1 else "W"
    playerC = "B" if player == 1 else "W"
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if move != None : 
        r, c = move
        board[r][c] = playerC 
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            flipped_positions = []

            while 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == opponent:
                flipped_positions.append((nr, nc))
                nr += dr
                nc += dc

            if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == playerC:
                for fr, fc in flipped_positions:
                    board[fr][fc] = playerC
    else:
        pass 
    return board

class Othello:
    def __init__(self):
        self.board = np.full((8, 8), ' ', dtype=str)
        self.board[3][3] = "W"
        self.board[3][4] = "B"
        self.board[4][3] = "B"
        self.board[4][4] = "W"
        self.winner = 0
        
    def get_board(self):
        return self.board.copy()
    
    def print_board(self):
        for row in self.board:
            print(row)
    
    def get_possible_moves(self, player):
        return possible_moves(self.board, player)
    
    def game_over(self):
        return self.winner != 0

    def make_move(self, move, player):
        if self.winner != 0:
            raise Exception('This game is already over!')

        self.board = update_board(self.board, move, player)

        if ' ' not in self.board or no_moves(self.board):
            blackP = np.count_nonzero(self.board == "B")
            whiteP = np.count_nonzero(self.board == "W")

            if blackP > whiteP:
                self.winner = 1
            elif blackP == whiteP:
                self.winner = 2
            else:
                self.winner = -1


class HumanPlayer:
    def __init__(self, player):
        self.player = player

    def move(self, env):
        board = env.get_board()
        posbl_moves = possible_moves(board, self.player)
        if len(posbl_moves) == 0:
            print("No moves available. Skip.")
        else:
            for row, col in posbl_moves:
                board[row][col] = "X"
            print("\nBoard Display:")
            for row in board:
                print(row)
            print("Possible moves:", posbl_moves)
            valid = False
            while not valid:
                ui = input("Introduce your move[row col]:")
                move = tuple(map(int, ui.split()))
                valid = move in posbl_moves
                if not valid:
                    print("Invalid move.\n")
            env.make_move(move, self.player)
            print("\nBoard after your move:")

class RandomChoice:
    def __init__(self, player):
        self.player = player

    def move(self, env):
        board = env.get_board()
        posbl_moves = possible_moves(board, self.player)
        if len(posbl_moves) == 0:
            print("No moves available. Skip.")
        else:
            move = random.choice(list(posbl_moves))
            env.make_move(move, self.player)
            print("\nBoard after random move:")
            



class EvalAlphaBeta:
    def __init__(self, player, max_time = 1.0):
        self.player = player
        self.max_time = max_time

    def move(self, env):
        board = env.get_board()
        best_move = None
        best_score = -np.inf
        alpha = -np.inf
        beta = np.inf
        
        moves = list(possible_moves(board, self.player))
        if not moves:
            print("No moves available, I skip.")
            return
        else:
            # Move Ordering: Prioritize stronger moves first to improve the efficiency of alpha beta pruning
            moves.sort(key=lambda move: self.move_heuristic(move), reverse=True)
            t = time.time_ns()
            for move in moves:
                board_copy = board.copy()
                score = self.min_player(update_board(board_copy, move, self.player), alpha, beta, t)
                if score > best_score:
                    best_score = score
                    best_move = move
                    alpha = max(alpha, score)

            print(f"\nMy move: {best_move}")
            env.make_move(best_move, self.player)

    def min_player(self, board, alpha, beta, t):
        # Minimizing opponent's move
        cur = time.time_ns()
        if ((cur-t) > self.max_time - (0.25*1000000000)):
            return self.evaluate(board)
        
        if has_won(board, self.player) != 2:
            return self.evaluate(board)
        
        
        score = np.inf
        moves = list(possible_moves(board, -self.player))
        moves.sort(key=lambda move: self.move_heuristic(move))  # Move ordering to improve alpha beta pruning efficiency
        if not moves:
            board_copy = board.copy()
            score = min(score, self.max_player(update_board(board_copy, None, -self.player), alpha, beta, t))
        for move in moves:
            board_copy = board.copy()
            score = min(score, self.max_player(update_board(board_copy, move, -self.player), alpha, beta, t))
            beta = min(beta, score)
            if beta <= alpha:
                break  # Alpha-beta pruning

        return score

    def max_player(self, board, alpha, beta, t):
        # Maximizing AI's move 
        
        cur = time.time_ns()
        if ((cur-t) > self.max_time - (0.25*1000000000)):
            return self.evaluate(board)
        
        if has_won(board, self.player) != 2 :
            return self.evaluate(board)

        
        
        score = -np.inf
        moves = list(possible_moves(board, self.player))
        moves.sort(key=lambda move: self.move_heuristic(move), reverse=True)  # Move ordering to improve alpha beta pruning efficiency
        if not moves:
            board_copy = board.copy()
            score = max(score, self.min_player(update_board(board_copy, None, self.player), alpha, beta, t))
        for move in moves:
            board_copy = board.copy()
            score = max(score, self.min_player(update_board(board_copy, move, self.player), alpha, beta, t))
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Alpha-beta pruning

        return score

    def evaluate(self, board):
        # Weighted board evaluation (corners are good, edges are neutral, interior is bad, spaces next to the corners are very bad(you allow your opponent to get a corner))
        weights = np.array([
            [100, -20, 10,  5,  5, 10, -20, 100],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [ 10,  -2,  3,  3,  3,  3,  -2,  10],
            [  5,  -2,  3,  3,  3,  3,  -2,   5],
            [  5,  -2,  3,  3,  3,  3,  -2,   5],
            [ 10,  -2,  3,  3,  3,  3,  -2,  10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [100, -20, 10,  5,  5, 10, -20, 100]
        ])

        black_score = np.sum(weights * (board == "B"))
        white_score = np.sum(weights * (board == "W"))

        return black_score - white_score if self.player == 1 else white_score - black_score

    def move_heuristic(self, move):
        #Assign a heuristic value to a move based on board position. 
        row, col = move
        move_values = np.array([
            [100, -20, 10,  5,  5, 10, -20, 100],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [ 10,  -2,  3,  3,  3,  3,  -2,  10],
            [  5,  -2,  3,  3,  3,  3,  -2,   5],
            [  5,  -2,  3,  3,  3,  3,  -2,   5],
            [ 10,  -2,  3,  3,  3,  3,  -2,  10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [100, -20, 10,  5,  5, 10, -20, 100]
        ])
        return move_values[row, col]


def main():
    hPlayer = input("Black[B] or White[W]: ")
    iTime = float(input("Introduce maximum response time (seconds):")) * (1000000000)
    aiMoveCont = 0
    start_time = time.time()
    if hPlayer == "B":
        player1 = HumanPlayer(1)
        #player2 = HumanPlayer(-1) #uncheck this if you want human vs human 
        player2 = EvalAlphaBeta(-1, max_time = iTime)
        #player1 = EvalAlphaBeta(1, max_time = iTime + 5000000000) #uncheck this if you want ai vs ai
        #player1 = RandomChoice(1) # uncheck this if you want random moves
    else:
        player1 = EvalAlphaBeta(1, max_time = iTime)
        player2 = HumanPlayer(-1)
        #player2 = RandomChoice(-1) # uncheck this if you want random moves 
    
    env = Othello()
    
    env.print_board()
    while True:
        player1.move(env)
        env.print_board()
        if hPlayer != "B": aiMoveCont += 1
        if env.game_over(): break

        player2.move(env)
        env.print_board()
        if hPlayer == "B": aiMoveCont += 1
        if env.game_over(): break
        

    wPlayer = " "
    print('Result:')
    end_time = time.time()
    if(env.winner == 1):
        print('Black wins!\n')
        wPlayer == "B"
    elif(env.winner == -1):
        print('White wins!\n')
        wPlayer == "W"
    else:
        print('Draw')
    print('Black score: ' + str(np.count_nonzero(env.board == "B")))
    print('White score: ' + str(np.count_nonzero(env.board == "W")))
    if(hPlayer == wPlayer):
        print('You win. Congratulations!')
    elif(wPlayer != " "):
        print('Computer wins. Try again!')
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")    
    avg_move_t = execution_time/aiMoveCont
    print(f"Average time per move: {avg_move_t:.6f} seconds" )
        

    

if __name__ == '__main__': main()
