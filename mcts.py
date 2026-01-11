import math
import copy
import chess

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.n = 0
        self.w = 0

    @property
    def value(self):
        if self.n == 0: return 0
        return self.w / self.n

class MCTS:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def search(self, board, iterations=400):
        root = MCTSNode(board.copy())
        
        for _ in range(iterations):
            node = root
            
            # 1. Selection
            while node.children:
                # UCT formula
                best_score = -float('inf')
                best_child = None
                
                log_n = math.log(max(1, node.n))
                for child in node.children.values():
                    # Flip value because chess is zero-sum
                    # Q + C * sqrt(ln(N)/n)
                    score = (-child.value) + 1.4 * math.sqrt(log_n / (1 + child.n))
                    if score > best_score:
                        best_score = score
                        best_child = child
                node = best_child
            
            # 2. Expansion & Evaluation
            if not node.board.is_game_over():
                for move in node.board.legal_moves:
                    new_board = node.board.copy()
                    new_board.push(move)
                    node.children[move] = MCTSNode(new_board, parent=node, move=move)
                
                # Evaluate the position
                # Value is from the perspective of the side to move at 'node'
                v = self.evaluator.evaluate(node.board)
            else:
                # Terminal
                res = node.board.result()
                if res == "1-0":
                    v = 1.0 if node.board.turn == chess.WHITE else -1.0
                elif res == "0-1":
                    v = -1.0 if node.board.turn == chess.WHITE else 1.0
                else:
                    v = 0.0
            
            # 3. Backpropagation
            while node:
                node.n += 1
                node.w += v
                v = -v
                node = node.parent
                
        # Return best move by visit count
        if not root.children: return None
        return max(root.children.items(), key=lambda x: x[1].n)[0]