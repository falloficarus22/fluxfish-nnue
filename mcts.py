import math
import time
import chess

# MVV-LVA values for move ordering
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

class MCTSNode:
    def __init__(self, move=None, parent=None, prior=0.0):
        self.move = move
        self.parent = parent
        self.children = {}
        self.n = 0
        self.w = 0
        self.p = prior # Prior probability for PUCT
        self.is_expanded = False

    @property
    def value(self):
        if self.n == 0: return 0
        return self.w / self.n

class MCTS:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def get_move_priority(self, board, move):
        """Heuristic for Prior Probability (P)."""
        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            victim_val = PIECE_VALUES.get(victim.piece_type, 1) if victim else 1
            attacker = board.piece_at(move.from_square)
            attacker_val = PIECE_VALUES.get(attacker.piece_type, 1) if attacker else 1
            score = 10 + (victim_val * 10) - attacker_val
        elif board.gives_check(move):
            score = 5
        
        # Center control bonus
        to_sq = move.to_square
        if to_sq in [chess.D4, chess.D5, chess.E4, chess.E5]:
            score += 2
        
        return score

    def quiescence_search(self, board, alpha=-2.0, beta=2.0, depth=0):
        """Simple Quiescence search to avoid horizon effect."""
        stand_pat = self.evaluator(board)
        if depth >= 2: return stand_pat # Limit depth for performance
        
        if stand_pat >= beta: return beta
        if stand_pat > alpha: alpha = stand_pat
        
        # Only search captures in Q-search
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        # Sort by MVV-LVA
        captures.sort(key=lambda m: self.get_move_priority(board, m), reverse=True)
        
        for move in captures:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, depth + 1)
            board.pop()
            
            if score >= beta: return beta
            if score > alpha: alpha = score
            
        return alpha

    def search_ranked(self, board, iterations=1000, time_limit=None):
        """Search using PUCT (Polynomial Upper Confidence Trees)."""
        start_time = time.time()
        root = MCTSNode()
        
        # Initial expansion
        moves = list(board.legal_moves)
        priorities = [self.get_move_priority(board, m) for m in moves]
        total_p = sum(priorities) + len(moves)
        
        for i, move in enumerate(moves):
            # Softmax-like prior
            p = (priorities[i] + 1) / total_p
            root.children[move] = MCTSNode(move=move, parent=root, prior=p)
        root.is_expanded = True

        sim_board = board.copy()

        for i in range(iterations):
            if time_limit and (i & 31 == 0):
                if time.time() - start_time > time_limit:
                    break
            
            node = root
            path = []
            
            # 1. Selection (PUCT)
            while node.is_expanded and node.children:
                best_score = -float('inf')
                best_move = None
                
                # Selection constants
                cpuct = 1.5
                sqrt_n = math.sqrt(node.n) if node.n > 0 else 1.0
                
                for move, child in node.children.items():
                    # PUCT Formula: Q + C * P * sqrt(N) / (1 + n)
                    u = cpuct * child.p * sqrt_n / (1 + child.n)
                    score = (-child.value) + u
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                
<<<<<<< HEAD
                # Evaluate the position
                # Value is from the perspective of the side to move at 'node'
                v = self.evaluator(node.board)
=======
                if best_move is None: break
                node = node.children[best_move]
                sim_board.push(best_move)
                path.append(best_move)

            # 2. Expansion & Evaluation
            if not sim_board.is_game_over() and not node.is_expanded:
                # Use Quiescence Search for better evaluation
                v = self.quiescence_search(sim_board)
                
                # Expand
                moves = list(sim_board.legal_moves)
                priorities = [self.get_move_priority(sim_board, m) for m in moves]
                total_p = sum(priorities) + len(moves)
                
                for j, move in enumerate(moves):
                    p = (priorities[j] + 1) / total_p
                    node.children[move] = MCTSNode(move=move, parent=node, prior=p)
                node.is_expanded = True
>>>>>>> 0548ecf (First successful game played against the bot)
            else:
                if sim_board.is_game_over():
                    res = sim_board.result()
                    if res == "1-0": v = 1.0 if sim_board.turn == chess.WHITE else -1.0
                    elif res == "0-1": v = -1.0 if sim_board.turn == chess.WHITE else 1.0
                    else: v = 0.0
                else:
                    v = self.quiescence_search(sim_board)

            # 3. Backpropagation
            while node:
                node.n += 1
                node.w += v
                v = -v
                node = node.parent
<<<<<<< HEAD
                
        # Return best move by visit count
        if not root.children: return None
        return max(root.children.items(), key=lambda x: x[1].n)[0]
    
    def search_ranked(self, board, iterations=400):
        """Search and return all moves ranked by visit count."""
        root = MCTSNode(board.copy())
        
        for _ in range(iterations):
            node = root
            
            # Selection
            while node.children:
                best_score = -float('inf')
                best_child = None
                
                log_n = math.log(max(1, node.n))
                for child in node.children.values():
                    score = (-child.value) + 1.4 * math.sqrt(log_n / (1 + child.n))
                    if score > best_score:
                        best_score = score
                        best_child = child
                node = best_child
            
            # Expansion & Evaluation
            if not node.board.is_game_over():
                for move in node.board.legal_moves:
                    new_board = node.board.copy()
                    new_board.push(move)
                    node.children[move] = MCTSNode(new_board, parent=node, move=move)
                
                v = self.evaluator(node.board)
            else:
                res = node.board.result()
                if res == "1-0":
                    v = 1.0 if node.board.turn == chess.WHITE else -1.0
                elif res == "0-1":
                    v = -1.0 if node.board.turn == chess.WHITE else 1.0
                else:
                    v = 0.0
            
            # Backpropagation
            while node:
                node.n += 1
                node.w += v
                v = -v
                node = node.parent
        
        # Return all moves ranked by visit count
        if not root.children:
            return []
        
        ranked = sorted(root.children.items(), key=lambda x: x[1].n, reverse=True)
        return [(move, child.n) for move, child in ranked]
=======
                if path:
                    sim_board.pop()
                    path.pop()
        
        if not root.children: return []
        return sorted(root.children.items(), key=lambda x: x[1].n, reverse=True)

    def search(self, board, iterations=400):
        """Legacy search method returning best move."""
        ranked = self.search_ranked(board, iterations)
        return ranked[0][0] if ranked else None
>>>>>>> 0548ecf (First successful game played against the bot)
