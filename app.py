import eventlet
eventlet.monkey_patch()
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import numpy as np
import time
from collections import defaultdict
import uuid




# --- Flask & SocketIO Setup ---
app = Flask(__name__)
# Allow CORS for development, especially for SocketIO
CORS(app, resources={r"/*": {"origins": "*"}}) 
app.config['SECRET_KEY'] = 'your_secret_key_here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")
# ============================================================
# GAME MANAGER (For Online Games)
# ============================================================
class GameManager:
    def __init__(self):
        self.games = {}  # {game_id: {board, players: {1: sid, 2: sid}, turn, size}}
        self.sid_to_game = {} # {sid: game_id}
        self.engines = {} # {game_id: ReversiEngine}

    def create_game(self, host_sid, board_size):
        game_id = str(uuid.uuid4())[:8]
        board = self._init_board(board_size)
        
        self.games[game_id] = {
            "id": game_id,
            "size": board_size,
            "board": board,
            "players": {1: host_sid, 2: None}, # Player 1 is Black
            "turn": 1,
            "status": "waiting"
        }
        self.sid_to_game[host_sid] = game_id
        self.engines[game_id] = ReversiEngine(board_size)
        return self.games[game_id]

    # In GameManager class in app.py

    def join_game(self, game_id, joiner_sid):
        game = self.games.get(game_id)
        
        # Check if game exists
        if not game:
            return None

        # IDEMPOTENCY FIX: If this user is ALREADY player 2, return success immediately
        if game["players"][2] == joiner_sid:
            return game

        # Standard check: must be waiting
        if game["status"] != "waiting":
            return None
        
        # Prevent host from joining their own game as player 2
        if game["players"][1] == joiner_sid:
            return None

        # Player 2 is White
        game["players"][2] = joiner_sid
        game["status"] = "playing"
        self.sid_to_game[joiner_sid] = game_id
        return game

    def get_game(self, game_id):
        return self.games.get(game_id)
    
    def remove_game(self, game_id):
        game = self.games.pop(game_id, None)
        if game:
            for sid in game["players"].values():
                if sid in self.sid_to_game:
                    del self.sid_to_game[sid]
            if game_id in self.engines:
                del self.engines[game_id]
        return game

    def get_valid_moves(self, game_id):
        engine = self.engines.get(game_id)
        game = self.games.get(game_id)
        if not engine or not game:
            return []
        return engine.get_valid_moves(game["board"], game["turn"])

    def make_move(self, game_id, row, col, player):
        game = self.games.get(game_id)
        engine = self.engines.get(game_id)

        if not game or not engine or game["turn"] != player:
            return False, "Invalid turn or game."
        
        # 1. Validate move
        valid_moves = engine.get_valid_moves(game["board"], player)
        if not any(m["row"] == row and m["col"] == col for m in valid_moves):
            return False, "Invalid move coordinates."
        
        # 2. Execute move
        new_board = engine.make_move(game["board"], row, col, player)
        game["board"] = new_board
        
        # 3. Determine next player
        next_player = 3 - player
        
        # Check if next player has moves
        next_moves = engine.get_valid_moves(new_board, next_player)
        
        if len(next_moves) > 0:
            game["turn"] = next_player
            return True, None
        
        # Next player has no moves, check if current player has moves (skip turn)
        current_moves = engine.get_valid_moves(new_board, player)
        
        if len(current_moves) > 0:
            # Skip opponent's turn, current player plays again
            game["turn"] = player
            return True, "Opponent skipped turn."
        else:
            # Game over, no one can move
            game["status"] = "finished"
            return True, "Game Over"

    def _init_board(self, size):
        board = [[0]*size for _ in range(size)]
        mid = size // 2
        board[mid-1][mid-1] = 2
        board[mid-1][mid] = 1
        board[mid][mid-1]  = 1
        board[mid][mid] = 2
        return board

# Instantiate the manager for online games
online_manager = GameManager()


# ============================================================
# HELPER: Safe JSON conversion for NumPy ints
# ============================================================

def to_py_int(x):
    if isinstance(x, (np.int32, np.int64, np.int16, np.int8)):
        return int(x)
    return x

def deep_convert(obj):
    if isinstance(obj, dict):
        return {k: deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_convert(x) for x in obj]
    return to_py_int(obj)


# ============================================================
# ZOBRIST HASHING (REMAINS UNCHANGED)
# ============================================================

class ZobristHash:
    def __init__(self, board_size=8):
        self.board_size = board_size
        np.random.seed(42)
        self.table = np.random.randint(
            0, 2**63, size=(board_size, board_size, 3), dtype=np.int64
        )

    def hash_board(self, board):
        h = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    h ^= int(self.table[i][j][board[i][j]])
        return h


# ============================================================
# TRANSPOSITION TABLE (REMAINS UNCHANGED)
# ============================================================

class TranspositionTable:
    def __init__(self):
        self.table = {}
        self.hits = 0
        self.misses = 0

    def store(self, h, depth, score, move, flag="exact"):
        self.table[h] = {
            "depth": int(depth),
            "score": int(score),
            "move": move,
            "flag": flag
        }

    def lookup(self, h, depth, alpha, beta):
        if h not in self.table:
            self.misses += 1
            return None

        entry = self.table[h]
        self.hits += 1

        if entry["depth"] >= depth:
            if entry["flag"] == "exact":
                return entry
            if entry["flag"] == "lower" and entry["score"] >= beta:
                return entry
            if entry["flag"] == "upper" and entry["score"] <= alpha:
                return entry

        return None
# ============================================================
# REVERSI ENGINE (FOR AI/SOLVER/LOCAL GAMES)
# ============================================================

class ReversiEngine:
    # ... (All methods of ReversiEngine remain unchanged) ...
    DIRECTIONS = [
        (-1,-1), (-1,0), (-1,1),
        (0,-1), (0,1),
        (1,-1), (1,0), (1,1)
    ]

    def __init__(self, board_size=8):
        self.board_size = board_size
        self.zobrist = ZobristHash(board_size)
        self.tt = TranspositionTable()
        self.killer_moves = defaultdict(list)

        self.stats = {
            "nodes_explored": 0,
            "pruning_count": 0,
            "tt_hits": 0,
            "max_depth": 0
        }

        self._init_positional_weights()

    # --------------------------------------------------------
    # POSITIONAL WEIGHTS
    # --------------------------------------------------------
    def _init_positional_weights(self):
        s = self.board_size
        self.weights = np.ones((s, s), dtype=int)

        corners = [(0,0),(0,s-1),(s-1,0),(s-1,s-1)]
        for r, c in corners:
            self.weights[r][c] = 100

        if s >= 8:
            xs = [(1,1),(1,s-2),(s-2,1),(s-2,s-2)]
            for r, c in xs:
                self.weights[r][c] = -25

            cs = [
                (0,1),(1,0),(0,s-2),(1,s-1),
                (s-1,1),(s-2,0),(s-2,s-1),(s-1,s-2)
            ]
            for r, c in cs:
                self.weights[r][c] = -10

        for i in range(s):
            if self.weights[0][i] == 1: self.weights[0][i] = 10
            if self.weights[s-1][i] == 1: self.weights[s-1][i] = 10
            if self.weights[i][0] == 1: self.weights[i][0] = 10
            if self.weights[i][s-1] == 1: self.weights[i][s-1] = 10

    # --------------------------------------------------------
    # VALID MOVES
    # --------------------------------------------------------
    def get_valid_moves(self, board, player):
        moves = []
        opp = 3 - player

        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r][c] != 0:
                    continue

                flips_total = 0
                for dr, dc in self.DIRECTIONS:
                    flips_total += self._count_flips(board, r, c, dr, dc, player, opp)

                if flips_total > 0:
                    moves.append({
                        "row": int(r),
                        "col": int(c),
                        "flips": int(flips_total)
                    })

        return moves

    def _count_flips(self, board, r, c, dr, dc, player, opp):
        r += dr
        c += dc
        flips = 0

        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if board[r][c] == 0:
                return 0
            if board[r][c] == player:
                return flips
            flips += 1
            r += dr
            c += dc

        return 0

    # --------------------------------------------------------
    # APPLY MOVE
    # --------------------------------------------------------
    def make_move(self, board, row, col, player):
        newb = [r[:] for r in board]
        newb[row][col] = int(player)
        opp = 3 - player

        for dr, dc in self.DIRECTIONS:
            r, c = row + dr, col + dc
            flips = []

            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if newb[r][c] == 0:
                    break
                if newb[r][c] == player:
                    for fr, fc in flips:
                        newb[fr][fc] = int(player)
                    break
                flips.append((r, c))
                r += dr
                c += dc

        return newb

    # --------------------------------------------------------
    # EVALUATION FUNCTION
    # --------------------------------------------------------
    def evaluate(self, board, player):
        opp = 3 - player
        score = 0
        pd = od = ps = os = 0

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == player:
                    pd += 1
                    w = int(self.weights[i][j])
                    score += w
                    if w == 100:
                        ps += 1
                elif board[i][j] == opp:
                    od += 1
                    w = int(self.weights[i][j])
                    score -= w
                    if w == 100:
                        os += 1

        player_moves = len(self.get_valid_moves(board, player))
        opp_moves = len(self.get_valid_moves(board, opp))

        score += (player_moves - opp_moves) * 20
        score += (ps - os) * 80
        score += (pd - od) * 5

        return int(score)

    # --------------------------------------------------------
    # MOVE ORDERING
    # --------------------------------------------------------
    def order_moves(self, moves, depth):
        ordered = []
        for m in moves:
            r, c = m["row"], m["col"]
            score = 0

            if int(self.weights[r][c]) == 100:
                score += 10000

            if m in self.killer_moves[depth]:
                score += 5000

            if int(self.weights[r][c]) == 10:
                score += 1000

            score += m["flips"] * 10

            if int(self.weights[r][c]) < 0:
                score -= 1000

            ordered.append((score, m))

        ordered.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in ordered]

    # --------------------------------------------------------
    # MINIMAX + ALPHA-BETA
    # --------------------------------------------------------
    def minimax(self, board, depth, alpha, beta, maximizing, player, root_depth):
        self.stats["nodes_explored"] += 1

        h = self.zobrist.hash_board(board)
        tt = self.tt.lookup(h, depth, alpha, beta)
        if tt:
            self.stats["tt_hits"] += 1
            return tt["score"], tt["move"]

        if depth == 0:
            score = self.evaluate(board, player)
            self.tt.store(h, depth, score, None)
            return score, None

        current = player if maximizing else (3 - player)
        moves = self.get_valid_moves(board, current)

        if not moves:
            opp_moves = self.get_valid_moves(board, 3 - current)
            if not opp_moves:
                score = self.evaluate(board, player)
                return score, None
            # Pass move: opponent plays again
            return self.minimax(board, depth-1, alpha, beta, not maximizing, player, root_depth)

        moves = self.order_moves(moves, depth)

        if maximizing:
            best = float('-inf')
            best_move = moves[0]

            for m in moves:
                nb = self.make_move(board, m["row"], m["col"], current)
                val, _ = self.minimax(nb, depth-1, alpha, beta, False, player, root_depth)
                if val > best:
                    best = val
                    best_move = m
                alpha = max(alpha, best)
                if beta <= alpha:
                    self.killer_moves[depth].append(m)
                    self.stats["pruning_count"] += 1 # Pruning count update
                    break

            self.tt.store(h, depth, best, best_move, "lower" if best > beta else "exact")
            return best, best_move

        else:
            best = float('inf')
            best_move = moves[0]

            for m in moves:
                nb = self.make_move(board, m["row"], m["col"], current)
                val, _ = self.minimax(nb, depth-1, alpha, beta, True, player, root_depth)
                if val < best:
                    best = val
                    best_move = m
                beta = min(beta, best)
                if beta <= alpha:
                    self.killer_moves[depth].append(m)
                    self.stats["pruning_count"] += 1 # Pruning count update
                    break

            self.tt.store(h, depth, best, best_move, "upper" if best < alpha else "exact")
            return best, best_move

    # --------------------------------------------------------
    # ITERATIVE DEEPENING
    # --------------------------------------------------------
    def search(self, board, player, max_depth):
        start = time.time()
        best_move = None
        best_score = None

        self.stats = {
            "nodes_explored": 0,
            "pruning_count": 0,
            "tt_hits": 0,
            "max_depth": 0
        }
        self.tt.table.clear() # Clear TT for fresh search

        # Perform the actual search and update stats
        for depth in range(1, max_depth+1):
            current_score, current_move = self.minimax(
                board, depth,
                float('-inf'),
                float('inf'),
                True, player, depth
            )
            # Only use result if a move was found and the search completed for this depth
            if current_move is not None:
                best_move = current_move
                best_score = current_score
                self.stats["max_depth"] = depth
            
            # TODO: Implement time limit check here for early exit

        elapsed = int((time.time() - start)*1000)

        # The pruning rate should use the TT hits count from the engine's TT object
        tt_hits_count = self.tt.hits
        self.tt.hits = 0 # Reset TT hits for next search
        self.tt.misses = 0 # Reset TT misses

        return {
            "move": best_move,
            "score": int(best_score) if best_score is not None else 0,
            "stats": {
                "nodes_explored": int(self.stats["nodes_explored"]),
                "pruning_count": int(self.stats["pruning_count"]),
                "tt_hits": int(tt_hits_count),
                "depth_reached": int(self.stats["max_depth"]),
                "time_ms": int(elapsed),
                "pruning_rate": round(
                    self.stats["pruning_count"] / max(self.stats["nodes_explored"], 1) * 100,
                    2
                )
            }
        }

# ============================================================
# API ROUTES (REMAINS FOR LOCAL/AI/SOLVER)
# ============================================================

engines = {}  # one engine per browser session for local/ai/solver


# ------------------------------------------------------------
# INIT GAME
# ------------------------------------------------------------
@app.route("/api/init", methods=["POST"])
def api_init():
    data = request.json
    size = int(data.get("board_size", 8))
    session = data.get("session_id", "default")

    # Reuse or create a local engine
    engines[session] = ReversiEngine(size)

    board = online_manager._init_board(size) # Use manager's init board utility

    return jsonify({
        "success": True,
        "board": board
    })


# ------------------------------------------------------------
# GET VALID MOVES
# ------------------------------------------------------------
@app.route("/api/valid_moves", methods=["POST"])
def api_valid():
    data = request.json
    board = data["board"]
    player = int(data["player"])
    session = data.get("session_id", "default")

    if session not in engines:
        engines[session] = ReversiEngine(len(board))

    moves = engines[session].get_valid_moves(board, player)

    return jsonify({"success": True, "moves": moves})


# ------------------------------------------------------------
# APPLY MOVE
# ------------------------------------------------------------
@app.route("/api/make_move", methods=["POST"])
def api_make():
    data = request.json
    board = data["board"]
    row = int(data["row"])
    col = int(data["col"])
    player = int(data["player"])
    session = data.get("session_id", "default")

    if session not in engines:
        engines[session] = ReversiEngine(len(board))

    engine = engines[session]

    # Validate move
    valid_moves = engine.get_valid_moves(board, player)
    if not any(m["row"] == row and m["col"] == col for m in valid_moves):
        return jsonify({"success": False, "message": "Invalid move"})

    new_board = engine.make_move(board, row, col, player)

    return jsonify({"success": True, "board": new_board})


# ------------------------------------------------------------
# AI MOVE (Minimax + Alpha-Beta)
# ------------------------------------------------------------
@app.route("/api/ai_move", methods=["POST"])
def api_ai():
    data = request.json
    board = data["board"]
    player = int(data["player"])
    difficulty = data.get("difficulty", "medium")
    session = data.get("session_id", "default")

    depth_map = {
        "easy": 4,
        "medium": 6,
        "hard": 8,
        "expert": 10
    }
    depth = depth_map.get(difficulty, 6)

    if session not in engines:
        engines[session] = ReversiEngine(len(board))

    engine = engines[session]
    result = engine.search(board, player, depth)

    if not result["move"]:
        return jsonify({"success": False, "message": "No moves available"})

    move = result["move"]
    new_board = engine.make_move(board, move["row"], move["col"], player)

    return jsonify({
        "success": True,
        "move": move,
        "board": new_board,
        "stats": result["stats"]
    })


# ------------------------------------------------------------
# SOLVER MODE (Get Top 3 Best Moves)
# ------------------------------------------------------------
@app.route("/api/solver", methods=["POST"])
def api_solver():
    data = request.json
    board = data["board"]
    player = int(data["player"])
    session = data.get("session_id", "default")

    if session not in engines:
        engines[session] = ReversiEngine(len(board))

    engine = engines[session]

    moves = engine.get_valid_moves(board, player)
    result = []

    # Solver uses a fixed search depth of 5
    solver_depth = 5 
    
    # Clear engine stats before each solver run (for clean stats if needed)
    engine.stats = {
        "nodes_explored": 0, "pruning_count": 0, "tt_hits": 0, "max_depth": 0
    }
    engine.tt.table.clear()


    for m in moves:
        nb = engine.make_move(board, m["row"], m["col"], player)
        
        # Minimax on the resulting board state, assuming opponent plays optimally (False = minimizing)
        score, _ = engine.minimax(
            nb, solver_depth, float('-inf'), float('inf'),
            False, player, solver_depth
        )

        result.append({
            "move": {
                "row": int(m["row"]),
                "col": int(m["col"]),
                "flips": int(m["flips"])
            },
            "score": int(score),
            "evaluation": (
                "Winning" if score > 1000 else
                "Strong" if score > 500 else
                "Good" if score > 100 else
                "Equal" if score > -100 else
                "Weak" if score > -500 else
                "Losing"
            )
        })

    result.sort(key=lambda x: x["score"], reverse=True)

    return jsonify({
        "success": True,
        "top_moves": result[:3]
    })


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"})


# ============================================================
# SOCKETIO EVENTS (FOR ONLINE HVH)
# ============================================================

# Use request.sid to get the unique session ID (socket ID)
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    emit('session_ready', {'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')
    game_id = online_manager.sid_to_game.get(request.sid)
    if game_id:
        game = online_manager.get_game(game_id)
        if game and game["status"] != "finished":
            # Notify the other player that their opponent left
            opponent_sid = game["players"][1] if game["players"][2] == request.sid else game["players"][2]
            if opponent_sid:
                emit('opponent_left', {'game_id': game_id, 'message': 'Opponent disconnected. Game forfeited.'}, room=opponent_sid)
            
        online_manager.remove_game(game_id)
        leave_room(game_id)


@socketio.on('create_game')
def on_create_game(data):
    sid = request.sid
    board_size = data.get('board_size', 8)

    # Create new game
    new_game = online_manager.create_game(sid, board_size)

    # Put host in the room
    join_room(new_game["id"])

    # Send game creation confirmation to host ONLY
    emit('game_created', {
        'game_id': new_game["id"],
        'board_size': new_game["size"],
        'player_color': 1  # Black
    }, room=sid)

    # 🔥 IMPORTANT FIX: Broadcast updated open games list to ALL clients
    open_games = [
        {"id": g["id"], "size": g["size"], "host_sid": g["players"][1]}
        for g in online_manager.games.values()
        if g["status"] == "waiting"
    ]

    socketio.emit('open_games_list', {"games": open_games})

@socketio.on('get_open_games')
def on_get_open_games():
    open_games = [
        {"id": g["id"], "size": g["size"], "host_sid": g["players"][1]}
        for g in online_manager.games.values() if g["status"] == "waiting"
    ]
    print(f"📤 Sending {len(open_games)} open games to {request.sid}")
    print(f"   Games: {[g['id'] for g in open_games]}")
    emit('open_games_list', {"games": open_games}, room=request.sid)

# In app.py

@socketio.on('join_game')
def on_join_game(data):
    sid = request.sid
    game_id = str(data.get('game_id', '')).strip()
    
    print(f"🎮 Join request from {sid} for game '{game_id}'")
    
    game = online_manager.join_game(game_id, sid)
    
    if game:
        print(f"✅ Player {sid} joined game {game_id}")
        
        # 1. Join the SocketIO room
        join_room(game_id)
        
        # 2. Prepare payload
        payload = {
            'game_id': game["id"],
            'board_size': game["size"],
            'board': game["board"],
            'scores': {1: 2, 2: 2} # Initial scores
        }

        # 3. Notify the JOINER (Player 2 - White)
        # This triggers the frontend to switch screens
        payload['player_color'] = 2
        emit('game_joined', payload, room=sid)

        # 4. Notify the HOST (Player 1 - Black)
        # This triggers the host to stop seeing "Waiting for opponent..."
        payload['player_color'] = 1
        host_sid = game["players"][1]
        emit('game_joined', payload, room=host_sid)

        # 5. Broadcast to room that game is starting (ensures sync)
        emit('game_state_update', {
            'game_id': game_id,
            'status': 'playing',
            'turn': 1,
            'board': game['board'],
            'message': 'Game Started! Black to move.'
        }, room=game_id)

        # 6. Update the Open Games list for everyone else in the lobby
        # (Remove this game from the list since it's now playing)
        open_games = [
            {"id": g["id"], "size": g["size"], "host_sid": g["players"][1]}
            for g in online_manager.games.values()
            if g["status"] == "waiting"
        ]
        socketio.emit('open_games_list', {"games": open_games})

    else:
        available = [g for g in online_manager.games.keys() if online_manager.games[g]['status'] == 'waiting']
        print(f"❌ Game {game_id} not found or full.")
        emit('join_failed', {
            'message': f"Game '{game_id}' unavailable. Available: {available}"
        }, room=sid)

@socketio.on('make_online_move')
def on_make_online_move(data):
    sid = request.sid
    game_id = online_manager.sid_to_game.get(sid)
    
    if not game_id:
        emit('move_error', {'message': 'Not in a game.'}, room=sid)
        return

    game = online_manager.get_game(game_id)
    
    # Identify player number (1 or 2)
    player = 0
    if game["players"][1] == sid:
        player = 1
    elif game["players"][2] == sid:
        player = 2
        
    if player == 0 or player != game["turn"]:
        emit('move_error', {'message': 'It is not your turn or you are not a player.'}, room=sid)
        return

    row = data.get('row')
    col = data.get('col')
    
    success, message = online_manager.make_move(game_id, row, col, player)
    
    if success:
        # Check for game end and determine winner/score
        if game["status"] == "finished":
            flat = [cell for row in game["board"] for cell in row]
            scores = {
                1: flat.count(1),
                2: flat.count(2)
            }


            winner = 0
            if scores[1] > scores[2]:
                winner = 1
            elif scores[2] > scores[1]:
                winner = 2
                
            game_state = {
                'game_id': game_id,
                'board': game["board"],
                'turn': game["turn"],
                'status': game["status"],
                'scores': scores,
                'winner': winner,
                'message': message or "Game Over"
            }
            socketio.emit('game_state_update', game_state, room=game_id)
            online_manager.remove_game(game_id)
            leave_room(game_id)
            
        else:
            # Broadcast new state
            game_state = {
                'game_id': game_id,
                'board': game["board"],
                'turn': game["turn"],
                'status': game["status"],
                'message': message or f"Player {player} moved to ({row}, {col})."
            }
            socketio.emit('game_state_update', game_state, room=game_id)
    else:
        emit('move_error', {'message': message}, room=sid)


# ============================================================
# MAIN RUNNER (MODIFIED TO RUN WITH SOCKETIO)
# ============================================================
if __name__ == "__main__":
    print("⚡ Reversi AI Backend (REST + SocketIO) running at http://localhost:5000")
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)


