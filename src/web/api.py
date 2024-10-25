
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.alphazero.model import AlphaZeroNetwork
from src.alphazero.mcts import MCTS
from src.alphazero.game import GameState
import chess

app = FastAPI()

# Initialize the AI model
model = AlphaZeroNetwork()  # You might need to load weights here
mcts = MCTS(model)

class MoveRequest(BaseModel):
    fen: str
    move: str

class MoveResponse(BaseModel):
    ai_move: str
    game_over: bool
    result: str

@app.post("/make_move", response_model=MoveResponse)
async def make_move(move_request: MoveRequest):
    try:
        # Create a chess board from the FEN string
        board = chess.Board(move_request.fen)
        
        # Make the player's move
        board.push_san(move_request.move)
        
        if board.is_game_over():
            return MoveResponse(ai_move="", game_over=True, result=board.result())
        
        # Convert chess.Board to GameState
        game_state = GameState.from_board(board)
        
        # Use MCTS to get the best move
        for _ in range(800):  # Number of MCTS simulations
            mcts.search(game_state)
        
        action = mcts.get_best_action(game_state)
        
        # Convert action to chess move
        ai_move = chess.Move.from_uci(action)
        board.push(ai_move)
        
        return MoveResponse(
            ai_move=ai_move.uci(),
            game_over=board.is_game_over(),
            result=board.result() if board.is_game_over() else ""
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Chess AI API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
