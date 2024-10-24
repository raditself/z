
let board = null;
let game = new Chess();
let currentPlayer = 'w';

function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false;
    if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
        (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
        return false;
    }
}

function onDrop(source, target) {
    let move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    if (move === null) return 'snapback';

    updateStatus();
    
    // Make AI move
    makeAIMove();
}

function onSnapEnd() {
    board.position(game.fen());
}

function updateStatus() {
    let status = '';
    let moveColor = 'White';
    if (game.turn() === 'b') {
        moveColor = 'Black';
    }

    if (game.in_checkmate()) {
        status = 'Game over, ' + moveColor + ' is in checkmate.';
    } else if (game.in_draw()) {
        status = 'Game over, drawn position';
    } else {
        status = moveColor + ' to move';
        if (game.in_check()) {
            status += ', ' + moveColor + ' is in check';
        }
    }

    document.getElementById('game-status').innerText = status;
    document.getElementById('current-player').innerText = moveColor;
}

function makeAIMove() {
    fetch('/make_move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ move: [game.history({ verbose: true }).pop()] }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
            return;
        }
        
        game.load(data.board);
        board.position(game.fen());
        updateStatus();
        
        if (data.game_over) {
            alert('Game Over! Winner: ' + (data.winner === 1 ? 'White' : 'Black'));
        }
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

let config = {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd
};

board = Chessboard('chessboard', config);

updateStatus();

document.getElementById('reset-game').addEventListener('click', function() {
    fetch('/reset_game', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                game = new Chess();
                board.start();
                updateStatus();
            }
        })
        .catch((error) => {
            console.error('Error:', error);
        });
});

// Initial game state
fetch('/get_game_state')
    .then(response => response.json())
    .then(data => {
        game.load(data.board);
        board.position(game.fen());
        updateStatus();
    })
    .catch((error) => {
        console.error('Error:', error);
    });
