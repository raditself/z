
let board = null;
let game = new Chess();

function onDragStart(source, piece, position, orientation) {
    if (game.game_over()) return false;
    if (piece.search(/^b/) !== -1) return false;
}

function makeRandomMove() {
    let possibleMoves = game.moves();
    if (game.game_over() || possibleMoves.length === 0) return;

    let randomIdx = Math.floor(Math.random() * possibleMoves.length);
    game.move(possibleMoves[randomIdx]);
    board.position(game.fen());
}

function onDrop(source, target) {
    let move = game.move({
        from: source,
        to: target,
        promotion: 'q'
    });

    if (move === null) return 'snapback';

    fetch('/make_move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ move: move }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.game_over) {
            alert('Game Over! Winner: ' + data.winner);
        } else {
            board.position(data.board);
        }
    });
}

function onSnapEnd() {
    board.position(game.fen());
}

let config = {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd
};

board = Chessboard('chessboard', config);

document.getElementById('reset-game').addEventListener('click', function() {
    fetch('/reset_game', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                game = new Chess();
                board.start();
            }
        });
});
