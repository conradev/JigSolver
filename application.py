from flask import Flask, request
from werkzeug import secure_filename
import os
import puzzle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

@app.route('/solve', methods=['POST'])
def solve():
    piece_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(request.files['piece'].filename))
    board_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(request.files['board'].filename))
    request.files['piece'].save(piece_filename)
    request.files['board'].save(board_filename)
    return puzzle.compute(piece_filename, board_filename)

if __name__ == '__main__':
    app.run(debug=True)
