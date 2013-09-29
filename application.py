from flask import Flask, request, Response
from werkzeug import secure_filename
import os
import puzzle
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

@app.route('/solve', methods=['POST'])
def solve():
    piece_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(request.files['piece'].filename))
    board_filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(request.files['board'].filename))
    request.files['piece'].save(piece_filename)
    request.files['board'].save(board_filename)
    results = puzzle.solve_puzzle(piece_filename, board_filename)
    return Response(json.dumps([{'x':int(x),'y':int(y)} for x,y in results]), content_type='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(debug=True)
