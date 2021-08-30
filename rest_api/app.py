from flask import Flask
from flask_restful import Api, Resource, request
import subprocess
import os
import sys
import diff
import main
# sys.path.insert(0, '../python/diff.py')

app = Flask(__name__)
api = Api(app)


@api.resource("/position_<int:pos_num>")
class Position(Resource):
    def post(self, pos_num):
        png_file = request.files['file']
        filename = '/tmp/{}'.format(png_file.filename)
        png_file.save(filename)
        pos = main.run(filename)
        filesize = subprocess.run(["du", "-hs", filename], stdout=subprocess.PIPE).stdout.decode("utf-8").split()[0]
        return {"file size": filesize, "input": "png file of image of board", "output": "64 underscore-separated numbers for representation of board", "position": pos}


@api.resource("/diff_<string:prev_pos>_<string:cur_pos>")
class Diff(Resource):
    def get(self, prev_pos, cur_pos):
        pos1, pos2 = diff.diff(prev_pos, cur_pos)
        return {"position1": pos1, "position2": pos2}


@api.resource("/save")
class Save(Resource):
    def post(self):
        pgn_file = request.files['file']
        pgn_file.save("/tmp/" + pgn_file.filename)
        return "successfully saved, thank you for your cooperation"


app.run(debug=True, host='0.0.0.0')
