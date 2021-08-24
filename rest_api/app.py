from flask import Flask
from flask_restful import Api, Resource, reqparse, request
import subprocess
import os

app = Flask(__name__)
api = Api(app)


@api.resource("/")
class Server(Resource):
    def post(self):
        png_file = request.files['file']
        print(png_file)
        filename = '/tmp/{}'.format(png_file.filename)
        png_file.save(filename)
        # os.system('sxiv {} &'.format(filename))
        filesize = subprocess.run(["du", "-hs", filename], stdout=subprocess.PIPE).stdout.decode("utf-8").split()[0]
        return {"filesize": filesize}

app.run(debug=True, host='0.0.0.0')
