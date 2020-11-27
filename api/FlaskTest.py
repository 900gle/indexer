
from flask import Flask
# from flask_restplus import Resource, Api
#

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

app.run()