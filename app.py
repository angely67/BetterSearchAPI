from flask import Flask
from txtai.embeddings import Embeddings

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/test/<name>', methods=['GET'])
def testing(name=None):
    return "This is testing. Name="+name

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)