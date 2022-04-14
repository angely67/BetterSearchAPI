from flask import Flask, request
from txtai.embeddings import Embeddings
from txtai.pipeline import Extractor
import json
embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})
extractor = Extractor(embeddings, "distilbert-base-cased-distilled-squad")

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/test/<name>', methods=['GET'])
def testing(name=None):
    return "This is testing. Name="+name


@app.route('/api/qa', methods=['POST'])
def question_answer():
    data = request.json["data"]

    question = request.json["question"]

    print("----", question, "----")
    answer = extractor([(question, question, question, False)], data)
    print(answer[0][1])
    embeddings.index([(uid, text, None) for uid, text in enumerate(data)])
    searched = embeddings.search(answer[0][1], 100)
    searched = [(data[r[0]], r[1]) for r in searched]
    print(searched)

    searched = embeddings.search(answer[0][0], 100)
    searched = [(data[r[0]], r[1]) for r in searched]

    print(searched)

    searched = embeddings.search(answer[0][0]+" "+answer[0][1], 100)
    searched = [(data[r[0]], r[1]) for r in searched]

    print(searched)

    return json.dumps(answer)


@app.route('/api/semantic', methods=['POST'])
def semantic():
    print(request.json)
    body = request.json
    data = body["data"]
    query = body["query"]
    '''data = ["US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day"]'''
    embeddings.index([(uid, text, None) for uid, text in enumerate(data)])
    print("%-20s %s" % ("Query", "Best Match"))
    print("-" * 50)

    searched = embeddings.search(query, 100)
    print(searched)

    results = []
    if len(searched) > 0:
        results = [data[r[0]] for r in searched[1:] if r[1] >= 0.1]
    results.insert(0, data[searched[0][0]])

    print(results)

    return json.dumps(results)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)