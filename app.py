from flask import Flask, request
from flask_cors import CORS, cross_origin
from txtai.embeddings import Embeddings
from txtai.pipeline import Extractor
import json
embeddings = Embeddings({"path": "sentence-transformers/nli-mpnet-base-v2"})
extractor = Extractor(embeddings, "distilbert-base-cased-distilled-squad")

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/api/qa', methods=['POST'])
@cross_origin()
def question_answer():
    data = request.json["data"]
    question = request.json["question"]
    
    try:
        answer = extractor([(question, question, question, False)], data)

        embeddings.index([(uid, text, None) for uid, text in enumerate(data)])

        searched = embeddings.search(answer[0][0]+" "+answer[0][1], 100)
        results = []
        if len(searched) > 0:
            results = [data[r[0]] for r in searched[1:] if r[1] >= max(searched[0][1] * 2 / 3, 0.15)]
        results.insert(0, data[searched[0][0]])
        return json.dumps({"answer": answer, "evidence": results})
    except:
        return "Internal Server Error", 500


@app.route('/api/semantic', methods=['POST'])
@cross_origin()
def semantic():
    body = request.json
    data = body["data"]
    query = body["query"]
    try:
        embeddings.index([(uid, text, None) for uid, text in enumerate(data)])
        searched = embeddings.search(query, 100)
        results = []
        if len(searched) > 0:
            results = [data[r[0]] for r in searched[1:] if r[1] >= max(searched[0][1] * 2 / 3, 0.15)]
        results.insert(0, data[searched[0][0]])
        return json.dumps(results)
    except:
        return "Internal Server Error", 500

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)