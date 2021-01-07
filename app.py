import emotion_recognizer as tm
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def hello_world():
    try:
        sentence = request.args.get('sentence')
        result = tm.predict(sentence)
        if result['result'] == -1:
            return None 
        track = tm.suggest_song(result)

        return {
            'result': result,
            'track' : track
        }
    except:
        return {
            'message': 'error'
        },500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
