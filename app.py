from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
import emotion_recognizer as tm
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})



ZEMBEREK_PATH = r'data/zemberek-full.jar'
startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
morphology = TurkishMorphology.createWithDefaults()

@app.route('/')
def hello_world():
    try:
        sentence = request.args.get('sentence')
        result = tm.predict(sentence, morphology)
        if result['result'] == -1:
            return None 
        track = tm.suggest_song(result)

        return {
            'result': result,
            'track' : track
        }
    except Exception as e:
        print(e)
        return {
            'message': 'error'
        },500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
