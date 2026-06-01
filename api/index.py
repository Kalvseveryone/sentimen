from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

model = None
vectorizer = None

def load_models():
    global model, vectorizer
    if model is None:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    if vectorizer is None:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        load_models()
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        
        # 0 = Negative, 1 = Positive
        sentiment = 'POSITIF' if prediction == 1 else 'NEGATIF'
        
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

import random
@app.route('/api/random', methods=['GET'])
def get_random():
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'amazon_cells_labelled.txt')
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            return jsonify({'error': 'Dataset empty'}), 404
            
        random_line = random.choice(lines).strip()
        parts = random_line.split('\t')
        if len(parts) >= 2:
            text = parts[0]
            label = int(parts[1])
            sentiment = 'POSITIF' if label == 1 else 'NEGATIF'
            return jsonify({'text': text, 'actual_sentiment': sentiment})
        else:
            return jsonify({'error': 'Invalid dataset format'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5328)
