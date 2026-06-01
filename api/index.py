from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import math
import re
import random

app = Flask(__name__)
CORS(app)

model_data = None

def load_model():
    global model_data
    if model_data is None:
        json_path = os.path.join(os.path.dirname(__file__), 'model_data.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

def manual_predict(text):
    load_model()
    # Tokenize the same way as CountVectorizer
    tokens = re.findall(r"(?u)\b\w\w+\b", text.lower())
    
    # Count frequencies
    token_counts = {}
    for token in tokens:
        if token in model_data['vocab']:
            idx = model_data['vocab'][token]
            token_counts[idx] = token_counts.get(idx, 0) + 1
            
    # Calculate log probabilities for each class
    scores = []
    for c_idx in range(len(model_data['classes'])):
        score = model_data['class_log_prior'][c_idx]
        for token_idx, count in token_counts.items():
            score += count * model_data['feature_log_prob'][c_idx][token_idx]
        scores.append(score)
        
    # Get argmax
    pred_class_idx = scores.index(max(scores))
    return model_data['classes'][pred_class_idx]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        prediction = manual_predict(text)
        
        # 0 = Negative, 1 = Positive
        sentiment = 'POSITIF' if prediction == 1 else 'NEGATIF'
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
