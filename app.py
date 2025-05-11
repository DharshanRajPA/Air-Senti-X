from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.ensemble_predictor import EnsemblePredictor
from utils.evaluation import get_emotion, calculate_urgency
import os

app = Flask(__name__)
CORS(app)

MODEL_PATHS = [
    "models/saved/bert_lstm.h5",
    "models/saved/bert_bilstm.h5",
    "models/saved/bert_cnn.h5"
]

for path in MODEL_PATHS:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Model file not found: {path}")

ensemble = EnsemblePredictor(model_paths=MODEL_PATHS)

@app.route('/')
def home():
    return jsonify({"message": "Air-Senti-X backend is running!"})

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json(force=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text input is empty"}), 400

    try:
        result = ensemble.predict(text)

        if "probabilities" in result:
            result["all_scores"] = list(result["probabilities"].values())

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
