from flask import Flask, request, jsonify, render_template
import logging
import time
import os
from model_utils import preprocess_input, predict

app = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Key
API_KEY = os.getenv('PREDICT_API_KEY')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    start_time = time.time()

    # API Key check
    if API_KEY and request.headers.get('X-API-Key') != API_KEY:
        return jsonify({'error': 'Invalid API Key'}), 401

    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Assume model selection or default to 'rf'. If client requests 'all', return all models.
        model_name = data.pop('model', 'rf')

        # Normalize input values: form inputs are strings, ensure numeric types where applicable.
        normalized = {}
        for k, v in data.items():
            if isinstance(v, str):
                v_str = v.strip()
                if v_str == '':
                    normalized[k] = v_str
                    continue
                # try integer
                try:
                    normalized[k] = int(v_str)
                    continue
                except Exception:
                    pass
                # try float
                try:
                    normalized[k] = float(v_str)
                    continue
                except Exception:
                    normalized[k] = v_str
            else:
                normalized[k] = v

        # Preprocess
        processed = preprocess_input(normalized)

        # If client asked for all models, iterate a known list and return per-model predictions.
        if model_name == 'all':
            models = ['dt', 'svm', 'rf', 'xgb', 'ann']
            results = {}
            for m in models:
                try:
                    pred, proba = predict(m, processed)
                    results[m] = {'prediction': int(pred), 'probability': float(proba)}
                except Exception as e:
                    results[m] = {'error': str(e)}

            response = {'results': results}
        else:
            # Predict single model for backward compatibility
            prediction, probability = predict(model_name, processed)
            response = {
                'prediction': int(prediction),
                'probability': float(probability),
                'model': model_name
            }

        logging.info(f"Prediction made in {time.time() - start_time:.2f}s")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
