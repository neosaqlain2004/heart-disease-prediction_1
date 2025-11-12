# Cardiovascular Disease Prediction ML Project

This project trains multiple machine learning models on the Kaggle Cardiovascular Disease dataset and provides a Flask web frontend for predictions.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train models:
   ```
   python train.py --sample-size 10000
   ```

## Usage

### Training
Run the training script to train models and save them to `/model/`:
```
python train.py --sample-size 10000
```

### Web Server
Start the Flask app:
```
python app.py
```
Or with Gunicorn:
```
gunicorn app:app --bind 0.0.0.0:8080
```

### API Usage
POST to `/predict` with JSON:
```json
{
  "age": 18250,
  "gender": 1,
  "height": 170,
  "weight": 70,
  "ap_hi": 120,
  "ap_lo": 80,
  "cholesterol": 1,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1,
  "model": "rf"
}
```
Response:
```json
{
  "prediction": 0,
  "probability": 0.234,
  "model": "rf"
}
```

## Deployment

### Railway/Render
- Use `Procfile` for deployment.
- Set environment variable `PREDICT_API_KEY` for API access.

### Memory Notes
- Models are loaded lazily to reduce memory usage.
- For production with all models, ensure sufficient RAM (e.g., paid plans).
- Models are compressed with joblib compress=3 for smaller size.

## Project Structure
- `train.py`: Training pipeline.
- `app.py`: Flask web app.
- `model_utils.py`: Helper functions.
- `/model/`: Saved models and preprocessors.
- `/templates/index.html`: Web UI.
- `/static/`: CSS and JS.
- `/notebooks/EDA.ipynb`: Basic EDA.
- `/tests/test_predict.py`: Simple test.

## Models Trained
- Decision Tree
- SVM
- Random Forest
- XGBoost
- ANN (Keras)

## Disclaimer
This tool is informational only — not medical advice.
