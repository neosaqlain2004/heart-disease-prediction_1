import os
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model as load_keras_model

# Lazy loading cache
model_cache = {}

def load_model(name):
    """Lazy load model."""
    if name not in model_cache:
        if name == 'ann':
            model_cache[name] = load_keras_model(f'model/{name}.h5')
        else:
            model_cache[name] = joblib.load(f'model/{name}.pkl')
    return model_cache[name]

def preprocess_input(input_data):
    """Preprocess input data."""
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Convert age to years if present
    if 'age' in df.columns:
        df['age_years'] = (df['age'] / 365).round(1)
    elif 'age_years' not in df.columns:
        raise ValueError("Age must be provided as 'age' or 'age_years'")

    # Load top features
    try:
        top_features = joblib.load('model/top_features.pkl')
    except Exception as e:
        raise RuntimeError(f"Missing or unreadable model/top_features.pkl: {e}")

    # Load encoders and imputer
    encoders = {}
    for col in ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']:
        enc_path = f'model/encoders/encoder_{col}.pkl'
        if os.path.exists(enc_path):
            encoders[col] = joblib.load(enc_path)

    try:
        imputer = joblib.load('model/imputer.pkl')
    except Exception as e:
        raise RuntimeError(f"Missing or unreadable model/imputer.pkl: {e}")

    # Encode
    for col, enc in encoders.items():
        if col in df.columns:
            # encoders expect 1D arrays / Series
            df[col] = enc.transform(df[col])

    # Drop the raw 'age' column if present; top_features likely expect 'age_years'
    if 'age' in df.columns and 'age_years' in df.columns:
        df = df.drop(columns=['age'])

    # Impute: ensure the imputer sees the same columns order it was fit on.
    # If a saved feature columns list exists, use that, otherwise use df.columns.
    # Determine expected feature order for the imputer
    if hasattr(imputer, 'feature_names_in_'):
        expected_cols = list(imputer.feature_names_in_)
    else:
        feature_cols_path = 'model/feature_columns.pkl'
        if os.path.exists(feature_cols_path):
            expected_cols = joblib.load(feature_cols_path)
        else:
            expected_cols = list(df.columns)

    # Add missing columns with NaN to match expected columns, then reindex to expected order
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    df_for_impute = df.reindex(columns=expected_cols)

    try:
        imputed_arr = imputer.transform(df_for_impute)
    except Exception as e:
        raise RuntimeError(f"Imputer failed: {e}")

    df_imputed = pd.DataFrame(imputed_arr, columns=df_for_impute.columns)

    # Select top features
    missing_top = [c for c in top_features if c not in df_imputed.columns]
    if missing_top:
        raise RuntimeError(f"Missing top features in input after preprocessing: {missing_top}")

    df_top = df_imputed[top_features]

    # Poly and scale
    poly = joblib.load('model/poly.pkl')
    scaler = joblib.load('model/scaler.pkl')

    df_poly = poly.transform(df_top)
    df_scaled = scaler.transform(df_poly)

    return df_scaled

def predict(model_name, input_df):
    """Make prediction."""
    model = load_model(model_name)
    if model_name == 'ann':
        proba = model.predict(input_df)[0][0]
        pred = int(proba > 0.5)
    else:
        proba = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

    # Cast to native python types
    try:
        proba = float(proba)
    except Exception:
        proba = float(np.asarray(proba).item())

    try:
        pred = int(pred)
    except Exception:
        pred = int(np.asarray(pred).item())

    return pred, proba
