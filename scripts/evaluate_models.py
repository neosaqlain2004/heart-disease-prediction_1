import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import sys

# Add project root to path so we can import model_utils
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from model_utils import preprocess_input, load_model

DATA_CSV = 'cardio_train.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_and_split(path):
    df = pd.read_csv(path, sep=';')
    df.drop('id', axis=1, inplace=True)
    df['age_years'] = (df['age'] / 365).round(1)
    X = df.drop(columns=['cardio'])
    y = df['cardio']
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def evaluate_model(name, model, X_test, y_test):
    # Preprocess test rows in batches
    # model_utils.preprocess_input expects a full DataFrame; we'll preprocess all at once
    Xp = preprocess_input(X_test.reset_index(drop=True))
    if name == 'ann':
        proba = model.predict(Xp).flatten()
        preds = (proba > 0.5).astype(int)
        proba_vals = proba
    else:
        if hasattr(model, 'predict_proba'):
            proba_vals = model.predict_proba(Xp)[:,1]
        else:
            # if no predict_proba (rare), use decision_function then normalize
            try:
                dec = model.decision_function(Xp)
                proba_vals = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
            except Exception:
                proba_vals = np.zeros(len(Xp))
        preds = (proba_vals > 0.5).astype(int)

    auc = roc_auc_score(y_test, proba_vals) if len(np.unique(y_test))>1 else None
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'roc_auc': auc
    }
    return metrics


def main():
    if not os.path.exists(DATA_CSV):
        print(f"Data file {DATA_CSV} not found in project root.")
        return
    X_train, X_test, y_train, y_test = load_and_split(DATA_CSV)
    model_names = ['dt','svm','rf','xgb','ann']
    results = {}
    for name in model_names:
        path = f'model/{name}.pkl' if name!='ann' else f'model/{name}.h5'
        if not os.path.exists(path):
            print(f"Skipping {name}: model artifact not found at {path}")
            continue
        print(f"Loading {name}...", end=' ')
        model = load_model(name)
        print("done")
        metrics = evaluate_model(name, model, X_test, y_test)
        results[name] = metrics

    print('\nEvaluation Results:')
    print('Model\tAccuracy\tPrecision\tRecall\tF1\tROC-AUC')
    for name, m in results.items():
        auc = f"{m['roc_auc']:.3f}" if m['roc_auc'] else 'N/A'
        print(f"{name}\t{m['accuracy']:.3f}\t{m['precision']:.3f}\t{m['recall']:.3f}\t{m['f1']:.3f}\t{auc}")

if __name__ == '__main__':
    main()
