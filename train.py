import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(csv_path, sample_size=10000):
    """Load and optionally sample the dataset."""
    try:
        df = pd.read_csv(csv_path, sep=';')  # Assuming semicolon separator as per Kaggle
        logging.info(f"Loaded dataset with shape: {df.shape}")
        logging.info(f"Cardio distribution: {df['cardio'].value_counts()}")

        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            df.to_csv('df_sample.csv', index=False)
            logging.info(f"Sampled {sample_size} rows and saved to df_sample.csv")

        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """Preprocess the data: drop id, convert age to years."""
    df = df.copy()
    df.drop('id', axis=1, inplace=True)
    df['age_years'] = (df['age'] / 365).round(1)
    return df

def prepare_features_target(df, target='cardio'):
    """Prepare features and target."""
    features = ['height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'gender', 'age_years']
    X = df[features]
    y = df[target]
    return X, y

def encode_and_impute(X, encoders_path='model/encoders/', imputer_path='model/imputer.pkl'):
    """Encode categorical features and impute missing values."""
    os.makedirs(encoders_path, exist_ok=True)
    categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']  # Assuming these are categorical
    encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
            joblib.dump(le, os.path.join(encoders_path, f'encoder_{col}.pkl'), compress=3)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    joblib.dump(imputer, imputer_path, compress=3)
    return X_imputed, encoders

def check_balance_and_smote(X_train, y_train):
    """Check class balance and apply SMOTE if imbalanced."""
    class_counts = y_train.value_counts()
    minority_ratio = class_counts.min() / class_counts.sum()
    if minority_ratio < 0.4:
        logging.info("Applying SMOTE due to imbalance.")
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        return X_train_sm, y_train_sm
    else:
        logging.info("SMOTE skipped — already balanced.")
        return X_train, y_train

def feature_selection(X_train, y_train, top_features_path='model/top_features.pkl'):
    """Select top 10 features using RandomForest."""
    rf_temp = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_temp.fit(X_train, y_train)
    importances = rf_temp.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10
    top_features = X_train.columns[indices].tolist()
    logging.info(f"Top features: {top_features}")
    joblib.dump(top_features, top_features_path, compress=3)
    return top_features

def apply_poly_and_scale(X_train, X_test, top_features, poly_path='model/poly.pkl', scaler_path='model/scaler.pkl'):
    """Apply polynomial features and scaling."""
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X_train_poly = poly.fit_transform(X_train_top)
    X_test_poly = poly.transform(X_test_top)
    joblib.dump(poly, poly_path, compress=3)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    joblib.dump(scaler, scaler_path, compress=3)

    return X_train_scaled, X_test_scaled

def tune_and_train_model(model_name, model, param_grid, X_train, y_train, X_test, y_test):
    """Tune hyperparameters and train model."""
    search = RandomizedSearchCV(model, param_grid, n_iter=50, cv=2, scoring='recall', n_jobs=1, random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    logging.info(f"Best params for {model_name}: {search.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    if hasattr(best_model, 'predict_proba'):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': auc
    }
    return best_model, metrics

def build_ann(X_train, y_train, X_test, y_test, apply_class_weight=False):
    """Build and train ANN model."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    class_weight = None
    if apply_class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=[early_stop], class_weight=class_weight, verbose=0)

    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    y_proba = model.predict(X_test).flatten()
    auc = roc_auc_score(y_test, y_proba)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': auc
    }
    return model, metrics

def main(sample_size):
    # Load and preprocess
    df = load_data('cardio_train.csv', sample_size)
    df = preprocess_data(df)
    X, y = prepare_features_target(df)

    # Encode and impute
    X, encoders = encode_and_impute(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Train class balance: {y_train.value_counts()}")
    logging.info(f"Test class balance: {y_test.value_counts()}")

    # SMOTE
    X_train, y_train = check_balance_and_smote(X_train, y_train)

    # Feature selection
    top_features = feature_selection(X_train, y_train)

    # Poly and scale
    X_train_scaled, X_test_scaled = apply_poly_and_scale(X_train, X_test, top_features)

    # Models and params
    models = {
        'dt': (DecisionTreeClassifier(random_state=42), {'max_depth': [5,10,15,None], 'min_samples_split': [2,5,10], 'min_samples_leaf': [1,2,4]}),
        'svm': (SVC(probability=True, random_state=42), {'C': [0.1,1,10,100], 'gamma': ['scale','auto',0.01,0.1], 'kernel': ['rbf','linear']}),
        'rf': (RandomForestClassifier(random_state=42), {'n_estimators': [50,100,200], 'max_depth': [5,10,15,None], 'min_samples_split': [2,5,10], 'max_features': ['sqrt','log2',None]}),
        'xgb': (XGBClassifier(random_state=42), {'n_estimators': [50,100,200], 'max_depth': [3,5,7], 'learning_rate': [0.01,0.1,0.2], 'subsample': [0.6,0.8,1.0], 'gamma': [0,0.1,0.2]})
    }

    results = {}
    apply_class_weight = not (y_train.value_counts().min() / y_train.value_counts().sum() >= 0.4)  # If SMOTE not applied and imbalanced

    for name, (model, params) in models.items():
        try:
            best_model, metrics = tune_and_train_model(name, model, params, X_train_scaled, y_train, X_test_scaled, y_test)
            results[name] = metrics
            joblib.dump(best_model, f'model/{name}.pkl', compress=3)
        except Exception as e:
            logging.error(f"Error training {name}: {e}")

    # ANN
    try:
        ann_model, ann_metrics = build_ann(X_train_scaled, y_train, X_test_scaled, y_test, apply_class_weight)
        results['ann'] = ann_metrics
        ann_model.save('model/ann.h5')
    except Exception as e:
        logging.error(f"Error training ANN: {e}")

    # Print results
    print("\nModel Evaluation Results:")
    print("Model\tAccuracy\tPrecision\tRecall\tF1\tROC-AUC")
    for name, metrics in results.items():
        print(f"{name}\t{metrics['accuracy']:.3f}\t{metrics['precision']:.3f}\t{metrics['recall']:.3f}\t{metrics['f1']:.3f}\t{metrics['roc_auc']:.3f}" if metrics['roc_auc'] else f"{name}\t{metrics['accuracy']:.3f}\t{metrics['precision']:.3f}\t{metrics['recall']:.3f}\t{metrics['f1']:.3f}\tN/A")

    logging.info("Training complete. Models saved to /model/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=10000, help='Sample size if dataset > 10000')
    args = parser.parse_args()
    main(args.sample_size)
