import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import preprocess_input, predict

def test_predict():
    # Sample input
    sample_input = {
        'age': 50 * 365,  # 50 years
        'gender': 1,
        'height': 170,
        'weight': 70,
        'ap_hi': 120,
        'ap_lo': 80,
        'cholesterol': 1,
        'gluc': 1,
        'smoke': 0,
        'alco': 0,
        'active': 1
    }

    try:
        processed = preprocess_input(sample_input)
        pred, proba = predict('rf', processed)
        assert isinstance(pred, int)
        assert 0 <= proba <= 1
        print("Test passed: Prediction output keys exist and are valid.")
    except Exception as e:
        print(f"Test failed: {e}")
        raise

if __name__ == '__main__':
    test_predict()
