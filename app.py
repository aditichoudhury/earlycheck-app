from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ── Load model files on startup ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, 'breast_cancer_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    print("=" * 50)
    print("  Model loaded successfully!")
    print(f"  Features expected: {len(feature_names)}")
    print("=" * 50)
except FileNotFoundError as e:
    print(f"ERROR: Could not find model file: {e}")
    print("Make sure breast_cancer_model.pkl, scaler.pkl,")
    print("and feature_names.json are in the same folder as app.py")
    model = None
    scaler = None
    feature_names = []


# ── Convert UI answers to model features ──
def answers_to_features(answers, score):
    """
    Maps the conversational UI answers to the 30 numeric
    features the Wisconsin dataset model expects.

    Base values = median benign patient values from the dataset.
    We shift them toward malignant ranges based on the risk score.
    """

    # Baseline values (typical benign patient medians)
    base = {
        'radius_mean': 12.0,
        'texture_mean': 18.0,
        'perimeter_mean': 78.0,
        'area_mean': 463.0,
        'smoothness_mean': 0.096,
        'compactness_mean': 0.104,
        'concavity_mean': 0.089,
        'concave points_mean': 0.048,
        'symmetry_mean': 0.181,
        'fractal_dimension_mean': 0.062,
        'radius_se': 0.405,
        'texture_se': 1.22,
        'perimeter_se': 2.87,
        'area_se': 40.3,
        'smoothness_se': 0.007,
        'compactness_se': 0.025,
        'concavity_se': 0.032,
        'concave points_se': 0.012,
        'symmetry_se': 0.020,
        'fractal_dimension_se': 0.004,
        'radius_worst': 14.3,
        'texture_worst': 25.7,
        'perimeter_worst': 92.0,
        'area_worst': 625.0,
        'smoothness_worst': 0.132,
        'compactness_worst': 0.254,
        'concavity_worst': 0.272,
        'concave points_worst': 0.115,
        'symmetry_worst': 0.290,
        'fractal_dimension_worst': 0.084
    }

    # Risk multiplier based on score
    # Score 0-4  = low    → stay near benign baseline
    # Score 5-10 = medium → shift 25% toward malignant
    # Score 11+  = high   → shift 55% toward malignant
    if score <= 4:
        multiplier = 1.0
    elif score <= 10:
        multiplier = 1.25
    else:
        multiplier = 1.55

    # Apply multiplier to size features (larger = more concerning)
    size_features = [
        'radius_mean', 'perimeter_mean', 'area_mean',
        'radius_worst', 'perimeter_worst', 'area_worst',
        'radius_se', 'area_se'
    ]
    for key in size_features:
        base[key] *= multiplier

    # Apply multiplier to shape irregularity features
    shape_features = [
        'concavity_mean', 'compactness_mean', 'concave points_mean',
        'concavity_worst', 'compactness_worst', 'concave points_worst',
        'concavity_se', 'compactness_se'
    ]
    for key in shape_features:
        base[key] *= multiplier

    # Build array in exact column order the model was trained on
    features = [base[name] for name in feature_names]
    return features


# ── Guidance text per risk level ──
def get_guidance(risk_level, malignant_prob):
    if risk_level == 'low':
        return (
            "Based on what you shared, your profile aligns more closely with "
            "lower-risk patterns. Keep doing monthly self-exams at home and "
            "schedule an annual clinical check-up with your doctor. "
            "This is not a medical diagnosis."
        )
    elif risk_level == 'medium':
        return (
            "Some of your responses indicate patterns that are worth investigating "
            "further with a healthcare professional. We recommend booking a clinical "
            "breast exam within the next 4 weeks. Only a doctor can give you a "
            "proper diagnosis, but paying attention early is the right move."
        )
    else:
        return (
            "Your responses indicate patterns that need prompt medical attention. "
            "Please contact a doctor or breast health clinic this week. "
            "Early detection significantly improves outcomes. "
            "You are not alone — bring someone you trust with you."
        )


# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

# Health check — open this in browser to confirm server is running
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'message': 'Early Check API is running!',
        'model_loaded': model is not None,
        'features_count': len(feature_names)
    })


# Main prediction endpoint — called by index.html
@app.route('/predict', methods=['POST'])
def predict():

    # Block if model failed to load
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Check that .pkl files are in the same folder as app.py'
        }), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        answers = data.get('answers', [])
        score   = data.get('score', 0)

        if not isinstance(score, (int, float)):
            return jsonify({'error': 'score must be a number'}), 400

        # Convert answers → feature vector
        features = answers_to_features(answers, score)

        # Scale using the same scaler from training
        features_array  = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Get prediction and probabilities from model
        prediction  = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        benign_prob    = round(float(probability[0]) * 100, 1)
        malignant_prob = round(float(probability[1]) * 100, 1)

        # Map probability to risk level
        if malignant_prob < 25:
            risk_level = 'low'
        elif malignant_prob < 55:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        # Build full response
        response = {
            'prediction':      'malignant' if prediction == 1 else 'benign',
            'risk_level':      risk_level,
            'malignant_prob':  malignant_prob,
            'benign_prob':     benign_prob,
            'score':           int(score),
            'guidance':        get_guidance(risk_level, malignant_prob),
            'model_version':   'breast_cancer_rf_v1'
        }

        print(f"Prediction: {response['prediction']} | "
              f"Risk: {risk_level} | "
              f"Malignant prob: {malignant_prob}% | "
              f"Score: {score}")

        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
