print("Starting script...")

from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

print("Imports done.")

app = Flask(__name__)

try:
    model = joblib.load("xgb_student_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    import sys
    sys.exit(1)

expected_features = ['Age', 'Gender', 'SocioeconomicStatus', 'Grades', 'Attendance', 'TimeSpentOnHomework', 'ClassParticipation']

def encode_input(data):
    try:
        age = float(data['Age'])
        gender = 1 if str(data['Gender']).lower() == 'male' else 0
        ses = {'low': 0, 'medium': 1, 'high': 2}.get(str(data['SocioeconomicStatus']).lower(), 1)
        grades = float(data['Grades'])
        attendance = float(data['Attendance'])
        time_hw = float(data['TimeSpentOnHomework'])
        participation = {'low': 0, 'medium': 1, 'high': 2}.get(str(data['ClassParticipation']).lower(), 1)

        values = [age, gender, ses, grades, attendance, time_hw, participation]
        return np.array(values).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Error encoding input data: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()

    missing = [f for f in expected_features if f not in data]
    if missing:
        return jsonify({'error': f"Missing fields: {missing}"}), 400

    try:
        X = encode_input(data)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        result = 'Pass' if prediction == 1 else 'Fail'
        return jsonify({'prediction': result})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    print("Starting Flask app...")
    try:
        print("Attempting to start server on http://127.0.0.1:8080")
        app.run(host='127.0.0.1', port=8080, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error starting Flask app: {e}")
        import traceback
        traceback.print_exc()
