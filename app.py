from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

model   = joblib.load('model/heart_disease_model.pkl')
scaler  = joblib.load('model/scaler.pkl')
columns = joblib.load('model/feature_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_vals = [float(data.get(col, 0)) for col in columns]
        input_arr  = np.array(input_vals).reshape(1, -1)
        scaled     = scaler.transform(input_arr)
        prediction  = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]
        return jsonify({
            'prediction' : int(prediction),
            'result'     : '⚠️ Heart Disease Detected' if prediction == 1 else '✅ No Heart Disease',
            'probability': f"{probability * 100:.1f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
