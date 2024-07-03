import os

# Set the working directory to the directory where app.py is located
os.chdir(os.path.dirname(os.path.abspath("C:/Users/adith/crop/best_crop_prediction_model.joblib")))

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the best model and scaler
best_model = joblib.load('best_crop_prediction_model.joblib')
scaler = joblib.load('crop_prediction_scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        features = [
            float(request.form.get('N', 0.0)),
            float(request.form.get('P', 0.0)),
            float(request.form.get('K', 0.0)),
            float(request.form.get('temperature', 0.0)),
            float(request.form.get('humidity', 0.0)),
            float(request.form.get('ph', 0.0)),
            float(request.form.get('rainfall', 0.0))
        ]

        # Preprocess input data
        input_data = pd.DataFrame([features], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = best_model.predict(input_data_scaled)[0]

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('result.html', prediction='Error: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
