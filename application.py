from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

# Load your machine learning model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return render_template('./index.html', prediction_text=f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
