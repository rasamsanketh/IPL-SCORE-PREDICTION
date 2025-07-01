from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import requests

app = Flask(__name__)
CORS(app)

# Load the model and other preprocessing tools
model = load_model("model/ipl_model.h5")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/encoders.pkl")

# Replace with your OpenWeatherMap API key
WEATHER_API_KEY = "580b1a1fcd8cc58636cad2981b646c1b"

@app.route('/')
def home():
    return "🏏 IPL Score Predictor API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Get weather based on a city (or use fixed for demo)
    city = "Mumbai"  # You can let the frontend send this too
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}"

    try:
        weather_data = requests.get(weather_url).json()
        weather_main = weather_data['weather'][0]['main'].lower()

        if "rain" in weather_main:
            return jsonify({'message': "🌧️ It's raining. No match today!"})
    except:
        return jsonify({'message': "⚠️ Weather service failed. Try again."})

    try:
        input_data = [
            encoders['batting_team'].transform([data['batting_team']])[0],
            encoders['bowling_team'].transform([data['bowling_team']])[0],
            encoders['venue'].transform([data['venue']])[0],
            data['current_score'],
            data['balls_left'],
            data['wickets_left'],
            data['crr']
        ]

        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)
        predicted_score = int(prediction[0][0])

        return jsonify({'prediction': predicted_score})

    except Exception as e:
        return jsonify({'message': f"Prediction failed: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
