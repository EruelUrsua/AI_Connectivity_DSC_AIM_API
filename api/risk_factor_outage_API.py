from flask import Flask, request, jsonify
from api.model_loader import load_model

# Initialize Flask app
app = Flask(__name__)

model = load_model("risk_factor_outage_model.bin")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_features = ["Temperature", "Humidity", "Precipitation", "Signal Strength", "Packet Loss", "Latency"]
        if not all(feature in data for feature in required_features):
            return jsonify({"error": "Missing one or more required input features"}), 400

        # Extract input features
        features = [
            data["Temperature"],
            data["Humidity"],
            data["Precipitation"],
            data["Signal Strength"],
            data["Packet Loss"],
            data["Latency"]
        ]

        prediction = model.predict([features])

        prediction_value = int(prediction[0])

        return jsonify({"outage": prediction_value})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)