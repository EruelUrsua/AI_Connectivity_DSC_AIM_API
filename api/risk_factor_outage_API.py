from flask import Flask, request, jsonify
from flask_cors import CORS
from model_loader import load_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Load the model
model = load_model("risk_factor_outage_model.bin")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate input features
        required_features = ["Temperature", "Humidity", "Precipitation", "Signal Strength", "Packet Loss", "Latency"]
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing required input feature: {feature}"}), 400
            if not isinstance(data[feature], (int, float)):
                return jsonify({"error": f"{feature} must be a numeric value"}), 400

        # Extract features
        features = [
            data["Temperature"],
            data["Humidity"],
            data["Precipitation"],
            data["Signal Strength"],
            data["Packet Loss"],
            data["Latency"]
        ]

        # Model prediction
        prediction = model.predict([features])
        prediction_value = int(prediction[0])
        probabilities = model.predict_proba([features])[0]  # Get probabilities

        # Initialize response
        response = {
            "Outage": prediction_value,
            "Probabilities": {"No Outage": probabilities[0], "Outage": probabilities[1]}
        }

        # If outage occurs, add feature importance
        if prediction_value == 1:
            feature_importance = model.get_feature_importance()
            feature_names = required_features

            feature_importance_dict = dict(
                map(lambda x: (x[0], round(x[1], 2)), zip(feature_names, feature_importance))
            )
            sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

            # Top 3 causal features with explanations
            response["Due to"] = [
                {"Feature": name, "Importance": importance, "Explanation": f"{name} has a strong impact on outages."}
                for name, importance in sorted_features[:3]
            ]

        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)