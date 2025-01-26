from flask import Flask, request, jsonify
from model_loader import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = load_model("risk_factor_outage_model.bin")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_features = ["Temperature", "Humidity", "Precipitation", "Signal Strength", "Packet Loss", "Latency"]
        if not all(feature in data for feature in required_features):
            return jsonify({"error": f"Missing one or more required input features: {required_features}"}), 400

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

        # Initialize the outage response
        response = {
            "Outage": prediction_value
        }

        # If outage occurs, show the following:
        if prediction_value == 1:
            feature_importance = model.get_feature_importance()
            feature_names = required_features

            feature_importance_dict = dict(
                map(lambda x: (x[0], round(x[1], 2)), zip(feature_names, feature_importance))
            )

            sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

            # Top 3 causal features/variable
            top_three_features = sorted_features[:3]

            response["Due to"] = [{"Feature": name, "Importance": importance} for name, importance in top_three_features]

        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)