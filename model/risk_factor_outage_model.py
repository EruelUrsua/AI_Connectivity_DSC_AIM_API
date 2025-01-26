import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
from catboost import CatBoostClassifier

# Variable setting and value seeding
np.random.seed(42)

n_samples = 1500

temperature = np.random.uniform(20, 40, n_samples) # Temperature in °C
precipitation = np.random.uniform(30, 90, n_samples) # Precipitation in mm
humidity = np.random.uniform(0, 100, n_samples) # Humidity in gm-3
signal_strength = np.random.uniform(-100, -30, n_samples) # Signal strength in dB
packet_loss = np.random.uniform(0, 20, n_samples)  # Packet loss in %
latency = np.random.uniform(50, 200, n_samples) # Latency in ms

# Noise generation
noise = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

# Environmental conditionals affecting signal strength and packet loss
factors_signal_strength = (temperature > 38) | (humidity > 90) | (precipitation > 100)
factors_packet_loss = (temperature > 38) | (humidity > 90) | (precipitation > 100)
factors_latency = (temperature > 38) | (humidity > 90) | (precipitation > 100)

# Adjusted coefficient for connectivity factors

# signal_strength = signal_strength - (10 * factors_signal_strength.astype(int)) # Signal attenuation 10 dBm
# packet_loss = packet_loss + (5 * factors_packet_loss.astype(int)) # Packet loss increase by 5%
# latency = latency + (20 * factors_latency.astype(int)) # Latency increase by 20ms

signal_strength -= (temperature - 38) * 0.5  # 1°C increase = 0.5 dB decrease
signal_strength -= (precipitation - 100) * 0.2  # 1 mm increase = 0.2 dB decrease
signal_strength -= (humidity - 90) * 0.3  # 1 gm-3 increase = 0.3 dB decrease

packet_loss += (temperature - 38) * 0.1  # 1°C increase = 0.1% increase in packet loss
packet_loss += (precipitation - 100) * 0.2  # 1 mm increase = 0.2% increase in packet loss
packet_loss += (humidity - 90) * 0.1  # 1 gm-3 increase = 0.1% increase in packet loss

latency += (temperature - 38) * 0.05  # 1°C increase = 0.05 ms increase in latency
latency += (precipitation - 100) * 0.1  # 1 mm increase = 0.1 ms increase in latency
latency += (humidity - 90) * 0.05  # 1 gm-3 increase = 0.05 ms increase in latency

# connectivity = ""
# if (np.any((signal_strength > -90) & (signal_strength < -85)) or np.any((packet_loss > 5) & (packet_loss < 10))
#         or np.any((latency > 95) & (latency < 100))):
#     connectivity = "weak"
# else:
#     connectivity = "strong"

# outage = (
#     (temperature > 38).astype(int) |
#     (precipitation > 100).astype(int) |
#     (signal_strength < -90).astype(int) |
#     (packet_loss > 10).astype(int) |
#     (humidity > 90).astype(int) |
#     (latency > 100).astype(int) |
#     noise
# )

# Define connectivity as 'weak' or 'strong'
connectivity = np.where(
    (signal_strength > -90) & (signal_strength < -85) |
    (packet_loss > 5) & (packet_loss < 10) |
    (latency > 95) & (latency < 100) |
    noise,
    'weak',
    'strong'
)

# Map outage condition where 'weak' connectivity = (1) and 'strong' connectivity = (0)
outage = np.where(connectivity == 'weak', 1, 0)

data = pd.DataFrame({
    'Temperature': temperature,
    'Precipitation': precipitation,
    'Humidity': humidity,
    'Signal Strength': signal_strength,
    'Packet Loss': packet_loss,
    'Outage': outage,
})

print(data.head())

# Train model

target_col = 'Outage'

X = data.loc[:, data.columns != target_col]
y = data.loc[:, target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Training set size: {X_train.shape[0]} samples')
print(f'Testing set size: {X_test.shape[0]} samples')

X_train.head()

y_train.head()

features = list(X_train.columns)

print(y_train.value_counts())
print(y_test.value_counts())

model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, verbose = 100, eval_metric='F1')

model.fit(X_train, y_train, plot=True, eval_set=(X_test, y_test))

y_pred = model.predict(X_test)

# Evaluate model
print(f'Predictions: {y_pred[:100]}')

print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'F1: {f1_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_pred):.4f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model.save_model("risk_factor_outage_model.bin")
print("Model saved as risk_factor_outage_model.bin")

# Load model to file path
def load_model(file_path):
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(file_path)  # Load model using CatBoost's method
    print(f"Model loaded from {file_path}")
    return loaded_model

# Test loading the model
loaded_model = load_model("risk_factor_outage_model.bin")