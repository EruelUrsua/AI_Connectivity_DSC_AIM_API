import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "Temperature": 39,
    "Humidity": -70,
    "Precipitation": 1,
    "Signal Strength": -50,
    "Packet Loss": 10,
    "Latency": 5
}
response = requests.post(url, json=payload)
print(response.json())