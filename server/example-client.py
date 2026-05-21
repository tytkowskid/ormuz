import requests

# POST request
response = requests.post(
    "http://127.0.0.1:5000/buy",
    json={"ticker": "aapl", "quantity": 1.5}
)

print("POST response:")
print(response.json())