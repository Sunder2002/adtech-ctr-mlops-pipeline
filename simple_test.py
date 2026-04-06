import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_user(name, payload):
    print(f"\n--- Testing User: {name} ---")
    response = requests.post(f"{BASE_URL}/bid", json=payload)
    print(json.dumps(response.json(), indent=4))

# USER 1: High Intention (Shopping + Electronics interest + High History)
high_value_user = {
    "auction_id": "auc-high-001",
    "user_id": 888222,
    "user_interests": "Electronics,Sports,Gaming",
    "historical_ctr": 0.08,
    "site_category": "shopping",
    "floor_price": 2.50
}

# USER 2: Low Intention (News site + No relevant interests + Low History)
low_value_user = {
    "auction_id": "auc-low-002",
    "user_id": 111333,
    "user_interests": "Cooking,Gardening",
    "historical_ctr": 0.002,
    "site_category": "news",
    "floor_price": 1.00
}

if __name__ == "__main__":
    try:
        test_user("Electronics Buyer", high_value_user)
        test_user("Casual News Reader", low_value_user)
    except Exception as e:
        print(f"Error: API might not be running. {e}")