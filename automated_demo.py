import requests
import time
import random

BASE_URL = "http://127.0.0.1:8000"

# Real-world scenarios to prove ML discrimination
scenarios = [
    {"email": "tech_buyer@gmail.com", "query": "macbook pro m3 chip"},
    {"email": "car_enthusiast@yahoo.com", "query": "tesla model s tires"},
    {"email": "casual_user@outlook.com", "query": "how is the weather"},
    {"email": "fashion_icon@me.com", "query": "nike summer collection"}
]

def run_autopilot():
    print("🚀 Starting MiQ RTB Autopilot Simulation...")
    print("Check the /dashboard to see these results in real-time!")
    
    for i in range(10):
        s = random.choice(scenarios)
        try:
            res = requests.post(f"{BASE_URL}/bid", json=s).json()
            icon = "✅" if res['bid'] else "❌"
            print(f"[{i+1}] User: {s['email']} | Query: {s['query'][:15]}... | Bid: {icon} | Prob: {res['proba']*100:.1f}%")
        except Exception as e:
            print(f"Connection Error: {e}")
        time.sleep(2)

if __name__ == "__main__":
    run_autopilot()