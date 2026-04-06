import requests
import time
import random

BASE_URL = "http://127.0.0.1:8000"

# SCENARIOS
users = [
    {"email": "tech_guy@gmail.com", "queries": ["laptop", "macbook pro", "gaming pc"]},
    {"email": "car_fan@yahoo.com", "queries": ["tesla model 3", "offroad tires", "fast cars"]},
    {"email": "random_user@outlook.com", "queries": ["icecream", "weather today", "how to sleep"]}
]

def run_sim():
    print("🚀 Starting Automated MiQ Bidding Simulation...")
    for i in range(20):
        user = random.choice(users)
        query = random.choice(user["queries"])
        
        try:
            res = requests.post(f"{BASE_URL}/bid", json={
                "email": user["email"],
                "query": query
            }).json()
            
            status = "✅ WON" if res['bid'] else "❌ PASS"
            print(f"User: {user['email']} | Query: {query:15} | Result: {status} | Prob: {res['proba']*100:.1f}%")
        except:
            print("Error: Is 'make api' running?")
        
        time.sleep(1) # Simulate real-world traffic spacing

if __name__ == "__main__":
    run_sim()