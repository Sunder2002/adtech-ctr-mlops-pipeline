import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def run_final_validation():
    print("--- Phase 1: System Health Check ---")
    try:
        health = requests.get(f"{BASE_URL}/health").json()
        print(f"Status: {health['status']} | Model Loaded: {health['model_loaded']}")
    except:
        print("Error: API is not running. Run 'make api' first.")
        return

    print("\n--- Phase 2: Behavioral Bid Request (Electronics Intent) ---")
    # This matches the v7.1 BidRequest schema
    payload = {
        "email": "hruthik@miq.com",
        "query": "I am looking for a high-performance laptop"
    }
    
    res = requests.post(f"{BASE_URL}/bid", json=payload).json()
    print(json.dumps(res, indent=4))
    
    if res['bid']:
        print(f"✅ SUCCESS: Model predicted a click with {res['proba']*100:.1f}% confidence.")
        print(f"ACTION: Bidding ${res['price']} to win this auction.")
    else:
        print("❌ PASS: Low intent detected.")

if __name__ == "__main__":
    run_final_validation()