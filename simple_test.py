import requests
import json

# 1. Check API Health
print("--- Phase 1: Checking API Health ---")
try:
    response = requests.get("http://127.0.0.1:8000/health")
    health = response.json()
    print(f"Server Response: {health}")
except Exception as e:
    print(f"Could not connect to API: {e}")
    exit()

# 2. Execute Bidding Logic
# Match the key 'model_loaded' returned by our v2.5.0 API
if health.get("model_loaded") is True:
    print("\n--- Phase 2: Sending Production Bid Request ---")
    payload = {
        "auction_id": "miq-auction-777",
        "user_id": 9999,
        "site_domain": "premium-news.com",
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X)",
        "floor_price": 2.15
    }
    
    bid_response = requests.post("http://127.0.0.1:8000/bid", json=payload)
    
    if bid_response.status_code == 200:
        result = bid_response.json()
        print("✅ SUCCESS: Received Bid Response from Engine")
        print(json.dumps(result, indent=4))
        
        if result['bid']:
            print(f"\nRESULT: We are bidding ${result['price']} to win this user!")
        else:
            print("\nRESULT: Model predicted no click. We are passing on this auction.")
    else:
        print(f"❌ ERROR: API returned status {bid_response.status_code}")
        print(bid_response.text)
else:
    print("\n❌ FAIL: API reports model is NOT loaded. Check 'make api' logs.")