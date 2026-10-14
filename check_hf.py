import os
import sys
sys.path.insert(0, "C:\\Users\\HP\\AppData\\Roaming\\Python\\Python310\\site-packages")
from huggingface_hub import HfApi

token_file = os.path.join(os.path.dirname(__file__), "hf_token.txt")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN and os.path.exists(token_file):
    with open(token_file, "r") as f:
        HF_TOKEN = f.read().strip()

api = HfApi(token=HF_TOKEN)
user = api.whoami()
print("Username:", user["name"])
print("Email:",    user.get("email", "N/A"))
