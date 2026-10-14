import os
import sys
sys.path.insert(0, "C:\\Users\\HP\\AppData\\Roaming\\Python\\Python310\\site-packages")

from huggingface_hub import HfApi, create_repo, upload_folder

token_file = os.path.join(os.path.dirname(__file__), "hf_token.txt")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN and os.path.exists(token_file):
    with open(token_file, "r") as f:
        HF_TOKEN = f.read().strip()

HF_USERNAME = "TANUJ04"
REPO_NAME   = "electricity-theft-detection"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"
SPACE_DIR   = "huggingface_space_static"   # Static HTML dashboard (FREE)

api = HfApi(token=HF_TOKEN)

print(f"Creating STATIC Space (free): {REPO_ID} ...")
try:
    create_repo(
        repo_id    = REPO_ID,
        repo_type  = "space",
        space_sdk  = "static",     # <-- Static SDK is 100% FREE for everyone
        token      = HF_TOKEN,
        exist_ok   = True,
        private    = False
    )
    print(f"  Space created: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"  Note: {e}")

print(f"\nUploading files from '{SPACE_DIR}' ...")
try:
    upload_folder(
        folder_path    = SPACE_DIR,
        repo_id        = REPO_ID,
        repo_type      = "space",
        token          = HF_TOKEN,
        commit_message = "Deploy static ML dashboard — Electricity Theft Detection"
    )
    print(f"\nDEPLOYMENT COMPLETE!")
    print(f"   Live at: https://huggingface.co/spaces/{REPO_ID}")
    print(f"   Note: Refresh in ~30 seconds for the page to go live.")
except Exception as e:
    print(f"\nUPLOAD FAILED: {e}")
    print("   Tip: Check your HF_TOKEN has WRITE access at https://huggingface.co/settings/tokens")
