import subprocess
import sys
import os
from datetime import datetime

print("="*80)
print(f"🤖 RENDER AUTOMATION - STARTED: {datetime.now()}")
print("="*80)

# Step 1: Run scraper
print("\n1️⃣ Running scraper...")
try:
    result = subprocess.run(
        ['python', 'automated_scraper.py'],
        capture_output=True,
        text=True,
        timeout=7000  # 30 min timeout
    )
    
    if result.returncode == 0:
        print("✅ Scraper completed successfully")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ Scraper failed!")
        print(result.stderr)
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    print("❌ Scraper timeout (>30 min)")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error running scraper: {e}")
    sys.exit(1)

# Step 2: Git operations
print("\n2️⃣ Pushing to GitHub...")

# CSV files to update
csv_files = [
    'tablets_cleaned_continuous.csv',
    'mobile_cleaned_70K.csv' 
]

subprocess.run(['git', 'config', '--global', 'user.email', 'bot@render.com'], check=True)
subprocess.run(['git', 'config', '--global', 'user.name', 'Render Bot'], check=True)

added_files = []
for csv in csv_files:
    if os.path.exists(csv):
        subprocess.run(['git', 'add', csv], check=True)
        added_files.append(csv)
        print(f"✅ Added: {csv}")

if not added_files:
    print("❌ No CSV files found to update")
    sys.exit(1)

# Check for changes
result = subprocess.run(
    ['git', 'status', '--porcelain'], 
    capture_output=True, 
    text=True
)

if not result.stdout.strip():
    print("ℹ️  No changes to commit (CSV unchanged)")
    sys.exit(0)

# Commit
commit_msg = f"🤖 Auto-update prices: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
print(f"✅ Committed: {commit_msg}")

# Push using environment variables
github_token = os.environ.get('GITHUB_TOKEN')  
github_repo = os.environ.get('GITHUB_REPO') 

if not github_token:
    print("❌ Missing GITHUB_TOKEN environment variable")
    print("   Set this in Render dashboard under Environment")
    sys.exit(1)

if not github_repo:
    print("❌ Missing GITHUB_REPO environment variable")
    print("   Format should be: username/repo-name")
    sys.exit(1)

push_url = f"https://{github_token}@github.com/{github_repo}.git"

result = subprocess.run(
    ['git', 'push', push_url, 'main'],
    capture_output=True,
    text=True,
    timeout=60
)
    

print("\n" + "="*80)
print("✅ AUTOMATION COMPLETE - SUCCESS")
print("="*80)
print(f"Files updated: {', '.join(added_files)}")
print(f"Pushed to: https://github.com/{github_repo}")
print("Streamlit will auto-refresh in 1-2 minutes")
print("="*80)
