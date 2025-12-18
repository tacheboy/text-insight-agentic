import pandas as pd
import io
import requests

SHEET_URL = "https://docs.google.com/spreadsheets/d/1zzf4ax_H2WiTBVrJigGjF2Q3Yz-qy2qMCbAMKvl6VEE/edit?gid=1438203274#gid=1438203274"

try:
    print("Downloading Sheet...")
    response = requests.get(SHEET_URL)
    response.raise_for_status()
    
    # Load with the robust engine
    df = pd.read_csv(
        io.StringIO(response.content.decode('utf-8')), 
        on_bad_lines='skip', 
        engine='python'
    )
    
    print(f"\nSUCCESS: Loaded {len(df)} rows.")
    print("--- FIRST 50 COLUMNS ---")
    
    # Print the clean names exactly as the Agent will generate them
    clean_cols = [c.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') for c in df.columns]
    
    for i, col in enumerate(clean_cols[:50]):
        print(f"{i}: {col}")
        
except Exception as e:
    print(f"Error: {e}")