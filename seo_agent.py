import pandas as pd
import io
import requests
import traceback
from llm_client import LiteLLMClient

# Valid Sheet ID from your logs
SHEET_ID = "1zzf4ax_H2WiTBVrJigGjF2Q3Yz-qy2qMCbAMKvl6VEE"
GID = "1438203274"

SHEET_URLS = [
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}",
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}",
]

class SEOAgent:
    def __init__(self):
        self.llm = LiteLLMClient()
        self.df = None
        self._load_data()

    def _load_data(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        for i, url in enumerate(SHEET_URLS):
            try:
                print(f"\n[Attempt {i+1}] Connecting to Live Sheet...")
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                content = response.content.decode('utf-8')
                self.df = pd.read_csv(io.StringIO(content), on_bad_lines='skip', engine='python')

                # 1. CLEANUP: Normalize Column Names
                self.df.columns = [
                    c.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') 
                    for c in self.df.columns
                ]
                
                # 2. TYPE ENFORCEMENT
                # We try to convert obvious metrics, but we leave 'Object' columns alone 
                # so the LLM can regex them later.
                numeric_cols = ['Word_Count', 'Status_Code', 'Size_Bytes', 'Response_Time']
                for col in numeric_cols:
                    if col in self.df.columns:
                        if self.df[col].dtype == 'object':
                             self.df[col] = self.df[col].astype(str).str.replace(',', '')
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
                
                print(f"✅ SUCCESS: Loaded {len(self.df)} rows from live sheet.")
                return 
                
            except Exception as e:
                print(f"❌ Failed attempt {i+1}: {e}")
        
        print("\n⚠️ CRITICAL: Could not load data.")
        self.df = pd.DataFrame()

    def process_request(self, user_question: str) -> str:
        if self.df.empty:
            self._load_data()
            if self.df.empty:
                return "SEO Data is currently unavailable (Download Failed)."

        # --- KEY UPGRADE: DATA PREVIEW ---
        # We convert the first 3 rows to a string so the LLM can "SEE" the data content.
        # This allows it to find "Implicit" data (e.g., word counts hidden in violation text).
        data_preview = self.df.head(3).to_markdown(index=False)
        
        # Schema with types
        schema_info = [f"{col} ({dtype})" for col, dtype in self.df.dtypes.items()]
        schema_str = ", ".join(schema_info)
        
        system_prompt = f"""
        You are a Python Data Analyst. 
        
        ### DATA SCHEMA:
        [{schema_str}]

        ### DATA PREVIEW (First 3 Rows):
        {data_preview}
        
        ### TASK:
        Write Python code to answer: "{user_question}"
        
        ### RULES FOR DERIVED DATA:
        1. If the exact column (e.g., "Word_Count") is missing, LOOK AT THE DATA PREVIEW.
        2. Can you derive it? 
           - Example: If 'Size_Bytes' exists, you might estimate word count (Size / 5).
           - Example: If 'All_Violations' contains text like "Low word count (200)", extract it using regex.
        3. If no derivation is possible, explain clearly why based on the columns available.
        
        ### CODING RULES:
        1. Store result in variable `result`.
        2. Do not use print().
        3. Return ONLY valid python code.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {user_question}"}
        ]

        try:
            generated_code = self.llm.generate_completion(messages)
            cleaned_code = generated_code.replace("```python", "").replace("```", "").strip()
            
            local_vars = {"df": self.df}
            exec(cleaned_code, {}, local_vars)
            execution_result = local_vars.get("result", "No result variable found.")
            
            summary_prompt = [
                {"role": "system", "content": "Summarize the data into a natural language answer."},
                {"role": "user", "content": f"Q: {user_question}\nData: {execution_result}"}
            ]
            return self.llm.generate_completion(summary_prompt)

        except Exception as e:
            return f"Error executing analysis: {str(e)}"