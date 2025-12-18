import time
import os
import json
from openai import OpenAI, APIError

# Configuration
LITELLM_BASE_URL = "http://3.110.18.218"
# NOTE: In production, load this from os.environ.get("LITELLM_API_KEY")
LITELLM_API_KEY = "sk-pwxRnAJMGuCx8ek_PQBlJw"  # REPLACE THIS WITH YOUR ACTUAL KEY
# MODEL_NAME = "gemini-2.5-flash"
# MODEL_NAME = "gemini-2.5-pro"
MODEL_NAME = "gemini-3-pro-preview"

class LiteLLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=LITELLM_API_KEY,
            base_url=LITELLM_BASE_URL
        )

    def generate_completion(self, messages, json_mode=False):
        """
        Wraps the OpenAI call with the specific Exponential Backoff logic 
        requested by the Hackathon guidelines.
        """
        max_retries = 5
        base_delay = 1
        
        # Prepare kwargs
        kwargs = {
            "model": MODEL_NAME,
            "messages": messages,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            
            except APIError as e:
                if e.status_code == 429:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"[Warning] Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[Error] API Error {e.status_code}: {e.message}")
                    raise e
            except Exception as e:
                print(f"[Error] Unexpected error: {str(e)}")
                raise e
        
        raise Exception("Failed to make API call after multiple retries.")