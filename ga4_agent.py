import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, 
    DateRange, 
    Dimension, 
    Metric,
    OrderBy
)
from llm_client import LiteLLMClient

# --- 1. Define the 'Language' of the Agent (Pydantic Models) ---
class GA4QuerySchema(BaseModel):
    """
    Structured representation of a Google Analytics 4 API Request.
    The LLM must populate this structure.
    """
    start_date: str = Field(..., description="Start date (e.g., '2023-01-01', '30daysAgo', 'yesterday')")
    end_date: str = Field(..., description="End date (e.g., 'today', 'yesterday')")
    metrics: List[str] = Field(..., description="List of GA4 metric names (e.g., 'activeUsers', 'sessions', 'screenPageViews')")
    dimensions: List[str] = Field(default=[], description="List of GA4 dimension names (e.g., 'city', 'pagePath', 'date')")
    limit: int = Field(10, description="Row limit for the report")
    reasoning: str = Field(..., description="Brief explanation of why these metrics/dimensions were chosen.")

# --- 2. The Agent Class ---
class GA4Agent:
    def __init__(self):
        self.llm = LiteLLMClient()
        # Path to the credentials file (Must be at root as per hackathon rules)
        self.credentials_path = os.path.join(os.getcwd(), 'credentials.json')

    def _get_ga4_client(self):
        """Initializes the official GA4 Python Client using the JSON key."""
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"credentials.json not found at {self.credentials_path}")
        
        # Set the env var strictly for this process scope
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
        return BetaAnalyticsDataClient()

    def _generate_query_config(self, user_question: str) -> GA4QuerySchema:
        """
        Step 1: Ask LLM to convert Natural Language -> GA4 API Parameters
        """
        system_prompt = """
        You are an expert Google Analytics 4 (GA4) Data Engineer.
        Your goal is to translate a user's natural language question into a structured JSON configuration.

        ### REQUIRED OUTPUT FORMAT (JSON ONLY):
        {
            "start_date": "YYYY-MM-DD" or "30daysAgo",
            "end_date": "YYYY-MM-DD" or "today",
            "metrics": ["activeUsers", "sessions"],
            "dimensions": ["date", "pagePath"],
            "limit": 10,
            "reasoning": "Explain why you chose these metrics."
        }

        ### RULES:
        1. Keys MUST be snake_case (e.g., use 'start_date', NOT 'startDate').
        2. 'metrics' and 'dimensions' must be simple lists of strings. DO NOT use objects like {"name": "activeUsers"}.
        3. Use valid GA4 metric names (camelCase inside the strings): activeUsers, sessions, screenPageViews, totalUsers.
        4. Use valid GA4 dimension names: date, city, country, pagePath, sessionSource.
        5. If the user asks for "Trends", ALWAYS include "date" in dimensions.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Question: {user_question}\n\nGenerate the JSON schema."}
        ]

        # Call LLM
        response_str = self.llm.generate_completion(messages, json_mode=True)
        
        try:
            # Validate output using Pydantic
            data = json.loads(response_str)
            return GA4QuerySchema(**data)
        except Exception as e:
            print(f"LLM produced invalid JSON: {response_str}")
            # Fallback: Raise a clearer error so we know what happened
            raise ValueError(f"Schema Validation Failed. LLM Output: {response_str}") from e

    def _execute_ga4_request(self, property_id: str, config: GA4QuerySchema):
        """
        Step 2: Execute the query against the actual Google API.
        """
        client = self._get_ga4_client()
        
        request = RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=[Dimension(name=d) for d in config.dimensions],
            metrics=[Metric(name=m) for m in config.metrics],
            date_ranges=[DateRange(start_date=config.start_date, end_date=config.end_date)],
            limit=config.limit
        )

        try:
            response = client.run_report(request)
        except Exception as e:
            return {"error": f"GA4 API Failed: {str(e)}"}

        # Parse response into a clean dictionary
        result_data = []
        headers = [h.name for h in response.dimension_headers] + [h.name for h in response.metric_headers]
        
        for row in response.rows:
            item = {}
            # Add dimensions
            for i, dim_val in enumerate(row.dimension_values):
                item[response.dimension_headers[i].name] = dim_val.value
            # Add metrics
            for i, met_val in enumerate(row.metric_values):
                item[response.metric_headers[i].name] = met_val.value
            result_data.append(item)

        return {
            "metadata": {"row_count": response.row_count},
            "data": result_data,
            "config_used": config.dict()
        }

    def _summarize_results(self, user_question: str, data: dict) -> str:
        """
        Step 3: Convert the raw data back into a natural language answer.
        """
        if "error" in data:
            return f"I encountered an error retrieving the data: {data['error']}"
        
        if not data.get("data"):
            return "No data was found for this query period."

        messages = [
            {"role": "system", "content": "You are a helpful data analyst. Summarize the following JSON data to answer the user's question. Be concise and professional."},
            {"role": "user", "content": f"Question: {user_question}\n\nData: {json.dumps(data['data'])}"}
        ]
        
        return self.llm.generate_completion(messages)

    def process_request(self, property_id: str, user_question: str, return_json: bool = False):
        """
        Main entry point for the API.
        """
        print(f"--- Processing: {user_question} ---")
        
        # 1. Plan
        query_config = self._generate_query_config(user_question)
        print(f"Generated Plan: {query_config.json()}")

        # 2. Execute
        raw_result = self._execute_ga4_request(property_id, query_config)

        # 3. Respond
        if return_json:
            return raw_result
        else:
            return self._summarize_results(user_question, raw_result)

# --- Quick Test (Run this file directly) ---
if __name__ == "__main__":
    # Mock usage
    agent = GA4Agent()
    
    # You need a valid property ID here to test effectively
    TEST_PROPERTY_ID = "516806835" 
    
    try:
        answer = agent.process_request(
            property_id=TEST_PROPERTY_ID, 
            user_question="How many active users did we have in the last 7 days vs the previous period?"
        )
        print("\nFINAL ANSWER:\n", answer)
    except Exception as e:
        print("\nTest failed (Expected if no credentials/PropertyID):", e)