import os
import json
from typing import List
from pydantic import BaseModel, Field
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, 
    DateRange, 
    Dimension, 
    Metric
)
from llm_client import LiteLLMClient

# --- 1. SAFE ALLOWLISTS (Prevents API 400 Errors) ---
# These are the standard GA4 API names. Restricting the LLM to these prevents hallucinations.
VALID_METRICS = {
    "activeUsers", "sessions", "screenPageViews", "eventCount", "totalUsers", 
    "newUsers", "bounceRate", "averageSessionDuration", "conversions", "engagementRate"
}

VALID_DIMENSIONS = {
    "date", "city", "country", "pagePath", "deviceCategory", "sessionSource", 
    "browser", "operatingSystem", "platform", "region", "eventName"
}

# --- 2. Pydantic Models for Validation ---
class GA4QuerySchema(BaseModel):
    """
    Structured representation of a Google Analytics 4 API Request.
    """
    start_date: str = Field(..., description="Start date (e.g., '2023-01-01', '30daysAgo')")
    end_date: str = Field(..., description="End date (e.g., 'today', 'yesterday')")
    metrics: List[str] = Field(..., description="List of valid GA4 metric names")
    dimensions: List[str] = Field(default=[], description="List of valid GA4 dimension names")
    limit: int = Field(10, description="Row limit for the report")
    reasoning: str = Field(..., description="Brief explanation of why these metrics/dimensions were chosen.")

# --- 3. The Agent Class ---
class GA4Agent:
    def __init__(self):
        self.llm = LiteLLMClient()
        # Path to the credentials file (Must be at root as per hackathon rules)
        self.credentials_path = os.path.join(os.getcwd(), 'credentials.json')

    def _get_ga4_client(self):
        """
        Initializes the GA4 Client.
        CRITICAL: Checks for credentials existence on every call to support 
        evaluators swapping the file at runtime.
        """
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"credentials.json not found at {self.credentials_path}")
        
        # Set the env var strictly for this process scope
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
        return BetaAnalyticsDataClient()

    def _generate_query_config(self, user_question: str) -> GA4QuerySchema:
        """
        Step 1: Ask LLM to convert Natural Language -> GA4 API Parameters
        """
        system_prompt = f"""
        You are an expert Google Analytics 4 (GA4) Data Engineer.
        Your goal is to translate a user's natural language question into a structured JSON configuration.

        ### ALLOWED METRICS (Use these exact keys):
        {list(VALID_METRICS)}
        
        ### ALLOWED DIMENSIONS (Use these exact keys):
        {list(VALID_DIMENSIONS)}

        ### REQUIRED OUTPUT FORMAT (JSON ONLY):
        {{
            "start_date": "YYYY-MM-DD" or "30daysAgo",
            "end_date": "YYYY-MM-DD" or "today",
            "metrics": ["activeUsers"],
            "dimensions": ["date"],
            "limit": 10,
            "reasoning": "Explain why you chose these metrics."
        }}

        ### RULES:
        1. Keys MUST be snake_case (e.g., use 'start_date', NOT 'startDate').
        2. 'metrics' and 'dimensions' must be simple lists of strings.
        3. If the user asks for "Trends" or "Over time", ALWAYS include "date" in dimensions.
        4. If vague (e.g., "last week"), use '7daysAgo' to 'yesterday'.
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
            raise ValueError(f"Schema Validation Failed. LLM Output: {response_str}") from e

    def _sanitize_config(self, config: GA4QuerySchema) -> GA4QuerySchema:
        """
        Safety Filter: Removes hallucinated metrics/dimensions that would crash the API.
        """
        # Filter invalid metrics
        original_metrics = config.metrics
        config.metrics = [m for m in config.metrics if m in VALID_METRICS]
        
        # Fallback if all metrics were invalid
        if not config.metrics:
            print(f"WARNING: All metrics {original_metrics} were invalid. Defaulting to 'activeUsers'.")
            config.metrics = ["activeUsers"]

        # Filter invalid dimensions
        config.dimensions = [d for d in config.dimensions if d in VALID_DIMENSIONS]
        
        return config

    def _execute_ga4_request(self, property_id: str, config: GA4QuerySchema):
        """
        Step 2: Execute the query against the actual Google API.
        """
        client = self._get_ga4_client()
        
        try:
            request = RunReportRequest(
                property=f"properties/{property_id}",
                dimensions=[Dimension(name=d) for d in config.dimensions],
                metrics=[Metric(name=m) for m in config.metrics],
                date_ranges=[DateRange(start_date=config.start_date, end_date=config.end_date)],
                limit=config.limit
            )
            response = client.run_report(request)
        except Exception as e:
            # Return error dict rather than crashing
            return {"error": f"GA4 API Request Failed: {str(e)}"}

        # Parse response into a clean dictionary
        result_data = []
        
        # Handle empty rows (Zero Traffic)
        if not response.rows:
            return {
                "metadata": {"row_count": 0, "is_empty": True},
                "data": [],
                "config_used": config.dict()
            }

        # Extract Headers
        dim_headers = [h.name for h in response.dimension_headers]
        met_headers = [h.name for h in response.metric_headers]

        for row in response.rows:
            item = {}
            # Add dimensions
            for i, dim_val in enumerate(row.dimension_values):
                item[dim_headers[i]] = dim_val.value
            # Add metrics
            for i, met_val in enumerate(row.metric_values):
                item[met_headers[i]] = met_val.value
            result_data.append(item)

        return {
            "metadata": {"row_count": response.row_count, "is_empty": False},
            "data": result_data,
            "config_used": config.dict()
        }

    def _summarize_results(self, user_question: str, data: dict) -> str:
        """
        Step 3: Convert the raw data back into a natural language answer.
        """
        # Handle API Errors
        if "error" in data:
            return f"I encountered a technical error contacting Google Analytics: {data['error']}"
        
        # Handle Zero Data (Common in test properties)
        if data.get("metadata", {}).get("is_empty"):
            config = data.get("config_used", {})
            return (
                f"I checked Google Analytics for '{user_question}' "
                f"(Period: {config.get('start_date')} to {config.get('end_date')}), "
                "but the report returned no data. This usually means the property has no active traffic for this date range."
            )

        # Standard Summarization
        messages = [
            {"role": "system", "content": "You are a helpful data analyst. Summarize the following JSON data to answer the user's question. Be concise and professional."},
            {"role": "user", "content": f"Question: {user_question}\n\nData: {json.dumps(data['data'])}"}
        ]
        
        return self.llm.generate_completion(messages)

    def process_request(self, property_id: str, user_question: str) -> str:
        """
        Main entry point for the API.
        """
        print(f"--- Processing GA4: {user_question} ---")
        
        try:
            # 1. Plan & Sanitize
            query_config = self._generate_query_config(user_question)
            query_config = self._sanitize_config(query_config)
            print(f"Generated Plan: {query_config.json()}")

            # 2. Execute
            raw_result = self._execute_ga4_request(property_id, query_config)

            # 3. Respond
            return self._summarize_results(user_question, raw_result)
            
        except Exception as e:
            return f"An unexpected error occurred in the Analytics Agent: {str(e)}"