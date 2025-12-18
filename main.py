from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
import json

# Import your agents
from ga4_agent import GA4Agent
from seo_agent import SEOAgent
from llm_client import LiteLLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agents = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        agents["ga4"] = GA4Agent()
        agents["seo"] = SEOAgent()
        agents["llm"] = LiteLLMClient()
        logger.info("✅ ALL AGENTS INITIALIZED")
    except Exception as e:
        logger.error(f"❌ Agent Init Failed: {e}")
    yield
    agents.clear()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    propertyId: Optional[str] = None
    query: str

class QueryResponse(BaseModel):
    answer: str

def route_intent(query: str) -> str:
    """
    Decides if the query needs GA4, SEO, or BOTH.
    """
    prompt = """
    Classify the user question into:
    1. 'GA4' - Traffic, users, sessions, views, time-trends.
    2. 'SEO' - Status codes, https, meta tags, word count, crawl depth.
    3. 'BOTH' - If the user asks to correlate traffic/views with SEO metrics (e.g., "Top pages by views and their title tags").

    Return ONLY one word: 'GA4', 'SEO', or 'BOTH'.
    """
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": query}]
    return agents["llm"].generate_completion(messages).strip().upper()

async def handle_hybrid_query(property_id: str, query: str) -> str:
    """
    Tier 3 Logic: Multi-Agent Orchestration
    """
    # 1. Decompose
    plan_prompt = f"""
    You are a Planner Agent. The user asked: "{query}".
    Break this into two sub-questions:
    1. A question for Google Analytics (GA4) to get the pages/metrics.
    2. A question for the SEO Audit to check technical details for those pages.
    
    Return JSON format: {{"ga4_query": "...", "seo_query": "..."}}
    """
    
    try:
        plan_json = agents["llm"].generate_completion(
            [{"role": "user", "content": plan_prompt}], 
            json_mode=True
        )
        plan = json.loads(plan_json)
        logger.info(f"Hybrid Plan: {plan}")

        # 2. Execute Parallel (or Sequential)
        # Step A: Get GA4 Data
        ga4_result = agents["ga4"].process_request(property_id, plan["ga4_query"])
        
        # Step B: Get SEO Data (Contextualized)
        # We append the GA4 context so the SEO agent knows which pages to focus on
        seo_context_query = f"{plan['seo_query']}. Context from Analytics: {ga4_result}"
        seo_result = agents["seo"].process_request(seo_context_query)

        # 3. Synthesize
        final_prompt = f"""
        Synthesize a final answer based on these two reports:
        
        [GA4 REPORT]: {ga4_result}
        [SEO REPORT]: {seo_result}
        
        User Question: {query}
        """
        return agents["llm"].generate_completion([{"role": "user", "content": final_prompt}])

    except Exception as e:
        logger.error(f"Hybrid Error: {e}")
        return f"I tried to analyze both sources but failed: {e}"

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        # 1. Routing
        intent = "SEO"
        if request.propertyId:
            intent = route_intent(request.query)
        
        logger.info(f"Query: {request.query} | Route: {intent}")

        # 2. Execution
        if "BOTH" in intent or "HYBRID" in intent:
            if not request.propertyId:
                return QueryResponse(answer="Hybrid queries require a Property ID for the Analytics part.")
            result = await handle_hybrid_query(request.propertyId, request.query)
            return QueryResponse(answer=result)

        elif "GA4" in intent:
            if not request.propertyId:
                return QueryResponse(answer="Analytics questions require a Property ID.")
            return QueryResponse(answer=agents["ga4"].process_request(request.propertyId, request.query))
            
        else: # SEO
            return QueryResponse(answer=agents["seo"].process_request(request.query))

    except Exception as e:
        logger.error(f"System Error: {e}")
        return QueryResponse(answer=f"System Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)