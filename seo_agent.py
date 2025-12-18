import pandas as pd
import io
import json
import requests
from typing import Dict, Any, List
from llm_client import LiteLLMClient

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

SHEET_ID = "1zzf4ax_H2WiTBVrJigGjF2Q3Yz-qy2qMCbAMKvl6VEE"
GID = "1438203274"

SHEET_URLS = [
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}",
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}",
]

# Semantic field mapping (NL → dataframe)
FIELD_MAP = {
    "https": "uses_https",
    "indexability": "is_indexable",
    "title_length": "Title_1_Length",
    "meta_description_length": "Meta_Description_1_Length",
    "status_code": "Status_Code",
}

# ------------------------------------------------------------------
# SEO AGENT
# ------------------------------------------------------------------

class SEOAgent:
    def __init__(self):
        self.llm = LiteLLMClient()
        self.df = self._load_sheet()
        self.df = self._normalize_schema(self.df)
        self.df = self._enrich_features(self.df)

    # ------------------------------------------------------------------
    # DATA INGESTION
    # ------------------------------------------------------------------

    def _load_sheet(self) -> pd.DataFrame:
        headers = {"User-Agent": "Mozilla/5.0"}
        for url in SHEET_URLS:
            try:
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                return pd.read_csv(io.StringIO(r.text), on_bad_lines="skip")
            except Exception:
                continue
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # SCHEMA NORMALIZATION
    # ------------------------------------------------------------------

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df.columns = (
            df.columns.str.strip()
            .str.replace(" ", "_")
            .str.replace("(", "")
            .str.replace(")", "")
            .str.replace("-", "_")
        )

        numeric_cols = [
            "Title_1_Length",
            "Meta_Description_1_Length",
            "Word_Count",
            "Status_Code",
            "Size_Bytes",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "")
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0)
                )

        return df

    # ------------------------------------------------------------------
    # SEO FEATURE ENGINEERING (SAFE, TOTAL)
    # ------------------------------------------------------------------

    def _enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        # Defaults (CRITICAL for robustness)
        df["uses_https"] = False
        df["is_indexable"] = False
        df["title_too_long"] = False
        df["title_missing"] = False
        df["meta_missing"] = False
        df["meta_too_long"] = False
        df["is_error"] = False

        if "Address" in df.columns:
            df["uses_https"] = df["Address"].astype(str).str.startswith("https")

        if "Indexability" in df.columns:
            df["is_indexable"] = df["Indexability"].str.contains(
                "Indexable", case=False, na=False
            )

        if "Title_1_Length" in df.columns:
            df["title_too_long"] = df["Title_1_Length"] > 60
            df["title_missing"] = df["Title_1_Length"] == 0

        if "Meta_Description_1_Length" in df.columns:
            df["meta_missing"] = df["Meta_Description_1_Length"] == 0
            df["meta_too_long"] = df["Meta_Description_1_Length"] > 160

        if "Status_Code" in df.columns:
            df["is_error"] = df["Status_Code"] >= 400

        return df

    # ------------------------------------------------------------------
    # QUERY PLANNER (LLM → JSON)
    # ------------------------------------------------------------------

    def _plan_query(self, user_query: str) -> Dict[str, Any]:
        system_prompt = """
    You are an SEO query planner.

    Convert the user question into a structured JSON analysis plan.

    Allowed operations:
    - filter
    - groupby
    - metric

    Allowed fields:
    - https
    - indexability
    - title_length
    - meta_description_length
    - status_code

    Rules:
    - Return ONLY valid JSON
    - Do NOT explain
    - Do NOT include markdown
    - Do NOT include backticks
    """

        user_prompt = f"Question: {user_query}"

        for attempt in range(3):
            plan_text = self.llm.generate_completion([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])

            if not plan_text or not plan_text.strip():
                continue

            cleaned = (
                plan_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Ask model to correct itself
                user_prompt = f"""
    The previous response was not valid JSON.

    Return ONLY corrected JSON for this question:
    {user_query}
    """
                continue

        raise ValueError("Failed to generate a valid query plan.")


    # ------------------------------------------------------------------
    # EXECUTION ENGINE (NO LLM)
    # ------------------------------------------------------------------


    def _validate_and_normalize_plan(self, plan, user_query):
        if not isinstance(plan, dict):
            plan = {}

        # ----------------------------
        # Infer operation
        # ----------------------------
        if "operation" not in plan:
            if "conditions" in plan:
                plan["operation"] = "filter"
            elif "field" in plan and "n" in plan:
                plan["operation"] = "top_n"
            elif "field" in plan:
                plan["operation"] = "metric"
            else:
                q = user_query.lower()
                if "top" in q:
                    plan["operation"] = "top_n"
                elif "group" in q:
                    plan["operation"] = "groupby"
                else:
                    plan["operation"] = "filter"

        # ----------------------------
        # FILTER
        # ----------------------------
        if plan["operation"] == "filter":
            if not plan.get("conditions"):
                recovered = self._recover_conditions_from_query(user_query)
                plan["conditions"] = recovered["conditions"]
                plan["logic"] = recovered["logic"]

            plan.setdefault("logic", "and")

            if not plan["conditions"]:
                raise ValueError("Filter inferred but no conditions found.")

        # ----------------------------
        # TOP-N
        # ----------------------------
        if plan["operation"] == "top_n":
            if "field" not in plan:
                recovered_field = self._recover_top_n_field(user_query)
                if recovered_field:
                    plan["field"] = recovered_field
                else:
                    raise ValueError("Top-N requires a ranking field and none could be inferred.")

            plan.setdefault("n", 10)


        # ----------------------------
        # GROUPBY / METRIC
        # ----------------------------
        if plan["operation"] in {"groupby", "metric"}:
            if "field" not in plan:
                raise ValueError(f"{plan['operation']} requires 'field'")

        return plan



    def _execute_plan(self, plan):
        df = self.df.copy()

        # ----------------------------
        # FILTER with AND / OR
        # ----------------------------
        if plan["operation"] == "filter":
            masks = []

            for cond in plan["conditions"]:
                col = FIELD_MAP.get(cond["field"])
                if col not in df.columns:
                    continue

                if cond["op"] == "=":
                    masks.append(df[col] == cond["value"])
                elif cond["op"] == ">":
                    masks.append(df[col] > cond["value"])
                elif cond["op"] == "<":
                    masks.append(df[col] < cond["value"])

            if not masks:
                return df[["Address"]]

            if plan.get("logic", "and") == "or":
                final_mask = masks[0]
                for m in masks[1:]:
                    final_mask |= m
            else:
                final_mask = masks[0]
                for m in masks[1:]:
                    final_mask &= m

            return df[final_mask][["Address"]]

        # ----------------------------
        # TOP-N
        # ----------------------------
        if plan["operation"] == "top_n":
            col = FIELD_MAP.get(plan["field"])
            n = plan.get("n", 10)

            if col not in df.columns:
                return {}

            top = df.sort_values(col, ascending=False).head(n)
            return top[["Address", col]]

        # ----------------------------
        # GROUPBY
        # ----------------------------
        if plan["operation"] == "groupby":
            col = FIELD_MAP.get(plan["field"])
            return df.groupby(col).size().to_dict()

        # ----------------------------
        # METRIC
        # ----------------------------
        if plan["operation"] == "metric":
            col = FIELD_MAP.get(plan["field"])
            pct = round(df[col].mean() * 100, 2)
            return {"percentage": pct}

        return {}

    def _recover_top_n_field(self, user_query: str) -> str | None:
        q = user_query.lower()

        if "title" in q:
            return "title_length"

        if "meta" in q:
            return "meta_description_length"

        if "status" in q or "error" in q:
            return "status_code"

        return None


    # ------------------------------------------------------------------
    # SEO HEALTH HEURISTIC
    # ------------------------------------------------------------------

    def _seo_health(self, indexable_pct: float) -> str:
        if indexable_pct >= 90:
            return "good"
        if indexable_pct >= 70:
            return "average"
        return "poor"

    # ------------------------------------------------------------------
    # LLM REASONING / EXPLANATION
    # ------------------------------------------------------------------

    def _explain(self, user_query: str, result: Any) -> str:
        health = None
        if isinstance(result, dict) and "percentage" in result:
            health = self._seo_health(result["percentage"])

        system_prompt = """
You are a senior SEO analyst.

Explain the result clearly.
If percentages are provided, assess technical SEO health.
Mention risks and practical implications.
Be concise and professional.
"""

        user_prompt = f"""
User Question:
{user_query}

Computed Result:
{result}

SEO Health (if applicable):
{health}
"""

        return self.llm.generate_completion([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    # ------------------------------------------------------------------
    # PUBLIC ENTRYPOINT
    # ------------------------------------------------------------------

    def _recover_conditions_from_query(self, user_query: str) -> Dict[str, Any]:
        q = user_query.lower()

        conditions = []
        logic = "and"

        if " or " in q:
            logic = "or"

        if "https" in q and ("do not" in q or "not use" in q or "without" in q):
            conditions.append({
                "field": "https",
                "op": "=",
                "value": False
            })

        if "title" in q and "60" in q:
            conditions.append({
                "field": "title_length",
                "op": ">",
                "value": 60
            })

        if "meta" in q and ("missing" in q or "0" in q):
            conditions.append({
                "field": "meta_description_length",
                "op": "=",
                "value": 0
            })

        if "non-indexable" in q or "not indexable" in q:
            conditions.append({
                "field": "indexability",
                "op": "=",
                "value": False
            })

        return {
            "conditions": conditions,
            "logic": logic
        }



    def process_request(self, user_query: str) -> str:
        user_query = " ".join(user_query.split())

        if self.df.empty:
            return "SEO data could not be loaded."

        try:
            plan = self._plan_query(user_query)
            plan = self._validate_and_normalize_plan(plan, user_query)
            result = self._execute_plan(plan)

            if isinstance(result, pd.DataFrame):
                if result.empty:
                    return "No URLs matched the specified conditions."
                return result.head(50).to_string(index=False)

            return self._explain(user_query, result)

        except Exception as e:
            return f"System Error: {str(e)}"

