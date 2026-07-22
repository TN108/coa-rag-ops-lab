import json
import logging
import time
from typing import Any

import requests

from app.config import settings


logger = logging.getLogger(__name__)


class PlannerService:
    """
    Adaptive RAG planner.

    Uses llama3.2:1b to classify question intent
    and select controlled retrieval configuration.
    """

    ALLOWED_QUERY_TYPES = {
        "factoid",
        "explanation",
        "broad",
    }

    ALLOWED_TOP_K = {
        3,
        5,
        8,
    }

    ALLOWED_THRESHOLDS = {
        0.05,
        0.10,
        0.20,
    }

    ALLOWED_CHUNKING_METHODS = {
        "fixed",
        "semantic",
    }

    ALLOWED_NEIGHBOR_WINDOWS = {
        0,
        1,
        2,
    }

    FALLBACK_CONFIGURATION = {
        "query_type": "factoid",
        "top_k": 3,
        "min_retrieval_score": 0.20,
        "chunking_method": "semantic",
        "neighbor_window": 0,
    }

    def __init__(self):
        self.model = settings.PLANNER_MODEL
        self.base_url = settings.OLLAMA_BASE_URL

    
    def plan(
        self,
        question: str,
    ) -> dict[str, Any]:

        start_time = time.perf_counter()

        fallback_used = False
        error = None

        try:

            response = self._call_planner(
                question
            )

            decision = self._extract_json(
                response
            )

            validated = self._validate_decision(
                decision
            )

        except Exception as exception:

            logger.warning(
                "Planner failed: %s",
                exception,
            )

            validated = (
                self.FALLBACK_CONFIGURATION.copy()
            )

            fallback_used = True

            error = str(exception)


        planner_ms = (
            time.perf_counter()
            -
            start_time
        ) * 1000


        return {

            "planner_model": self.model,

            "planner_decision": validated,

            "planner_ms": round(
                planner_ms,
                2,
            ),

            "fallback_used": fallback_used,

            "error": error,
        }


    def _call_planner(
        self,
        question: str,
    ) -> str:

        prompt = self._build_prompt(
            question
        )


        response = requests.post(

            f"{self.base_url}/api/generate",

            json={

                "model": self.model,

                "prompt": prompt,

                "stream": False,

                "format": "json",

                "keep_alive": "10m",

                "options": {

                    "temperature": 0,

                    "top_p": 0.1,

                    "num_predict": 80,
                },
            },

            timeout=20,
        )


        response.raise_for_status()


        data = response.json()


        output = (
            data.get("response")
            or ""
        ).strip()


        if not output:

            raise ValueError(
                "Planner returned empty response."
            )


        return output



    def _build_prompt(
        self,
        question: str,
    ) -> str:


        return f"""

You are an Adaptive RAG retrieval planner.

Your task is only to classify the user's
question intent.

Do not answer the question.

Do not use document knowledge.

Return ONLY JSON.

Allowed categories:


FACTOID:

Use when the user asks for:

- definition
- name
- date
- specific fact
- short lookup


Return:

{{
"query_type":"factoid",
"top_k":3,
"min_retrieval_score":0.20,
"chunking_method":"semantic",
"neighbor_window":0
}}



EXPLANATION:

Use when the user asks:

- how something works
- why something happens
- architecture
- workflow
- process


Return:

{{
"query_type":"explanation",
"top_k":5,
"min_retrieval_score":0.10,
"chunking_method":"semantic",
"neighbor_window":1
}}



BROAD:

Use when the user asks:

- summary
- overview
- comparison
- multiple sections


Return:

{{
"query_type":"broad",
"top_k":8,
"min_retrieval_score":0.05,
"chunking_method":"semantic",
"neighbor_window":2
}}



Rules:

- Never use factoid for explain/how/why questions.
- Never use broad for simple definitions.
- Use only the allowed values.

Question:

{question}


Return JSON only.

""".strip()



    def _extract_json(
        self,
        text: str,
    ) -> dict[str, Any]:

        try:

            result = json.loads(
                text
            )


        except json.JSONDecodeError:


            start = text.find("{")

            end = text.rfind("}")


            if start == -1 or end == -1:

                raise ValueError(
                    "No JSON object found."
                )


            result = json.loads(
                text[
                    start:end + 1
                ]
            )


        if not isinstance(
            result,
            dict,
        ):

            raise ValueError(
                "Planner output must be JSON object."
            )


        return result



    def _validate_decision(
        self,
        decision: dict[str, Any],
    ) -> dict[str, Any]:


        required_fields = {

            "query_type",

            "top_k",

            "min_retrieval_score",

            "chunking_method",

            "neighbor_window",
        }


        missing = (
            required_fields
            -
            decision.keys()
        )


        if missing:

            raise ValueError(
                f"Missing fields: {missing}"
            )


        query_type = (

            str(
                decision["query_type"]
            )

            .lower()

            .strip()
        )


        chunking_method = (

            str(
                decision["chunking_method"]
            )

            .lower()

            .strip()
        )


        try:

            top_k = int(
                decision["top_k"]
            )


            threshold = round(

                float(
                    decision[
                        "min_retrieval_score"
                    ]
                ),

                2,
            )


            neighbor_window = int(

                decision[
                    "neighbor_window"
                ]
            )


        except Exception as error:

            raise ValueError(
                "Invalid planner numeric values."
            ) from error



        if query_type not in self.ALLOWED_QUERY_TYPES:

            raise ValueError(
                "Invalid query_type."
            )


        if top_k not in self.ALLOWED_TOP_K:

            raise ValueError(
                "Invalid top_k."
            )


        if threshold not in self.ALLOWED_THRESHOLDS:

            raise ValueError(
                "Invalid threshold."
            )


        if chunking_method not in self.ALLOWED_CHUNKING_METHODS:

            raise ValueError(
                "Invalid chunking method."
            )


        if neighbor_window not in self.ALLOWED_NEIGHBOR_WINDOWS:

            raise ValueError(
                "Invalid neighbor window."
            )



        return {

            "query_type": query_type,

            "top_k": top_k,

            "min_retrieval_score": threshold,

            "chunking_method": chunking_method,

            "neighbor_window": neighbor_window,
        }



planner_service = PlannerService()