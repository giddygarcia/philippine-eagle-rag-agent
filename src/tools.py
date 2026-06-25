"""
RAG-powered tool definitions for the Philippine Eagle conservation chatbot.

Each tool is built around a shared rag_search helper that uses MultiQueryRetriever
for query translation before hitting ChromaDB. Structured outputs are enforced via
Pydantic schemas so the agent always receives validated, typed responses rather than
free-text Markdown.
"""

import logging
from typing import Literal
import os
import requests
from urllib.parse import quote

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

IUCN_SPECIES = "Pithecophaga jefferyi"
_IUCN_CACHE = {}


class ViabilityInterpretation(BaseModel):
    risk_status: Literal["Stable", "Vulnerable", "Endangered", "Critical"] = Field(
        description="Overall population risk status based on the viability score"
    )
    interpretation: str = Field(
        description="2–3 sentence expert interpretation of the breeding pairs, habitat area, "
        "threat level, and viability score specific to the Philippine Eagle"
    )
    next_steps: str = Field(
        description="1–2 sentence specific recommended next steps for conservationists"
    )


class ConservationActionPlan(BaseModel):
    immediate_actions: list[str] = Field(
        description="Actions to take within days or weeks (🚨 Immediate)"
    )
    short_term_actions: list[str] = Field(
        description="Actions for the next 1–6 months (📋 Short-term)"
    )
    medium_term_actions: list[str] = Field(
        description="Actions for 6 months to 2 years (🌱 Medium-term)"
    )
    long_term_actions: list[str] = Field(
        description="Actions for 2+ years, policy and systemic level (🤝 Long-term)"
    )


def make_rag_tools(llm, retriever) -> list:
    """
    Construct and return the three conservation tools bound to the given LLM and retriever.

    Creates a shared MultiQueryRetriever that rewrites each query into multiple
    phrasings before retrieval, improving recall for domain-specific terminology.
    Two structured-output LLM variants are also created here so they are
    instantiated once and reused across all tool calls.

    Args:
        llm: A LangChain-compatible chat model (e.g. ChatOpenAI via OpenRouter).
        retriever: A LangChain retriever backed by the ChromaDB vectorstore.

    Returns:
        A list of three LangChain tools: search_eagle_knowledge,
        estimate_population_viability, and conservation_action_planner.
    """
    multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    def rag_search(query: str) -> tuple[str, str]:
        """
        Run a multi-query RAG search and return content and formatted citations.

        Rewrites the query into multiple phrasings via MultiQueryRetriever,
        deduplicates results, and formats the retrieved chunks into a content
        string and a numbered citation block.

        Args:
            query: The search string to retrieve documents for.

        Returns:
            A tuple of (content, citations). content is the concatenated page
            text of all retrieved documents. citations is a numbered list of
            source references with page numbers and 150-character snippets.
            Both are empty strings if no documents are found.
        """
        docs = multi_retriever.invoke(query)
        if not docs:
            logger.warning("RAG search returned no docs | query=%r", query)
            return "No relevant information found.", ""
        logger.debug("RAG search returned %d docs | query=%r", len(docs), query)

        content = "\n\n".join(d.page_content for d in docs)
        citations = "\n".join(
            f"[{i + 1}] {d.metadata.get('source', 'Knowledge base')}"
            + (f" (page {d.metadata.get('page')})" if d.metadata.get("page") else "")
            + f' — "{d.page_content[:150].strip()}..."'
            for i, d in enumerate(docs)
        )
        return content, citations

    viability_llm = llm.with_structured_output(ViabilityInterpretation)
    action_plan_llm = llm.with_structured_output(ConservationActionPlan)

    @tool
    def search_eagle_knowledge(query: str) -> str:
        """
        Search the Philippine Eagle knowledge base for information.

        Uses multi-query RAG retrieval to find relevant chunks from the
        ChromaDB knowledge base. Returns the retrieved content followed by
        a <<<SOURCES>>> block containing numbered citations.

        Args:
            query: A natural language question about the Philippine Eagle.

        Returns:
            Retrieved content as a string, with citations appended after
            <<<SOURCES>>> if any sources were found.
        """

        logger.info("search_eagle_knowledge called | query=%r", query)
        content, citations = rag_search(query)
        if not citations:
            return content
        return f"{content}\n\n<<<SOURCES>>>\n{citations}"

    VALID_THREAT_LEVELS = {"low", "medium", "high", "critical"}

    @tool
    def estimate_population_viability(
        breeding_pairs: int,
        habitat_hectares: float,
        threat_level: str = "low",
    ) -> str:
        """
        Estimate Philippine Eagle population viability given current conditions.

        Computes a deterministic viability score from breeding pairs, habitat
        area, and threat level, then uses a structured-output LLM call to
        produce an expert interpretation grounded in the knowledge base.

        The agent should infer threat_level from the user's description:
          - low      → isolated threats, protected area, active rangers
          - medium   → occasional poaching or localised logging
          - high     → active deforestation, frequent disturbance
          - critical → imminent habitat destruction, active hunting

        Args:
            breeding_pairs: Estimated number of active nesting pairs.
            habitat_hectares: Estimated remaining suitable forest area in hectares.
            threat_level: Threat classification — must be one of: low, medium, high, critical.
                If not specified by the user, defaults to low. The agent should infer from context when possible and only fall back to low when no threat information is available.

        Returns:
            A Markdown table of parameters, a validated risk status, expert
            interpretation, and recommended next steps. Returns a warning string
            on invalid input or LLM failure.
        """

        logger.info(
            "estimate_population_viability called | pairs=%d, ha=%.1f, threat=%s",
            breeding_pairs,
            habitat_hectares,
            threat_level,
        )
        if breeding_pairs < 0 or habitat_hectares < 0:
            return "⚠️ breeding_pairs and habitat_hectares must be non-negative."
        if threat_level.lower() not in VALID_THREAT_LEVELS:
            return f"⚠️ Invalid threat_level '{threat_level}'. Must be one of: {', '.join(VALID_THREAT_LEVELS)}."

        used_default_threat_level = threat_level == "low"

        threat_level = threat_level.lower()

        context, citations = rag_search(
            f"Philippine Eagle population viability breeding pairs habitat {threat_level} threat"
        )

        # Viability scoring — math stays hardcoded, interpretation comes from LLM
        pair_score = min(breeding_pairs / 50, 1.0)
        habitat_score = min(habitat_hectares / 5000, 1.0)
        threat_penalties = {"low": 0.0, "medium": 0.2, "high": 0.4, "critical": 0.6}
        threat_penalty = threat_penalties.get(threat_level.lower(), 0.3)

        viability_score = round(
            ((pair_score + habitat_score) / 2 - threat_penalty) * 100, 1
        )
        viability_score = max(0.0, min(100.0, viability_score))

        prompt = f"""You are a Philippine Eagle conservation biologist.
A population viability assessment shows:
- Breeding pairs: {breeding_pairs}
- Habitat area: {habitat_hectares} hectares
- Threat level: {threat_level}
- Calculated viability score: {viability_score}/100

Using this knowledge base context:
{context}

Provide a structured expert interpretation specific to the Philippine Eagle."""

        try:
            result: ViabilityInterpretation = viability_llm.invoke(prompt)
        except Exception as e:
            logger.error("Structured output failed for viability tool: %s", e)
            return "⚠️ Could not generate a validated interpretation. Please try again."

        output = f"""## Population Viability Estimate

| Parameter | Value |
|---|---|
| Breeding Pairs | {breeding_pairs} |
| Habitat Area | {habitat_hectares:,.0f} ha |
| Threat Level | {threat_level.title()} |
| Viability Score | {viability_score}/100 |
| Risk Status | **{result.risk_status}** |

**Expert Interpretation:**
{result.interpretation}

**Recommended Next Steps:**
{result.next_steps}
"""
        if citations:
            output += f"\n\n<<<SOURCES>>>\n{citations}"
        if used_default_threat_level:
            output += "\n\n> ℹ️ No threat level was specified — defaulted to **low**. You can be more specific: `low`, `medium`, `high`, or `critical`."
        return output

    @tool
    def conservation_action_planner(
        threat: str,
        location: str = "the Philippines",
    ) -> str:
        """
        Generate a prioritised, practical conservation action plan for a given threat and location.

        Retrieves knowledge-base context relevant to the threat and location,
        then produces a structured four-horizon action plan validated against
        the ConservationActionPlan schema.

        Args:
            threat: The conservation threat to address. Examples: illegal logging,
                poaching, mining, habitat loss, human disturbance.
            location: The geographic area in the Philippines. Examples: Mindanao,
                Leyte, Samar, Luzon, Davao. Defaults to "the Philippines" if
                the user does not specify.

        Returns:
            A Markdown action plan with four sections (immediate, short-term,
            medium-term, long-term), grounded in the knowledge base. Returns a
            warning string on LLM failure.
        """
        logger.info(
            "conservation_action_planner called | threat=%r, location=%r",
            threat,
            location,
        )

        context, citations = rag_search(
            f"conservation strategies Philippine Eagle {threat} {location}"
        )

        prompt = f"""You are a Philippine Eagle conservation expert.
Based on the following knowledge base context, generate a prioritized action plan
for this threat: {threat} in {location}.

Context:
{context}

Be specific to the Philippine Eagle, Philippine law, and the threat described.
Only use information grounded in the context provided."""

        try:
            plan: ConservationActionPlan = action_plan_llm.invoke(prompt)
        except Exception as e:
            logger.error("Structured output failed for action planner tool: %s", e)
            return "⚠️ Could not generate a validated action plan. Please try again."

        def fmt_list(items: list[str]) -> str:
            return "\n".join(f"- {item}" for item in items)

        output = f"""## Conservation Action Plan

**Threat:** {threat.title()}
**Location:** {location.title()}

### 🚨 Immediate Actions
{fmt_list(plan.immediate_actions)}

### 📋 Short-term Actions
{fmt_list(plan.short_term_actions)}

### 🌱 Medium-term Actions
{fmt_list(plan.medium_term_actions)}

### 🤝 Long-term Actions
{fmt_list(plan.long_term_actions)}
"""
        if citations:
            output += f"\n\n<<<SOURCES>>>\n{citations}"
        return output

    @tool
    def get_iucn_status() -> str:
        """
        Fetch IUCN Red List status and assessment summary for the Philippine Eagle
        (Pithecophaga jefferyi) using the IUCN v4 taxon endpoint.

        Returns taxonomic info, Red List category, assessment year, and population trend
        in a formatted Markdown string.

        Requires IUCN_API_TOKEN in environment variables.
        """

        key = "pithecophaga_jefferyi"

        # ✅ return cached result
        if key in _IUCN_CACHE:
            logger.info("IUCN cache hit")
            return _IUCN_CACHE[key]

        logger.info("IUCN cache miss")

        token = os.getenv("IUCN_API_TOKEN")
        if not token:
            return "⚠️ IUCN API token not found."
        # https://api.iucnredlist.org/api/v4/taxa/scientific_name?genus_name=pithecophaga&species_name=jefferyi

        response = requests.get(
            "https://api.iucnredlist.org/api/v4/taxa/scientific_name",
            headers={"Authorization": token.strip()},
            params={
                "genus_name": "pithecophaga",
                "species_name": "jefferyi",
            },
            timeout=20,
        )

        if response.status_code != 200:
            return f"⚠️ IUCN API request failed ({response.status_code})."

        data = response.json()
        record = data["assessments"][0] if "assessments" in data else {}

        result = f"""## Philippine Eagle (IUCN Red List)

    | Field | Value |
    |------|------|
    | Status | **{record.get("red_list_category_code", "Unknown")}** |
    | Year | {record.get("year_published", "N/A")} |

    > Source: IUCN API
    """

        # ✅ store result
        _IUCN_CACHE[key] = result

        return result

    return [
        search_eagle_knowledge,
        estimate_population_viability,
        conservation_action_planner,
        get_iucn_status,
    ]
