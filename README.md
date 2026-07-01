# 🦅 Philippine Eagle Conservation Research - an RAG-grounded LLM Agentic Chatbot
**Tech Stack:** Python, LangChain, OpenRouter, Streamlit, ChromaDB, HuggingFace, DeepEval

This repo provides an **agentic Retrieval-Augmented Generation (RAG) LLM research assistant** to help conservationists plan, and for any user to learn more, about the **great Philippine Eagle** - the critically endangered national bird of the Philippines. 

Powered by a **curated vector database** of peer-reviewed research, conservation fact sheets, and a handful of recent news articles. Equipped with **tools** - from citation sourcing, live IUCN status calling, to conservation planning across immediate to long-term timelines. Also includes a small **RAG evaluation pipeline** using the faithfulness metric to assess how well generated responses remain supported by retrieved evidence. Meant as an **accessible AI assistant** for researchers, conservationists, students, or the general public to learn more and take action to help with Philippine Eagle conservation.

<div align="center">
  <img src="visuals/viability-tool-demo-1.png" alt="Conservation Action  Tool Demo 1" width="49%"/>
  <img src="visuals/viability-tool-demo-2.png" alt="Conservation Action  Tool Demo 2" width="49%"/>
</div>

## Features:
- 🧠 Agentic LLM with 4 curated tools:
  - Citation Sourcing
  - Conservation Planning
  - Population Viability Estimator
  - [Live API] Latest IUCN Species Status
- 🔍 **Multi-Query Retrieval-Augmented Generation (RAG)** for improved document recall
- 🤖 **Multi-Model support** with 5 total options:
  - (1 Local option) Supports local LLMs on LM Studio using Gemma 4
  - (4 Cloud options) OpenRouter options of free / paid OpenAI, Google, and NVIDIA models
- 📚 Pre-built **ChromaDB vector database** populated with research publications, conservation fact sheets, and curated news articles
- 🗂️ Semantic search powered by HuggingFace embeddings
- 📖 Grounded, citation-backed responses to reduce hallucinations
- 📊 RAG evaluation pipeline using DeepEval's Faithfulness metric to measure grounding quality
- 💬 Interactive Streamlit web interface with comprehensive visualizations

<div align="center">
  <img src="visuals/demo.png" alt="Philippine Eagle Research Agent Demo" width="80%"/>
</div>

### 🛠️ Tools

Specific conservation science tools were integrated that allow the AI agent to autonomously determine and execute the appropriate functions:

| Tool | Description | Params | Example Prompt |
|---|---|---|---|
| `search_eagle_knowledge` | Searches the Philippine Eagle knowledge base using RAG and returns relevant information with cited sources | `query: str` | "What is the conservation status of the Philippine Eagle?" |
| `conservation_action_planner` | Generates a prioritized conservation action plan for a given threat and location, grounded in knowledge base context. Outputs immediate, short-term, medium-term, and long-term actions | `threat: str, location: str` | "Plan conservation actions for illegal logging in Mindanao" |
| `estimate_population_viability` | Calculates a viability score from population parameters and generates an expert interpretation grounded in the knowledge base. `threat_level` must be one of: `low`, `medium`, `high`, `critical` | `breeding_pairs: int, habitat_hectares: float, threat_level: str` | "There are only 8 breeding pairs left in 1200 hectares with a high threat level — how viable is this population?" |
| `get_iucn_status` | Fetches the **latest IUCN Red List status** and brief status summary for the species **using the IUCN v4 taxon API endpoint**. Returns taxonomic information, Red List category, assessment history, and population trend. | None (fixed species: Pithecophaga jefferyi) | "Get the IUCN status of the Philippine Eagle" / simply "status" |

<div align="center">
  <img src="visuals/planner-tool-demo-1.png" alt="Conservation Action Planner Tool Demo Part 1" width="49%"/>
  <img src="visuals/planner-tool-demo-2.png" alt="Conservation Action Planner Tool Demo Part 2" width="49%"/>
</div>

#### Other features:
- Structured Pydantic Schema Output
- Security with input validation against prompt injection, revealing eagle sanctuary locations, etc.
- Rate limiting, API key management
- Error handling and error messages
- Logs and monitoring

### 🛣️ Feature Roadmap:
- [ ] Load more information to the RAG database
- [ ] Expand usage of IUCN Red List's API to give a more comprehensive species status snapshot > simple species status and date
- [ ] **Integrate more conservationist scientific tools** - Implement more tools that are use-case specific and practical for the day-to-day of conservationists; more science-based like the Population Viability Analysis tool
- [ ] Evaluate all metrics of RAG (Context relevancy, etc., not just faithfulness)
- [ ] **`ingest.py`** - for learning, logging, and viewing purposes the RAG database was built in a notebook `RAG-base.ipynb`. Simple improvement to transfer it to a more typical `ingest.py`
- [ ] Built test cases and conduct unit testing for all tools
- [ ] Further optimize UI and improve aesthetics
- [ ] Possibly launch to the cloud for release / live demo & usage
- [ ] Cleaner RAG - parsed PDFs, webpages vary in exported quality
- [ ] **Contextual Compression Retriever** may yield more relevant chunk retrieval from the RAG 

### Limitations
* ⚠️ Data Limitations
    * Most complete public data, last newsletters and annual reports made publicly available by the Philippine Eagle Foundation, were last published 2023 (Last Annual Report - 2021)
* Limited to mini / lower-capacity models - model selection is currently limited for cost-effective API calling measures. 


## 📦 Libraries & Tools Used

*Most important project files can be found in the `src` folder that contains the streamlit app, system prompts, etc.*

### Tech Stack:
- Python
- Streamlit - UI dashboard used for front-end
- ChromaDB - vector database
- HuggingFace - embeddings
- LM Studio - Local inference for open-source models
- OpenAI Python SDK + OpenRouter API - unified client for OpenAI + OpenRouter LLM calls
- DeepEval - RAG evaluation

### 🔎 Viewing / Installation 

- **Viewing Option:** For general viewing, simply go through this README and the demo images.

- **Full Installation Option:** To run or develop the project locally:

1. Clone the repository:
   ```
   git clone https://github.com/giddygarcia/philippine-eagle-rag-agent.git
   cd philippine-eagle-rag-agent
   ```
2. Install dependencies with `uv`:
    ```
    uv sync
    ```
3. Run the application:
    ```
    uv run streamlit run src/app.py
    ```

#### 📚️ Sources:
- Main Resources: 
  - [Philippine Eagle Foundation](https://www.philippineeaglefoundation.org/philippine-eagle)
  - [BirdLife Species Status (IUCN Red List Assessment Data)](https://datazone.birdlife.org/species/factsheet/philippine-eagle-pithecophaga-jefferyi)
- Publications:
  1. [Priority conservation areas and a global population estimate for the critically endangered Philippine Eagle](https://www.philippineeaglefoundation.org/_files/ugd/d08c94_74d2622c339e4f1499c76d07852a85a7.pdf)
  2. [Philippine Eagle “Sinabadan” Rescued at Mt Tangkulan](https://www.philippineeaglefoundation.org/_files/ugd/d08c94_9e2dfe5081234dfc81ad12c5ab1421f2.pdf)
  3. [Preventing Philippine Eagle hunting: what are we missing?](https://www.threatenedtaxa.org/index.php/JoTT/article/view/2301/3831)
  4. [First Record of a Successful Nesting of South Philippine Hawk-Eagles](https://www.researchgate.net/profile/Tristan-Luap-Senarillos/publication/399615160_First_Record_of_a_Successful_Nesting_of_South_Philippine_Hawk-Eagles_Nisaetus_pinskeri_on_Mindanao_Philippines/links/69612b090f6f9e478e436913/First-Record-of-a-Successful-Nesting-of-South-Philippine-Hawk-Eagles-Nisaetus-pinskeri-on-Mindanao-Philippines.pdf)
  5. [Population Viability Analysis to Inform
  Conservation Planning for the Philippine Eagle](https://www.cpsg.org/sites/default/files/2025-12/Valle%20et%20al.%202025%20_%20Population%20Viability%20Analysis%20to%20Inform%20Conservation%20Planning%20for%20the%20Philippine%20Eagle%20%28Pithecophaga%20jefferyi%29.pdf)
  6. [Genomic analysis reveals recent population
  decline of the critically endangered
  Philippine eagle](https://link.springer.com/article/10.1186/s12864-026-12859-9)
- News articles:
  1. [The Oven of Life](https://www.gmanetwork.com/news/specials/andypenafuerte/284/the-oven-of-life/)
  2. [Saving the endangered Philippine eagle from extinction](https://mb.com.ph/2026/06/10/saving-the-endangered-philippine-eagle-from-extinction)
  3. [Harmonizing Policy and Action Through the Philippine Eagle Population Viability Analysis and Action Plan Updating](https://faps.bmb.gov.ph/faps/2025/09/29/harmonizing-policy-and-action-through-the-philippine-eagle-population-viability-analysis-and-action-plan-updating/)


## ✉️ Author and Contact Information
Developed by: Christine Garcia 

Have questions? Feel free to:
* email me at cavgarcia22@gmail.com 
* connect on [LinkedIn](www.linkedin.com/in/cavgarcia) 
* or [view more projects](https://github.com/giddygarcia) that I enjoyed making