# ⚗️ Chemistry Document Intelligence Platform
 
**Dynamic SDS/TDS-Aware RAG · Azure OpenAI · Azure AI Search · Azure Blob Storage**
 
An enterprise-grade chemistry document intelligence system that accepts PDF uploads at runtime, automatically classifies them as SDS or TDS, chunks and embeds them, and answers natural language questions using agentic tool-calling retrieval grounded in your documents.
 
---
 
## 🏗️ Architecture
 
```
User
│
├── Tab 1: Upload ──→ Blob Storage (raw PDF)
│                └──→ Azure AI Search (chunks + embeddings)
│
├── Tab 2: Chat ───→ Query Classification (SDS/TDS)
│                └──→ LLM Tool Call → Vector Search → Grounded Answer
│
└── Tab 3: Dashboard → Live analytics from Azure AI Search
```
 
### Key Design Principles
 
| Principle | Implementation |
|---|---|
| Dynamic uploads | No preloaded data — everything indexed at runtime |
| SDS/TDS routing | LLM classifies both documents and queries |
| Agentic retrieval | LLM decides when to call `retrieve_chunks` tool |
| Prompt governance | All prompts in `prompts/*.md` — zero hardcoded strings |
| Short-term memory | Session-only, last N turns, never replaces retrieval |
| User isolation | Every chunk tagged with `user_id`, all searches filtered |
| Azure-native | OpenAI + AI Search + Blob Storage only |
 
---
 
## 📁 Project Structure
 
```
chemistry-doc-intelligence/
│
├── app.py                        # Streamlit entry point (tab router only)
├── startup_check.py              # Pre-flight validator before launch
├── requirements.txt
├── .env                          # Azure credentials (never commit)
├── .streamlit/
│   └── config.toml               # Streamlit server + theme config
│
├── prompts/                      # Externalized prompt governance
│   ├── system_prompt.md
│   ├── document_classifier.md
│   ├── query_classifier.md
│   └── answer_generator.md
│
├── config/
│   └── settings.py               # Single source of all env config
│
├── services/                     # Azure service wrappers (stateless)
│   ├── openai_service.py         # LLM calls + embeddings
│   ├── blob_service.py           # Blob upload/download/list
│   └── search_service.py         # Index management + vector search
│
├── core/                         # Business logic
│   ├── document_processor.py     # PDF → chunks → classify → embed
│   ├── memory_manager.py         # Short-term session memory
│   ├── tool_handler.py           # retrieve_chunks tool schema + executor
│   └── query_engine.py           # Agentic RAG loop
│
├── ui/                           # Streamlit tab modules (rendering only)
│   ├── tab_upload.py             # Tab 1
│   ├── tab_chat.py               # Tab 2
│   └── tab_dashboard.py          # Tab 3
│
└── utils/
    ├── prompt_loader.py          # Loads .md files with LRU cache
    ├── chunker.py                # Token-accurate text chunking (tiktoken)
    └── helpers.py                # IDs, timestamps, user_id, sanitization
```
 
---
 
## 🗄️ Azure AI Search Index Schema
 
All chunks are indexed with these exact field names:
 
| Field | Type | Description |
|---|---|---|
| `id` | String (key) | Deterministic SHA-256 chunk ID |
| `content` | String (searchable) | Chunk text |
| `section` | String (filterable) | Section heading detected from document |
| `type` | String (filterable) | `"sds"` or `"tds"` |
| `source` | String (filterable) | Original filename |
| `contentVector` | Collection(Single) | 3072-dim HNSW embedding |
| `user_id` | String (filterable) | Session user — enforces isolation |
| `upload_timestamp` | String (sortable) | ISO 8601 upload time |
 
---
 
## 🚀 Setup & Launch
 
### 1. Clone and install
 
```bash
git clone <repo-url>
cd chemistry-doc-intelligence
pip install -r requirements.txt
```
 
### 2. Configure Azure credentials
 
Copy `.env` and fill in all placeholders:
 
```bash
cp .env .env.local   # keep .env as template
```
 
```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_EMBEDDING_DIMENSIONS=3072
 
AZURE_SEARCH_ENDPOINT=https://<your-search>.search.windows.net
AZURE_SEARCH_API_KEY=<your-admin-key>
AZURE_SEARCH_INDEX_NAME=chemistry-docs-index
 
AZURE_BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
AZURE_BLOB_CONTAINER_NAME=chemistry-docs
```
 
### 3. Run pre-flight check
 
```bash
python startup_check.py
```
 
All 7 checks must pass before launching.
 
### 4. Launch
 
```bash
streamlit run app.py
```
 
---
 
## 🧪 Running Tests
 
Each phase has its own isolated test file. Run them in order:
 
```bash
python test_phase1.py   # Utils, config, chunker, prompts
python test_phase2.py   # Azure service wrappers (mocked)
python test_phase3.py   # Core business logic (mocked)
python test_phase4.py   # UI wiring and structure (AST-based)
python test_phase5.py   # End-to-end integration (mocked)
```
 
All tests run **without real Azure credentials** — Azure SDK calls are mocked via `unittest.mock`.
 
---
 
## 🔄 End-to-End Flows
 
### Upload Flow (Tab 1)
 
```
PDF uploaded by user
    → blob_service:       store raw file in Azure Blob Storage
    → document_processor: extract text (pypdf)
    → openai_service:     classify as SDS or TDS
    → chunker:            split into token-accurate overlapping chunks
    → openai_service:     generate 3072-dim embedding per chunk
    → search_service:     index chunks with full metadata
```
 
### Query Flow (Tab 2)
 
```
User question
    → memory_manager:    load last N conversation turns
    → openai_service:    classify query → "sds" or "tds"
    → query_engine:      build messages[] = system + memory + question
    → chat_completion:   LLM call with retrieve_chunks tool attached
    → tool_handler:      LLM calls tool → embed query → vector_search
    → search_service:    HNSW search filtered by type + user_id
    → chat_completion:   LLM generates grounded answer from chunks
    → memory_manager:    store turn (question + answer)
```
 
### Dashboard Flow (Tab 3)
 
```
User opens tab / clicks Refresh
    → search_service: count all chunks (user-scoped)
    → search_service: count SDS chunks
    → search_service: count TDS chunks
    → search_service: fetch recent uploads (deduplicated by source)
    → UI renders: KPI metrics + bar chart + uploads table
```
 
---
 
## 🔐 Security Model
 
- Every indexed chunk stores the uploader's `user_id`
- All retrieval filters: `type eq '<type>' and user_id eq '<id>'`
- `dispatch_tool_call()` always uses the **server-side** `user_id`, ignoring any `user_id` the LLM passes in tool arguments (prompt injection prevention)
- Session memory is isolated per browser session and cleared on refresh
- Blob Storage paths are prefixed with `user_id/` for folder-level isolation
 
---
 
## ⚙️ Configuration Reference
 
All tuneable settings in `.env`:
 
| Variable | Default | Description |
|---|---|---|
| `MAX_MEMORY_TURNS` | `5` | Conversation turns kept in session memory |
| `TOP_K_CHUNKS` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `500` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between consecutive chunks |
| `APP_ENV` | `development` | Environment label |
 
---
 
## 📝 Prompt Governance
 
All LLM instructions live in `prompts/*.md`. To update a prompt:
 
1. Edit the relevant `.md` file
2. Call `reload_prompt("filename.md")` or restart the app
3. No code changes required
 
| File | Purpose |
|---|---|
| `system_prompt.md` | Assistant persona and rules |
| `document_classifier.md` | SDS vs TDS classification at upload |
| `query_classifier.md` | SDS vs TDS routing at query time |
| `answer_generator.md` | Grounded answer generation rules |