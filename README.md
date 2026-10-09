# AI Knowledge Assistant

An AI-powered knowledge assistant that allows users to upload PDF documents and ask questions about their content.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and uses an LLM to generate answers. It also supports conversational memory, document management, tool calling, structured responses, and a web interface.

## Key Features

- PDF document upload and ingestion
- Document chunking and vector embeddings
- Chroma vector database for persistent storage
- Semantic document retrieval
- Retrieval-Augmented Generation (RAG)
- Conversational chat with Redis-based memory
- AI agent with tools
- Mathematical calculator tool
- Structured AI responses with sources
- Document change detection using SHA-256 hashing
- FastAPI backend
- Streamlit frontend
- Environment-based configuration
- Basic protection against prompt injection
- Request timing and error logging

## Architecture

```text
                    ┌──────────────────┐
                    │    Streamlit UI  │
                    └────────┬─────────┘
                             │
                             │ HTTP
                             ▼
                    ┌──────────────────┐
                    │   FastAPI API    │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
        ┌─────────────────┐      ┌─────────────────┐
        │ LangChain Agent │      │      Redis      │
        │                 │      │ Conversation    │
        │  ┌───────────┐  │      │    Memory       │
        │  │ GPT-4o-mini│ │      └─────────────────┘
        │  └─────┬─────┘  │
        │        │        │
        │   ┌────┴─────┐  │
        │   │  Tools   │  │
        │   ├──────────┤  │
        │   │ Search   │  │
        │   │ Calculator│ │
        │   └────┬─────┘  │
        └─────────┼────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Chroma Vector DB  │
        │                   │
        │ Embeddings +      │
        │ Document Chunks   │
        └───────────────────┘


PDF Upload Flow
────────────────────────────────────────────────────

PDF
 │
 ▼
PyPDFLoader
 │
 ▼
Document Chunks
 │
 ▼
OpenAI Embeddings
 │
 ▼
Chroma Vector DB


Question Flow
────────────────────────────────────────────────────

User Question
 │
 ▼
LangChain Agent
 │
 ├── Calculator Tool ──────► Mathematical calculation
 │
 └── Search Tool
          │
          ▼
     Chroma Retrieval
          │
          ▼
     Relevant Chunks
          │
          ▼
       GPT-4o-mini
          │
          ▼
   Structured Response
    ┌────────┴────────┐
    │                 │
  Answer           Sources
```

## Technology Stack

| Component            | Technology                     |
| -------------------- | ------------------------------ |
| Programming Language | Python                         |
| LLM                  | OpenAI GPT-4o-mini             |
| Framework            | LangChain                      |
| Vector Database      | Chroma                         |
| Embeddings           | OpenAI text-embedding-3-small  |
| Backend              | FastAPI                        |
| Frontend             | Streamlit                      |
| Conversation Memory  | Redis                          |
| Document Processing  | PyPDFLoader                    |
| Text Splitting       | RecursiveCharacterTextSplitter |
| Structured Output    | Pydantic                       |

## Project Structure

```text
langchain-knowledge-assistant/
│
├── main.py                 # LangChain, RAG, tools, agent and ingestion logic
├── api.py                  # FastAPI backend and API endpoints
├── ui.py                   # Streamlit frontend
│
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment configuration
├── .gitignore              # Files excluded from Git
├── README.md               # Project documentation
│
├── documents/              # Local uploaded PDFs - not committed
└── chroma_db/              # Local vector database - not committed
```

## Setup & Installation

### 1. Clone the Repository

```bash
git clone <[your-github-repository-url](https://github.com/GnaneshKumar731/langchain-knowledge-assistant)>
cd langchain-knowledge-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
REDIS_HOST=localhost
REDIS_PORT=6379
API_BASE_URL=http://127.0.0.1:8000
```

Do not commit the `.env` file to GitHub.

### 5. Start Redis

Make sure your Redis-compatible server is running locally.

### 6. Run the FastAPI Backend

```bash
uvicorn api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 7. Run the Streamlit Frontend

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run ui.py
```

## API Endpoints

### `GET /`

Checks whether the FastAPI backend is running.

**Response:**

```json
{
  "message": "AI Knowledge Assistant API is running"
}
```

### `POST /chat`

Sends a question to the AI assistant.

**Request:**

```json
{
  "session_id": "unique-conversation-id",
  "message": "What is LangChain?"
}
```

**Response:**

```json
{
  "answer": "LangChain is a framework for developing applications powered by language models.",
  "sources": ["langchain_notes.pdf, Page 1"]
}
```

### `POST /upload`

Uploads a PDF document and adds its content to the knowledge base.

**Request:**

Multipart form-data with a PDF file.

**Response:**

```json
{
  "message": "example.pdf uploaded and ingested successfully"
}
```

### API Documentation

When the FastAPI server is running, interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Example Usage

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Upload a PDF document through the Streamlit interface.
4. The document is loaded, split into chunks, converted into embeddings, and stored in Chroma.
5. Ask a question about the uploaded document.
6. The LangChain agent decides whether to search the knowledge base or use the calculator tool.
7. Relevant document content is retrieved and provided to the LLM.
8. The assistant returns a structured response containing the answer and source information.

### Example Questions

```text
What is LangChain?
Explain the difference between embeddings and a vector database.
What does this document say about Python?
What is 25 * 48?
```

The assistant maintains conversation history using Redis, allowing follow-up questions to use the context of the previous conversation.

## Future Improvements

- Authentication and user authorization
- Restrict CORS origins for production deployment
- Cloud-based vector database and document storage
- Streaming responses from the LLM
- Better observability and centralized logging
- Automated evaluation of RAG responses
- Support for additional document formats
- Production deployment using Docker
- Improved document metadata and filtering
- Background document ingestion for large files
