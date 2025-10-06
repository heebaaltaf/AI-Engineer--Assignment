# AI Pipeline with LangChain, LangGraph, and LangSmith

This project demonstrates a simple AI pipeline using LangChain, LangGraph, and LangSmith for handling weather queries and PDF-based Q&A with RAG.

## Features

- **LangGraph Pipeline**: Agentic workflow with decision-making nodes.
- **Weather Integration**: Fetches real-time weather data from OpenWeatherMap API.
- **PDF Q&A with RAG**: Retrieves and answers questions from uploaded PDF documents.
- **Embeddings and Vector DB**: Uses OpenAI embeddings and Qdrant for storage and retrieval.
- **LLM Processing**: Processes data using OpenAI's GPT models.
- **LangSmith Evaluation**: Traces and evaluates LLM responses.
- **Streamlit UI**: Simple chat interface for interaction.
- **Unit Tests**: Tests for API handling, LLM processing, and retrieval logic.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI-Engineer--Assignment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `.env` and fill in your API keys:
     - `OPENWEATHER_API_KEY`: Get from [OpenWeatherMap](https://openweathermap.org/api)
     - `OPENAI_API_KEY`: Get from [OpenAI](https://platform.openai.com/)
     - `LANGCHAIN_API_KEY`: Get from [LangSmith](https://smith.langchain.com/)
     - `LANGCHAIN_TRACING_V2=true`
     - `LANGCHAIN_PROJECT=neura_dynamics_pipeline`

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Upload a PDF** in the sidebar and start querying.

## Implementation Details

- **Pipeline Flow**:
  - Decision Node: Classifies query as 'weather' or 'pdf'.
  - Weather Node: Calls OpenWeatherMap API.
  - PDF Node: Retrieves relevant chunks from vector store.
  - LLM Node: Generates response using OpenAI.
  - Store Node: Embeds response and stores in Qdrant.

- **RAG Setup**: PDF is loaded, chunked, embedded, and stored in Qdrant on upload.

- **Evaluation**: LangSmith traces all operations for evaluation.

## Testing

Run tests with:
```bash
pytest tests/
```

## Deliverables

- Python code in this repository.
- LangSmith traces for evaluation.
- Test results.
- Streamlit demo.

For LangSmith logs, check your LangSmith dashboard after running queries.
