from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_openai import OpenAI
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os


from .utils import get_weather  # make sure src/utils.py exists with a get_weather(city) function
# ✅ LangSmith Tracing Setup
# ---------------------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGCHAIN_PROJECT", "NeuraDynamics-Demo")
# --- Environment setup ---
load_dotenv()

# --- Initialize models ---
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise EnvironmentError(
        "❌ OPENAI_API_KEY not found! Please set it in your .env file or environment variables."
    )

print("✅ OpenAI key detected:", openai_key[:8] + "..." if openai_key else "Missing")

llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=openai_key
)


embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(":memory:")  # in-memory DB for demo




from langchain_community.vectorstores import Qdrant

# No manual QdrantClient here
vectorstore = None

def init_pdf(pdf_path: str):
    global vectorstore
    loader = PyPDFLoader(pdf_path)
    # loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # In-memory instance (does not persist after restart)
    vectorstore = Qdrant.from_documents(
        chunks,
        embeddings,
        location=":memory:",
        collection_name="pdf_docs"
    )


# --- Graph state definition ---
class State(TypedDict):
    query: str
    decision: str
    data: str
    response: str


def call_llm(prompt: str) -> str:
    """Wrapper to handle both string and AIMessage returns safely."""
    try:
        result = llm.invoke(prompt)
        if hasattr(result, "content"):
            return result.content.strip()
        elif isinstance(result, str):
            return result.strip()
        else:
            return str(result).strip()
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return ""


import time
import re

def safe_llm_call(prompt: str) -> str:
    """Safely call the LLM and always return plain text."""
    try:
        result = llm.invoke(prompt)
        if hasattr(result, "content"):
            text = result.content.strip()
        elif isinstance(result, str):
            text = result.strip()
        else:
            text = str(result).strip()
        return text
    except Exception as e:
        print(f"⚠️ LLM call failed: {e}")
        return ""

def decide_node(state: State) -> dict:
    """LLM-based decision between weather or PDF with retries."""
    query = state["query"].strip()
    print(f"🧠 Incoming query: {query}")

    prompt = f"""
You are a classification model.
Classify the user's intent into exactly ONE of the following categories:

- weather → if the user asks about temperature, rain, forecast, humidity, or other weather info.
- pdf → if the user refers to document, report, summary, or any text content from a file.

Return only the category name.
No explanation, no punctuation.

Query: "{query}"
Answer strictly as one word: weather or pdf.
    """

    decision = safe_llm_call(prompt)
    if not decision:
        print("⚠️ Empty or invalid LLM result (attempt 1/2). Retrying...")
        decision = safe_llm_call(prompt)
    if not decision:
        print("⚠️ No valid LLM output. Using keyword fallback.")
        decision = "weather" if "weather" in query else "pdf"

    print(f"🔍 Final classification decision: {decision}")

    if decision.startswith("weather"):
        return {"decision": "weather"}
    elif decision.startswith("pdf"):
        return {"decision": "pdf"}
    else:
        return {"decision": "pdf"}





def weather_node(state: State) -> dict:
    """Extracts city/state/country using LLM, fetches weather, and formats response."""
    query = state["query"]
    print(f"🌦️ Incoming weather query: {query}")

    # More explicit instruction to force JSON
    extract_prompt = f"""
You are a strict JSON extractor.
From the user query below, extract the **city**, **state**, and **country** if mentioned.

Rules:
- Always return a valid JSON object.
- Do not add explanations, commentary, or text outside the JSON.
- If only the city is provided (e.g. "weather in London"), fill it under 'city'.
- If nothing is found, return exactly:
{{"city": "unknown", "state": "", "country": ""}}

User query:
"{query}"
"""

    raw_output = call_llm(extract_prompt)
    print(f"🧩 Raw location output: {raw_output!r}")

    import json
    location = {"city": "unknown", "state": "", "country": ""}
    try:
        # Try extracting JSON from within the text (if the model added extra words)
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            location = json.loads(match.group(0))
        else:
            location = json.loads(raw_output)
    except Exception as e:
        print(f"⚠️ JSON decode failed: {e}")
        print(f"⚠️ Raw output was: {raw_output}")

    city = location.get("city", "").strip()
    state_name = location.get("state", "").strip()
    country = location.get("country", "").strip()
    location_str = ", ".join(x for x in [city, state_name, country] if x)

    if not city or city.lower() == "unknown":
        return {
            "data": "If you have a specific location in mind, please provide it so I can assist you better."
        }

    print(f"📍 Resolved location: {location_str}")

    # Get weather info
    data = get_weather(location_str)
    print(f"🌤️ Weather API response: {data}")
    if not data or "Error" in data:
        return {"data": f"Could not fetch weather data for {location_str}."}

    # Ask LLM to make the result conversational

    response_prompt = f"""
You are a helpful and friendly weather assistant.

Below is real weather data fetched from a live API.
Your goal is to generate detailed natural, *accurate* weather summary for the user.

Guidelines:
-Start with the location name {location_str}.
- Start the response naturally.
- Mention the temperature in °C, describe all the weather condition, and include humidity or wind speed if available.
- Keep it factual — no placeholders, instructions, or meta-comments.
- Do NOT say things like "complete the task", "as an AI", or any technical phrases.


Weather data:
{data}"""
    summary = call_llm(response_prompt).strip()
    print(f"🗣️ Final summary: {summary}")
    return {"data": summary}









def pdf_node(state: State) -> dict:
    """Retrieve relevant text from PDF and generate an answer."""
    if vectorstore is None:
        return {"data": "PDF not loaded"}

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["query"])  # returns relevant docs

    if not docs:
        return {"data": "No relevant information found in the PDF."}

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful assistant. 
    Use the following PDF excerpts to answer the question concisely and accurately.

    ### Question:
    {state['query']}

    ### Context:
    {context}

    ### Answer:
    """

    # Call the same LLM you used earlier
    answer = call_llm(prompt)
    return {"data": str(answer)}


def llm_node(state: State) -> dict:
    """Generic summarization node — handles both weather and pdf inputs."""
    text = state.get("data") or state.get("response") or ""
    if not text:
        return {"response": "No data found to summarize."}

    prompt = f"Provide a clear, helpful final response based on this data:\n\n{text}"
    response = call_llm(prompt)
    return {"response": str(response).strip()}


from qdrant_client.http import models as qmodels



def embed_store_node(state: State) -> dict:
    """Embed the response or data text and upsert it into Qdrant manually."""
    text = state.get("response") or state.get("data") or ""
    if not text:
        print("⚠️ No response or data to embed.")
        return {}

    embedding = embeddings.embed_query(text)
    point_id = abs(hash(text)) % (10**12)
    payload = {"text": text}

    # Ensure collection exists before inserting
    from qdrant_client.http import models as qmodels
    try:
        qdrant_client.get_collection("responses")
    except Exception:
        qdrant_client.recreate_collection(
            collection_name="responses",
            vectors_config=qmodels.VectorParams(
                size=len(embedding),
                distance=qmodels.Distance.COSINE
            ),
        )

    qdrant_client.upsert(
        collection_name="responses",
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        ],
    )
    print(f"✅ Stored response embedding for: {text[:80]}...")
    return {"response": text}



# --- Graph construction ---
graph = StateGraph(State)

graph.add_node("decide", decide_node)
graph.add_node("weather", weather_node)
graph.add_node("pdf", pdf_node)
graph.add_node("llm", llm_node)
graph.add_node("store", embed_store_node)

graph.set_entry_point("decide")

graph.add_conditional_edges(
    "decide",
    lambda x: x["decision"],
    {"weather": "weather", "pdf": "pdf"}
)



# Weather results already have final text → skip llm
graph.add_edge("weather", "store")

# PDF answers still need LLM summarization
graph.add_edge("pdf", "llm")
graph.add_edge("llm", "store")




compiled_graph = graph.compile()
