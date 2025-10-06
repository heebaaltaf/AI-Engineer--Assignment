# import streamlit as st
# from src.pipeline import compiled_graph, init_pdf
# import tempfile
# import os

# st.title("AI Pipeline Demo")

# st.sidebar.header("Upload PDF")
# uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# if uploaded_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         pdf_path = tmp_file.name
#     init_pdf(pdf_path)
#     st.sidebar.success("PDF loaded successfully!")
#     os.unlink(pdf_path)  # Clean up

# st.header("Chat Interface")
# query = st.text_input("Enter your query (e.g., 'weather in London' or 'What is the content about?'):")

# if st.button("Submit"):
#     if not query:
#         st.error("Please enter a query.")
#     else:
#         with st.spinner("Processing..."):
#             result = compiled_graph.invoke({"query": query})
#         st.write("**Response:**")
#         st.write(result.get('response', 'No response generated.'))


import streamlit as st
from src.pipeline import compiled_graph, init_pdf
import tempfile
import os

st.title("🧠 LangGraph + LangSmith AI Pipeline Demo")

st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
    init_pdf(pdf_path)
    st.sidebar.success("✅ PDF loaded successfully!")
    os.unlink(pdf_path)

st.header("💬 Chat Interface")
query = st.text_input("Enter your query (e.g., 'weather in London' or 'summarize PDF content'): ")

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            result = compiled_graph.invoke({"query": query})
        st.success("✅ Response:")
        st.write(result.get("response", "No response generated."))
