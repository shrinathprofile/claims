import streamlit as st
import os
import tempfile
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np

# --- Configuration ---
HANDBOOK_PATH = "user_uploaded_handbook.pdf"
OPENROUTER_MODEL = "microsoft/phi-4-reasoning:free"
EMBEDDING_MODEL = "text-embedding-ada-002"

# --- Custom Embedding Function for nomic-embed-text via OpenRouter ---
class NomicEmbeddings:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    def embed_documents(self, texts):
        try:
            embeddings = []
            for text in texts:
                response = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            st.warning(f"Please ensure the embedding model `{EMBEDDING_MODEL}` is available via OpenRouter.")
            st.stop()

    def embed_query(self, text):
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating query embedding: {e}")
            st.stop()

# --- Function to Load and Embed Handbook into FAISS ---
@st.cache_resource
def setup_vector_store(handbook_file, api_key):
    """Sets up the FAISS vector store with the uploaded insurance handbook."""
    if handbook_file is None:
        st.error("Please upload a claims handbook PDF to proceed.")
        st.stop()
    if not api_key:
        st.error("Please enter a valid OpenRouter API key.")
        st.stop()

    st.info("Loading and embedding handbook into FAISS...")
    try:
        # Save uploaded handbook to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(handbook_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load and process the handbook
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Embed documents using OpenRouter
        embeddings = NomicEmbeddings(api_key)
        vectorstore = FAISS.from_documents(splits, embeddings)

        st.success("Handbook loaded and embedded successfully into FAISS!")

        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        return vectorstore
    except Exception as e:
        st.error(f"Error setting up vector store: {e}")
        st.warning("Please ensure the OpenRouter API key is valid and the embedding model is accessible.")
        st.stop()

# --- Function to Get LLM Response for Claims ---
def get_claim_decision(vectorstore, claim_text, chat_history, api_key):
    """Uses the LLM and RAG to get a claim decision based on the handbook and claim details."""
    if not api_key:
        st.error("Please enter a valid OpenRouter API key.")
        st.stop()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    # 1. Formulate a search query based on chat history and claim
    history_text = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in chat_history])
    query_prompt = (
        f"Given the conversation history:\n{history_text}\n\n"
        f"Claim details:\n{claim_text}\n\n"
        "Generate a concise search query to find relevant sections in the insurance handbook."
    )
    try:
        query_response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": query_prompt}],
            temperature=0.0
        )
        search_query = query_response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating search query: {e}")
        st.stop()

    # 2. Retrieve relevant handbook sections
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(search_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 3. Generate claim decision
    decision_prompt = (
        "You are an AI assistant acting as an insurance claims adjuster for XYZ Insurance Company. "
        "Your task is to analyze the provided medical claim details and the relevant sections from the insurance handbook. "
        "Based *only* on the handbook information and the claim details, determine if the claim should be ACCEPTED or REJECTED. "
        "Provide a clear decision (ACCEPTED/REJECTED) in all caps and a concise, step-by-step reasoning citing the relevant rules from the handbook. "
        "If you cannot find specific coverage rules in the handbook for the exact claim details, state
