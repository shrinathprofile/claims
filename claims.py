import streamlit as st
import os
import tempfile
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
HANDBOOK_PATH = "user_uploaded_handbook.pdf"
OPENROUTER_MODEL = "google/gemma-3-27b-it:free"

# --- Function to Load and Setup BM25 Retriever ---
@st.cache_resource
def setup_retriever(handbook_file):
    """Sets up the BM25 retriever with the uploaded insurance handbook."""
    if handbook_file is None:
        st.error("Please upload a claims handbook PDF to proceed.")
        st.stop()

    st.info("Loading and processing handbook for keyword search...")
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

        # Create BM25 retriever
        retriever = BM25Retriever.from_documents(splits, k=3)

        st.success("Handbook loaded and processed successfully for keyword search!")

        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        return retriever
    except Exception as e:
        st.error(f"Error setting up retriever: {e}")
        st.stop()

# --- Function to Get LLM Response for Claims ---
def get_claim_decision(retriever, claim_text, chat_history, api_key):
    """Uses the LLM and keyword search to get a claim decision based on the handbook and claim details."""
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

    # 2. Retrieve relevant handbook sections using BM25
    relevant_docs = retriever.invoke(search_query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 3. Generate claim decision
    decision_prompt = (
        "You are an AI assistant acting as an insurance claims adjuster for XYZ Insurance Company. "
        "Your task is to analyze the provided medical claim details and the relevant sections from the insurance handbook. "
        "Based *only* on the handbook information and the claim details, determine if the claim should be ACCEPTED or REJECTED. "
        "Provide a clear decision (ACCEPTED/REJECTED) in all caps and a concise, step-by-step reasoning citing the relevant rules from the handbook. "
        "If you cannot find specific coverage rules in the handbook for the exact claim details, state that the claim details are unclear or the handbook lacks explicit coverage. "
        f"Handbook Context:\n{context}\n\n"
        f"Claim Details:\n{claim_text}\n\n"
        "Provide the decision and reasoning."
    )
    try:
        decision_response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": decision_prompt}],
            temperature=0.0
        )
        return decision_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating claim decision: {e}")
        st.stop()

# --- Streamlit Application ---
st.set_page_config(page_title="XYZ Insurance Claims Adjuster", layout="wide")
st.title("👨‍⚖️ XYZ Insurance Claims Adjuster (AI Powered)")
st.caption("Automated Insurance Claim Processing using OpenRouter and Keyword Search")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! Please enter your OpenRouter API key, upload your claims handbook PDF, and a medical bill PDF to determine coverage.")
    ]
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "handbook_uploaded" not in st.session_state:
    st.session_state.handbook_uploaded = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Sidebar for API Key and Handbook Upload
with st.sidebar:
    st.header("Configuration")
    st.session_state.api_key = st.text_input("OpenRouter API Key", type="password", value=st.session_state.api_key)
    
    st.header("Claims Handbook")
    handbook_file = st.file_uploader("Upload Claims Handbook (PDF)", type="pdf", key="handbook_uploader")
    
    if handbook_file and st.session_state.api_key:
        st.session_state.handbook_uploaded = True
        st.success(f"Handbook uploaded: {handbook_file.name}")
        # Initialize BM25 retriever with uploaded handbook
        st.session_state.retriever = setup_retriever(handbook_file)

    if st.button("View Handbook Text", disabled=not st.session_state.handbook_uploaded):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(handbook_file.getvalue())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            full_text = "\n\n".join([doc.page_content for doc in docs])
            st.text_area("Handbook Content", full_text, height=600, disabled=True)
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        except Exception as e:
            st.error(f"Error displaying handbook: {e}")

# Main content area for claim processing
st.header("Submit Your Claim")
if not st.session_state.api_key:
    st.warning("Please enter your OpenRouter API key in the sidebar.")
elif not st.session_state.handbook_uploaded:
    st.warning("Please upload a claims handbook PDF in the sidebar before submitting a claim.")
else:
    uploaded_bill = st.file_uploader("Upload Medical Bill (PDF)", type="pdf", key="bill_uploader")

    claim_text_area = st.empty()
    claim_decision_area = st.empty()

    if uploaded_bill:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_bill.getvalue())
            tmp_file_path = tmp_file.name

        st.info(f"File uploaded: {uploaded_bill.name}")

        try:
            # Extract text from the uploaded bill
            loader = PyPDFLoader(tmp_file_path)
            bill_docs = loader.load()
            bill_content = "\n\n".join([doc.page_content for doc in bill_docs])
            claim_text_area.text_area("Extracted Claim Details (from your bill):", bill_content, height=300, key="bill_content")

            with st.spinner("Analyzing claim... Please wait."):
                # Get claim decision
                response_content = get_claim_decision(st.session_state.retriever, bill_content, st.session_state.chat_history, st.session_state.api_key)

                # Update chat history
                st.session_state.chat_history.append(HumanMessage(content=f"User uploaded a bill. Extracted content: {bill_content[:200]}..."))
                st.session_state.chat_history.append(AIMessage(content=response_content))

                # Display the decision
                st.subheader("Claim Decision:")
                st.markdown(response_content)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.warning("Please ensure the OpenRouter API key is valid and the model is accessible.")
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

st.divider()
st.subheader("AI Adjuster Conversation Log")
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.markdown(f"**You:** {message.content}")
    else:
        st.markdown(f"**AI Adjuster:** {message.content}")
