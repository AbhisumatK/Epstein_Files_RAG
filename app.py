import os
os.environ["USE_TORCH"] = "1" # Force PyTorch, disable TensorFlow
import streamlit as st
import os
from dotenv import load_dotenv
from rag_chain import get_rag_chain

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Epstein Files RAG Explorer",
    page_icon="🔍",
    layout="wide"
)

# Sidebar
st.sidebar.title("Configuration")
provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["OLLAMA", "GROQ", "OPENROUTER"],
    index=0
)

# API Keys and Models
api_key = None
model_name = None

if provider == "GROQ":
    api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    model_name = st.sidebar.text_input("Groq Model", value="llama-3.3-70b-versatile")
elif provider == "OPENROUTER":
    api_key = st.sidebar.text_input("OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
    model_name = st.sidebar.text_input("OpenRouter Model", value="meta-llama/llama-3-8b-instruct")
else:
    model_name = st.sidebar.text_input("Ollama Model", value="llama3")

# Set environment variables for the current session
if api_key:
    if provider == "GROQ":
        os.environ["GROQ_API_KEY"] = api_key
    elif provider == "OPENROUTER":
        os.environ["OPENROUTER_API_KEY"] = api_key

# Main UI
st.title("🔍 Epstein Files RAG Explorer")
st.markdown("Query the unsealed Epstein court documents using open-source RAG.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about the Epstein documents..."):
    # Clear session if provider/model changed? (Simplified here)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                # Initialize chain (ideally cached, but simplified here)
                chain = get_rag_chain(provider=provider, model_name=model_name)
                
                # Run query
                response = chain.invoke({"input": prompt})
                
                answer = response["answer"]
                context = response["context"]
                
                # Display answer
                st.markdown(answer)
                
                # Display citations
                if context:
                    with st.expander("View Citations"):
                        for i, doc in enumerate(context):
                            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                            st.markdown(f"**Original File:** {doc.metadata.get('original_filename', 'Unknown')}")
                            st.code(doc.page_content[:500] + "...")
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {e}")
