import os
import warnings
import streamlit as st
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the embeddings model
model_name = 'all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function for password authentication
def password_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        password = st.text_input("Enter the password to access the chatbot:", type="password")
        if password == os.getenv("PASSWORD"):
            st.session_state['authenticated'] = True
            st.success("Password correct! You can now use the chatbot.")
            return True
        elif password:
            st.error("Incorrect password. Please try again.")
        return False
    return True

# Main function
def main():
    if not password_authentication():
        return

    st.title("Carl Jung Chatbot - RAG Model")
    st.markdown("This chatbot allows you to ask questions about Carl Jung's works based on his collected writings.")

    cache_file = "cache_data.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        text_data = cache['text_data']
        chunks = cache['chunks']
        vectorstore = cache['vectorstore']
        st.success("Loaded cached data successfully!")
    else:
        st.error("Cache file not found. Please ensure 'cache_data.pkl' exists.")
        return

    st.success("Ready to answer questions!")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Initialize OpenAI LLM
    llm = OpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

    # Set up Conversational Retrieval Chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True
    )

    # Input box at the top of the page
    user_query = st.text_input("Ask a question about Carl Jung's works:")

    if user_query:
        # Get response from the chain
        response = qa_chain({"question": user_query})

        # Display the response
        st.session_state['chat_history'].append(("You", user_query))
        st.session_state['chat_history'].append(("JungBot", response['answer']))

    # Display chat history
    if st.session_state['chat_history']:
        for speaker, message in st.session_state['chat_history']:
            st.markdown(f"**{speaker}:** {message}")

if __name__ == "__main__":
    main()
