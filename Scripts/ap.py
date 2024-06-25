import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import torch
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="ASHRAY", page_icon=":scales:")

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to create conversation chain
def get_conversation_chain():
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory()
    
    template = """You are ASHRAY, a compassionate and knowledgeable mental health assistant. 
    Your role is to provide support, guidance, and solutions to users experiencing mental health issues. 
    Always maintain a caring and professional tone. If you suspect the user is in immediate danger or 
    experiencing a severe crisis, advise them to seek professional help immediately.

    Current conversation:
    {history}
    Human: {input}
    ASHRAY:"""
    
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    return conversation_chain

# Function to handle user input and generate response
def handle_user_input(user_input, conversation):
    response = conversation.predict(input=user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
    return response

# Main function to run the Streamlit app
def main():
    st.header("ASHRAY - Your Mental Health Assistant :brain:")
    st.write("Hello! I'm ASHRAY, your personal mental health assistant. How can I help you today?")

    # Initialize conversation chain
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    user_input = st.chat_input("Share your thoughts or concerns...")

    if user_input:
        # Generate and display response
        response = handle_user_input(user_input, st.session_state.conversation)
        with st.chat_message("assistant"):
            st.write(response)

    # Add a disclaimer
    st.sidebar.markdown("""
    **Disclaimer:** ASHRAY is an AI assistant and not a substitute for professional mental health care. 
    If you're experiencing a mental health emergency, please contact your local emergency services or 
    a mental health crisis hotline immediately.
    """)

if __name__ == '__main__':
    main()