import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import torch
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Legal AI Assistant", page_icon=":scales:")

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()  # Replace with desired LLM
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input for PDF chat
def handle_userinput_pdf_chat(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            st.write(message.content)
    else:
        st.warning("Please process your documents first.")

# Function to generate legal document
def generate_legal_document(query):
    llm = ChatOpenAI(temperature=0.7)
    messages = [
        HumanMessage(content=f"You are a legal document assistant. Generate a detailed outline for a legal document based on the following query: {query}")
    ]
    response = llm(messages)
    return response.content

def get_document_details(outline):
    llm = ChatOpenAI(temperature=0.7)
    messages = [
        HumanMessage(content=f"Based on this outline for a legal document:\n\n{outline}\n\nWhat additional details should I ask the user to provide? List 3-5 specific questions."),
    ]
    response = llm(messages)
    return response.content

def generate_final_document(outline, additional_info):
    llm = ChatOpenAI(temperature=0.7)
    messages = [
        HumanMessage(content=f"Generate a comprehensive legal document based on this outline:\n\n{outline}\n\nAnd these additional details:\n\n{additional_info}")
    ]
    response = llm(messages)
    return response.content

# Main function to run the Streamlit app
def main():
    load_dotenv()
    st.header("Legal AI Assistant :scales:")

    # Mode selection
    mode = st.sidebar.radio("Select Mode", ("PDF Chat", "Legal Document Generator"))

    if mode == "PDF Chat":
        st.subheader("Chat with multiple PDFs")
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput_pdf_chat(user_question)

        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)

    elif mode == "Legal Document Generator":
        st.subheader("Generate Legal Documents")
        
        if 'stage' not in st.session_state:
            st.session_state.stage = 0

        if st.session_state.stage == 0:
            query = st.text_area("Enter your legal document requirements:", height=150)
            if st.button("Generate Outline"):
                if query:
                    with st.spinner("Generating outline..."):
                        try:
                            outline = generate_legal_document(query)
                            st.session_state.outline = outline
                            st.session_state.stage = 1
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                else:
                    st.warning("Please enter your legal document requirements.")

        elif st.session_state.stage == 1:
            st.text_area("Document Outline:", st.session_state.outline, height=300)
            with st.spinner("Analyzing outline..."):
                additional_questions = get_document_details(st.session_state.outline)
            st.subheader("Additional Information Needed:")
            st.write(additional_questions)
            additional_info = st.text_area("Please provide the additional details:", height=200)
            if st.button("Generate Final Document"):
                if additional_info:
                    with st.spinner("Generating final document..."):
                        try:
                            final_document = generate_final_document(st.session_state.outline, additional_info)
                            st.session_state.final_document = final_document
                            st.session_state.stage = 2
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                else:
                    st.warning("Please provide the additional details.")

        elif st.session_state.stage == 2:
            st.subheader("Final Legal Document")
            st.text_area("Generated Legal Document:", st.session_state.final_document, height=400)
            st.download_button("Download Document", st.session_state.final_document, "legal_document.txt")
            if st.button("Start Over"):
                st.session_state.stage = 0
                st.experimental_rerun()

        elif st.session_state.stage == 2:
            st.subheader("Final Legal Document")
            st.text_area("Generated Legal Document:", st.session_state.final_document, height=400)
            st.download_button("Download Document", st.session_state.final_document, "legal_document.txt")
            if st.button("Start Over"):
                st.session_state.stage = 0
                st.experimental_rerun()

if __name__ == '__main__':
    main()