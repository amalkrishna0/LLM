import time
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

load_dotenv()

from footer import footer

st.set_page_config(page_title="Law-GPT", layout="centered")

col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("images/banner.png", use_column_width=True)

def hide_hamburger_menu():
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

hide_hamburger_menu()

# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.initial_prompt = True  # Track if the initial prompt was shown
    st.session_state.section = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

@st.cache_resource
def load_embeddings():
    """Load and cache the embeddings model."""
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

embeddings = load_embeddings()
db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Prompt templates for different sections
advisory_prompt_template = """
<s>[INST]
As a legal advisor specializing in the Indian Penal Code, you are tasked with providing highly accurate and contextually appropriate advice. Ensure your answers meet these criteria:
- Respond in a bullet-point format to clearly delineate distinct aspects of the legal query.
- Each point should accurately reflect the breadth of the legal provision in question, avoiding over-specificity unless directly relevant to the user's query.
- Clarify the general applicability of the legal rules or sections mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
- Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
- Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations unless otherwise specified.
- Conclude with a brief summary that captures the essence of the legal discussion and corrects any common misinterpretations related to the topic.

If the user query is of type 'Advice':
- Provide practical legal advice based on the context of the user's query.
- Highlight any relevant sections of the IPC that might apply to the situation.
- Offer actionable steps the user can take based on the legal framework.
- Clarify any potential legal consequences and common pitfalls.
- Suggest when it might be necessary to consult with a legal professional for further guidance.
- **Include the number of years of imprisonment in bold characters where applicable.**

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
- [Detail the first key aspect of the law, ensuring it reflects general application]
- [Provide a concise explanation of how the law is typically interpreted or applied]
- [Correct a common misconception or clarify a frequently misunderstood aspect]
- [Detail any exceptions to the general rule, if applicable]
- [Include any additional relevant information that directly relates to the user's query]
- **[Include the number of years of imprisonment in bold characters where applicable]**
</s>[INST]
"""

ipc_prompt_template = """
<s>[INST]
As a legal chatbot specializing in the Indian Penal Code, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
- Respond in a bullet-point format to clearly delineate distinct aspects of the legal query.
- Each point should accurately reflect the breadth of the legal provision in question, avoiding over-specificity unless directly relevant to the user's query.
- Clarify the general applicability of the legal rules or sections mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
- Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
- Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations unless otherwise specified.
- Conclude with a brief summary that captures the essence of the legal discussion and corrects any common misinterpretations related to the topic.

If the user query is of type 'Query':
- Provide a detailed and accurate answer based on the Indian Penal Code.
- Include the specific sections of the IPC that are relevant to the query.
- Explain the general applicability of these sections.
- Correct any common misconceptions or frequently misunderstood aspects.
- Detail any exceptions to the general rule, if applicable.
- Include any additional relevant information that directly relates to the user's query.
- **Include the number of years of imprisonment in bold characters and the amount of fine in INR where applicable.**
- **Provide the exact statement of the relevant IPC code from the IPC book.**

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
- [Detail the first key aspect of the law, ensuring it reflects general application]
- [Provide a concise explanation of how the law is typically interpreted or applied]
- [Correct a common misconception or clarify a frequently misunderstood aspect]
- [Detail any exceptions to the general rule, if applicable]
- [Include any additional relevant information that directly relates to the user's query]
- **Section [IPC Code]: "[Exact statement of the IPC code from the IPC book]"**
- **[Include the number of years of imprisonment in bold characters and the amount of fine in INR where applicable]**
</s>[INST]
"""

# Function to initialize the ConversationalRetrievalChain based on the section
def initialize_qa(section):
    prompt_template = advisory_prompt_template if section == "advisory" else ipc_prompt_template
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

api_key = os.getenv('OPEN_API_KEY')
llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=1024)

def extract_answer(full_response):
    """Extracts the answer from the LLM's full response by removing the instructional text."""
    answer_start = full_response.find("Response:")
    if answer_start != -1:
        answer_start += len("Response:")
        answer_end = len(full_response)
        return full_response[answer_start:answer_end].strip()
    return full_response

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.initial_prompt = True
    st.session_state.section = None

# Display initial prompt for section choice
if st.session_state.initial_prompt:
    with st.chat_message("assistant"):
        st.markdown("**Please choose an option to begin:**\n\n1. Advisory session\n2. IPC code questions")
    st.session_state.initial_prompt = False

# Handle user's choice
if st.session_state.section is None:
    input_prompt = st.chat_input("Choose an option (1 or 2):")
    if input_prompt:
        if input_prompt == "1":
            st.session_state.section = "advisory"
            st.session_state.messages.append({"role": "assistant", "content": "You have chosen: Advisory session"})
        elif input_prompt == "2":
            st.session_state.section = "ipc"
            st.session_state.messages.append({"role": "assistant", "content": "You have chosen: IPC code questions"})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Invalid choice. Please choose either 1 or 2."})
            st.experimental_rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

else:
    qa = initialize_qa(st.session_state.section)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    input_prompt = st.chat_input("Ask your question...")
    if input_prompt:
        with st.chat_message("user"):
            st.markdown(f"**You:** {input_prompt}")

        st.session_state.messages.append({"role": "user", "content": input_prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking 💡..."):
                result = qa.invoke(input=input_prompt)
                message_placeholder = st.empty()
                answer = extract_answer(result["answer"])

                full_response = "⚠️ **_Gentle reminder: We generally ensure precise information, but do double-check._** \n\n\n"
                for chunk in answer:
                    full_response += chunk
                    time.sleep(0.0001)
                    message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer})

            if st.button('🗑️ Reset All Chat', on_click=reset_conversation):
               

                st.experimental_rerun()

footer()
