from time import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import display, Markdown

model = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
conversation_history = []

def format_conversation(history):
    formatted_history = ""
    for entry in history:
        role = entry["role"]
        content = entry["content"]
        formatted_history += f"{role}: {content}\n"
    return formatted_history

def add_to_conversation_history(history, role, content):
    history.append({"role": role, "content": content})


def query_model(
        system_message,
        user_message,
        temperature=0.7,
        max_length=1024
        ):
    start_time = time()
    user_message = "Question: " + user_message + " Answer:"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
        )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_p=0.9,
        temperature=temperature,
        #num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=max_length,
        return_full_text=False,
        pad_token_id=pipeline.model.config.eos_token_id
    )
    #answer = f"{sequences[0]['generated_text'][len(prompt):]}\n"
    answer = sequences[0]['generated_text']
    end_time = time()
    ttime = f"Total time: {round(end_time-start_time, 2)} sec."

    return user_message + " " + answer  + " " +  ttime

def query_model_with_history(system_message, user_message, temperature=0.1, max_length=256):
    # Add the system message and user message to the conversation history
    add_to_conversation_history(conversation_history, "System", system_message)
    add_to_conversation_history(conversation_history, "User", user_message)
    
    # Format the conversation history to be used as context
    context = format_conversation(conversation_history)
    
    # Query the model with the formatted context
    response = query_model(
        system_message=context,
        user_message=user_message,
        temperature=temperature,
        max_length=max_length
    )
    
    # Add the model's response to the conversation history
    add_to_conversation_history(conversation_history, "Assistant", response)
    
    # Display the response
    
    return response

# Define the strict system message for the legal chatbot
system_message = """
You are an expert legal assistant specialized in Indian law. Your role is to provide detailed insights, legal advice, and recommendations based on the Indian Constitution and the Indian Penal Code. You should strictly adhere to the following guidelines:

1. Only provide legal information and advice relevant to Indian law, specifically the Indian Constitution and the Indian Penal Code.
2. Refrain from answering questions that are out of context or unrelated to Indian law.
3. Always consider the legal principles, case laws, and statutory provisions applicable in India.
4. Offer practical recommendations on how to move forward with respect to legal queries.
5. Maintain professionalism and provide accurate, clear, and concise legal information.
6. Do not engage in any form of speculative advice or provide personal opinions.
7. Respect confidentiality and avoid discussing any sensitive or personal information that is not legally relevant.

Your primary goal is to assist users with their legal queries, ensuring that the advice is in accordance with Indian legal standards.

These are 2 examples 

#### User Query
"Describe about India with specialty? "

#### Legal Assistant Response
"I apologize, but I can only provide legal information and advice relevant to Indian law, specifically the Indian Constitution and the Indian Penal Code. Please ask a legal query so I can assist you accordingly."

#### User Query
"What is the process to file a patent in India and what are the relevant IPC codes?"

#### Legal Assistant Response
"To file a patent in India, you need to follow the process outlined under the Indian Patent Act, 1970. The Indian Penal Code (IPC) does not directly cover patents, but legal principles relevant to intellectual property rights fall under the Indian Patent Act.

### Steps to File a Patent in India:

1. **Determine Patentability**: Ensure your invention is patentable. The invention must be novel, non-obvious, and industrially applicable.

2. **Patent Search**: Conduct a thorough patent search to ensure your invention has not been previously patented.

3. **Prepare Patent Application**:
    - **Provisional Application**: If your invention is not yet finalized, you can file a provisional application to secure a filing date.
    - **Complete Specification**: If your invention is ready, file a complete specification detailing the invention.

4. **File the Patent Application**:
    - Submit your application to the Indian Patent Office. You can file it online or physically at one of the Patent Offices located in Kolkata, Mumbai, Chennai, and Delhi.

5. **Publication**: The patent application is published 18 months from the filing date. You can request early publication by paying a fee.

6. **Examination**: Request examination of your application within 48 months from the filing date. The patent office will review the application to ensure it meets all requirements.

7. **Response to Objections**: If the patent office raises any objections, you will need to respond and make necessary amendments to your application.

8. **Grant of Patent**: If the application meets all requirements and no objections remain, the patent is granted and published in the patent journal.

### Relevant Legal Codes:

- **Indian Patent Act, 1970**: The primary legislation governing patents in India.
- **Patent Rules, 2003**: These rules provide detailed procedures and requirements for filing patents.

### Practical Recommendations:

- **Consult a Patent Attorney**: It is advisable to consult with a patent attorney who can help you navigate the complexities of patent law and ensure your application is properly drafted and filed.
- **Use Online Resources**: The Indian Patent Office provides online resources and e-filing options which can streamline the application process.

By following these steps and considering the relevant legal provisions, you can effectively file a patent for your invention in India."
"""

def llama_qa(message):
    response = query_model(
        system_message,
        user_message=message,
        temperature=0.1,
        max_length=512)
    return response

def llama_chat(message):
    response = query_model_with_history(
    system_message=system_message,
    user_message=message,
    temperature=0.1,
    max_length=512
)