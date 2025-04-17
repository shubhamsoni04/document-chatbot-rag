import os
import time
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st


def load_docs_from_url(input_url):
    web_loader = WebBaseLoader(input_url)
    documents = web_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_documents = splitter.split_documents(documents)
    st.write('Documents Loaded from URL')
    return split_documents

def load_docs_from_pdf(file):
    start = time.time()
    with open("temp.pdf", "wb") as temp_file:
        temp_file.write(file.getbuffer())
    pdf_loader = PyPDFLoader("temp.pdf")
    pdf_documents = pdf_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    processed_documents = splitter.split_documents(pdf_documents)
    st.write('Documents Loaded')
    st.write(f"Time taken to load documents: {time.time() - start:.2f} seconds")
    os.remove("temp.pdf")
    return processed_documents

def build_vector_db(documents):
    start = time.time()
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"trust_remote_code": True})
    db = FAISS.from_documents(documents, embedder)
    st.write('DB is ready')
    st.write(f"Time taken to create DB: {time.time() - start:.2f} seconds")
    return db

def call_groq_api(chat_messages):
    load_dotenv()
    api_key = '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'
    groq_client = Groq(api_key=api_key)
    full_response = ''
    stream_output = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=chat_messages,
        max_tokens=1024,
        temperature=1.3,
        stream=True,
    )

    for chunk in stream_output:
        part = chunk.choices[0].delta.content
        if part:
            full_response += part
    return full_response

def summarize_conversation(history):
    history_text = " ".join([f"{entry['role']}: {entry['content']}" for entry in history])
    summary_prompt = f"Summarize the following chat history:\n\n{history_text}"
    summary_messages = [{'role': 'system', 'content': 'You are very good at summarizing the chat between User and Assistant'}]
    summary_messages.append({'role': 'user', 'content': summary_prompt})
    return call_groq_api(summary_messages)

def main():
    st.set_page_config(page_title='Chatbot')
    st.title("Document Chatbot using RAGs - Harsh Tripathi and Shubham Soni")

    with st.expander("Instructions to upload Text PDF/URL"):
        st.write("1. Open the sidebar by clicking the menu icon in the top-left corner.")
        st.write("2. If you're uploading a PDF, click 'Upload PDF', choose your file, and wait until you see the 'Documents Loaded' message.")
        st.write("3. If you're using a web URL, type it in, click 'Enter Web URL', then hit 'Process URL' and wait for the 'Documents Loaded from URL' confirmation.")
        st.write("4. Once the documents are loaded, click 'Create Vector Store' to begin processing. Note that documents can only be uploaded once per session.")
        st.write("5. Type your question into the text box and click submit to start interacting with the AI chatbot.")
        st.write("6. Click 'Generate Chat Summary' to view a summary of the entire chat session.")

    st.sidebar.subheader("Choose document source:")
    input_choice = st.sidebar.radio("Select one:", ("Upload PDF", "Enter Web URL"))

    if "loaded_docs" not in st.session_state:
        st.session_state.loaded_docs = None
    if "document_db" not in st.session_state:
        st.session_state.document_db = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "latest_output" not in st.session_state:
        st.session_state.latest_output = ""
    if "chat_summary" not in st.session_state:
        st.session_state.chat_summary = ""

    if input_choice == "Upload PDF":
        file_input = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
        if file_input is not None:
            if st.session_state.loaded_docs is None:
                with st.spinner("Loading documents..."):
                    documents = load_docs_from_pdf(file_input)
                st.session_state.loaded_docs = documents

    elif input_choice == "Enter Web URL":
        input_url = st.sidebar.text_input("Enter URL", key="url_input")
        if st.session_state.url_input != input_url:
            st.session_state.url_input = input_url
            st.session_state.loaded_docs = None
        if st.sidebar.button('Process URL'):
            if input_url and st.session_state.loaded_docs is None:
                with st.spinner("Fetching and processing documents from URL..."):
                    documents = load_docs_from_url(input_url)
                st.session_state.loaded_docs = documents

    if st.session_state.loaded_docs is not None:
        if st.sidebar.button('Create Vector Store'):
            with st.spinner("Creating vector store..."):
                db = build_vector_db(st.session_state.loaded_docs)
            st.session_state.document_db = db

    if st.session_state.document_db is not None:
        def submit_with_db():
            user_input = st.session_state.query_input
            if user_input:
                doc_retriever = st.session_state.document_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                results = doc_retriever.invoke(user_input)
                constructed_prompt = f'''
                Answer the user's question based on the latest input provided in the chat history. Ignore
                previous inputs unless they are directly related to the latest question. Provide a generic
                answer if the answer to the user's question is not present in the context by mentioning it
                as general information.

                Context: {results}

                Chat History: {st.session_state.conversation}

                Latest Question: {user_input}
                '''
                message_flow = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
                message_flow.append({'role': 'user', 'content': constructed_prompt})

                try:
                    reply = call_groq_api(message_flow)
                except Exception as err:
                    st.error(f"Error occurred during chat_groq execution: {str(err)}")
                    reply = "An error occurred while fetching response. Please try again."

                st.session_state.latest_output = reply
                st.session_state.conversation.append({'role': 'user', 'content': user_input})
                st.session_state.conversation.append({'role': 'assistant', 'content': reply})
                st.session_state.query_input = ""

    def handle_general_chat():
        user_input = st.session_state.query_input
        if user_input:
            prompt_text = f'''
            Answer the user's question based on the latest input provided in the chat history. Ignore
            previous inputs unless they are directly related to the latest
            question. 
            
            Chat History: {st.session_state.conversation}

            Latest Question: {user_input}
            '''
            message_flow = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
            message_flow.append({'role': 'user', 'content': prompt_text})

            try:
                reply = call_groq_api(message_flow)
            except Exception as err:
                st.error(f"Error occurred during chat_groq execution: {str(err)}")
                reply = "An error occurred while fetching response. Please try again."

            st.session_state.latest_output = reply
            st.session_state.conversation.append({'role': 'user', 'content': user_input})
            st.session_state.conversation.append({'role': 'assistant', 'content': reply})
            st.session_state.query_input = ""

    st.text_area("Enter your question:", key="query_input")
    if st.session_state.document_db is not None:
        st.button('Submit', on_click=submit_with_db)  
    else:
        st.button('Submit', on_click=handle_general_chat)

    if st.session_state.latest_output:
        st.write(st.session_state.latest_output)

    if st.button('Generate Chat Summary'):
        st.session_state.chat_summary = summarize_conversation(st.session_state.conversation)

    if st.session_state.chat_summary:
        with st.expander("Chat Summary"):
            st.write(st.session_state.chat_summary)

    with st.expander("Recent Chat History"):
        latest_entries = st.session_state.conversation[-8:][::-1]
        paired_entries = []
        for i in range(0, len(latest_entries), 2):
            if i+1 < len(latest_entries):
                paired_entries.extend([latest_entries[i+1], latest_entries[i]])
            else:
                paired_entries.append(latest_entries[i])
        for entry in paired_entries:
            st.write(f"{entry['role'].capitalize()}: {entry['content']}")

if __name__ == "__main__":
    main()
