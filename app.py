import streamlit as st
import os
import glob
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# ---------------------------------------------------------
# 0. INITIALIZATION
# ---------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# ---------------------------------------------------------
# UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Naruto Akatsuki RAG", page_icon="🍥", layout="centered")
st.title("🍥 Naruto Akatsuki AI Assistant")
st.markdown("Ask me anything about the Akatsuki from Naruto ! My answers are generated using the Fandom wiki corpus.")

# ---------------------------------------------------------
# I. INGESTION (Cached to run only once!)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading Akatsuki lore and building AI brain... Please wait.")
def init_rag():
    documents = []
    for filepath in glob.glob("corpus/akatsuki/*.txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            documents.append(Document(page_content=f.read(), metadata={"source": filepath}))
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore

# We load the database
vectorstore = init_rag()

# ---------------------------------------------------------
# II. CHAT HISTORY MANAGEMENT
# ---------------------------------------------------------
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------
# III. RETRIEVAL & GENERATION
# ---------------------------------------------------------
# React to user input
if question := st.chat_input("Ask a question (e.g., 'Who killed Deidara ?'):"):
    
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching in the corpus..."):
            
            # Retrieval
            retrieved_docs = vectorstore.similarity_search(question, k=10)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Prompting
            prompt = f"""You are an expert on the Naruto universe. Answer the question precisely using ONLY the following information. If the answer is not in the information, say that you do not know.

            Question: {question}

            Information extracted from the wiki:
            {context}
            """

            try:
                # Generation (using a stable free model)
                response = client.chat.completions.create(
                    model="openrouter/hunter-alpha",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content
                
                # Show the answer
                st.markdown(answer)
                
                # BONUS: Show the sources used to generate the answer!
                with st.expander("Sources used"):
                    for i, doc in enumerate(retrieved_docs):
                        # Extract just the filename to make it look clean
                        filename = os.path.basename(doc.metadata['source'])
                        st.caption(f"**Source {i+1}:** {filename}")
                        
            except Exception as e:
                answer = f"❌ An error occurred: {e}"
                st.error(answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})