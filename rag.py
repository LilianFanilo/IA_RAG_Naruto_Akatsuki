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
# I. INGESTION: Data Preparation (Runs only once!)
# ---------------------------------------------------------
print("1. Loading Akatsuki texts...")
documents = []
for filepath in glob.glob("corpus/akatsuki/*.txt"):
    with open(filepath, 'r', encoding='utf-8') as f:
        documents.append(Document(page_content=f.read(), metadata={"source": filepath}))

print("2. Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print("3. Creating vector database (FAISS)... This might take a few seconds.")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

print("\n" + "="*50)
print("🧠 RAG SYSTEM READY! Ask your questions about the Akatsuki.")
print("Type 'exit' or 'quit' to stop.")
print("="*50)

# ---------------------------------------------------------
# II & III. RETRIEVAL & GENERATION (Interactive Loop)
# ---------------------------------------------------------
while True:
    # Get user input from the terminal
    question = input("\n👉 Your question: ")

    # Check if the user wants to quit
    if question.lower() in ['exit', 'quit']:
        print("Goodbye! Shutting down the RAG...")
        break
        
    # Skip if the user just pressed Enter without typing anything
    if not question.strip():
        continue

    print("Searching for information and generating answer...")

    # Retrieval
    retrieved_docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Building the prompt
    prompt = f"""You are an expert on the Naruto universe. Answer the question precisely using ONLY the following information. If the answer is not in the information, say that you do not know.

Question: {question}

Information extracted from the wiki:
{context}
"""

    # Generation
    try:
        response = client.chat.completions.create(
            model="openrouter/hunter-alpha",
            messages=[{"role": "user", "content": prompt}]
        )
        
        print("\n🎯 RAG RESPONSE:")
        print(response.choices[0].message.content)
        print("-" * 50)
        
    except Exception as e:
        print(f"\n❌ An error occurred with OpenRouter: {e}")