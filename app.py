import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
import time

# ✅ Load environment variables
load_dotenv()

# ✅ Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Stop execution if keys are missing
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.error("🚨 API keys are missing! Set them in your environment variables.")
    st.stop()

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Define Pinecone Index Name
INDEX_NAME = "langchainvector"

# ✅ Check if the Pinecone index exists; if not, create it
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    st.info("ℹ️ Creating a new Pinecone index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(2)  # Wait for index creation
    st.success("✅ Pinecone index created successfully!")

# ✅ Ensure "uploads" directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Function to Load PDF Files
def load_pdfs(pdf_files):
    """Loads and extracts text from uploaded PDFs."""
    docs = []
    for pdf_file in pdf_files:
        file_path = os.path.join(UPLOAD_FOLDER, pdf_file.name)

        # Save PDF locally
        with open(file_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        # Load and extract text
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    return docs

# ✅ Function to Split Documents into Chunks
def split_docs(documents):
    """Splits documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# ✅ Function to Process PDFs and Store in Pinecone
def process_and_store_pdfs(pdf_files):
    """Processes PDFs, splits them into chunks, and stores embeddings in Pinecone."""
    documents = load_pdfs(pdf_files)
    if not documents:
        return "⚠️ No text extracted from PDFs."

    chunks = split_docs(documents)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Store in Pinecone
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=INDEX_NAME
    )

    return f"✅ Successfully stored {len(chunks)} chunks in Pinecone!"

# ✅ Function to Answer Queries with References
def answer_question_with_references(question):
    """Retrieves the best matching answer from stored PDFs or uses OpenAI if no context is found."""
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Load Pinecone vector store
        vector_store = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

        # Use a retriever with refined search
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Fetch top 5 relevant chunks

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(question)

        # Prepare context and references
        context = ""
        references = []
        for doc in retrieved_docs[:3]:  # Limit references to 3
            source = doc.metadata.get("source", "Unknown PDF")
            page = doc.metadata.get("page", "Unknown Page")
            text_snippet = doc.page_content[:300]  # Short preview of content
            context += f"{text_snippet}...\n"
            references.append(f"📄 **Source:** {source} | 📜 **Page:** {page}")

        # ✅ If enough context found, return the extracted answer
        if context.strip():
            return context, references

        # ✅ If no relevant content, use LLM (GPT-3.5) to generate answer
        chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

        # Define structured prompt
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an AI assistant answering based on the provided documents.\n\n"
                "**Context:**\n{context}\n\n"
                "**User Question:** {question}\n\n"
                "Provide a **concise and accurate answer** with references if applicable."
            ),
        )

        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt}
        )

        # Run the query
        response = qa_chain.invoke({"query": question})
        answer = response.get("result", "No answer found.")

        return answer, references if references else ["No relevant sources found."]

    except Exception as e:
        return f"⚠️ Error: {str(e)}", []

# ✅ Streamlit UI
st.title("📄 PDF Query System with AI-Powered Answers")

# ✅ Upload PDFs
pdf_files = st.file_uploader("📂 Upload PDFs", accept_multiple_files=True, type=["pdf"])

if st.button("🚀 Process PDFs"):
    if pdf_files:
        message = process_and_store_pdfs(pdf_files)
        st.success(message)
    else:
        st.warning("⚠️ Please upload at least one PDF.")

# ✅ Question Input
question = st.text_input("❓ Ask a question from the uploaded PDFs:")

if st.button("🔎 Get Answer"):
    if question:
        with st.spinner("⏳ Searching for an answer..."):
            answer, references = answer_question_with_references(question)

        st.write("### 💡 Answer:")
        st.write(answer)

        if references:
            st.write("### 📚 References:")
            for ref in references:
                st.write(f"- {ref}")
    else:
        st.warning("⚠️ Please enter a question.")
