# 📄 PDF Query System using OpenAI API & Pinecone

An **AI-powered PDF query system** that allows users to upload PDF files, store their embeddings in **Pinecone**, and ask questions from the documents using **OpenAI's GPT models**. Built entirely using **Streamlit** for a smooth and interactive experience.

---

## 🚀 Features

- 📂 Upload and process multiple PDF files
- 🔎 Store and search PDF content using Pinecone vector database
- 🧠 Get AI-generated answers using OpenAI (GPT-3.5 / GPT-4)
- 📚 Get references to the most relevant PDF sources
- 🧱 Built with **Streamlit** – no backend required!

---

## 📚 How It Works

1. **Upload PDFs** through the Streamlit UI.
2. PDFs are **parsed, chunked**, and **embedded** using OpenAI Embeddings.
3. Embeddings are stored in **Pinecone** for semantic search.
4. When a question is asked:
   - Relevant chunks are retrieved from Pinecone.
   - Context is passed to OpenAI GPT to generate a **concise answer**.
   - Answer includes **references** to original PDF sections.

---

## 🛠️ Tech Stack

| Technology     | Purpose                             |
|----------------|-------------------------------------|
| Streamlit      | UI / Frontend                       |
| OpenAI API     | Embedding + GPT-3.5/4-based answers |
| Pinecone       | Vector database for similarity search |
| LangChain      | Text splitting, QA chain management |
| PyPDFLoader    | PDF parsing and text extraction     |
| Python         | Core programming language           |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/pdf-query-openai.git
cd pdf-query-openai
