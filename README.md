# 📄 PDF Query System using Gemini API & Pinecone

This project is a **PDF-based AI query system** that allows users to upload PDFs, store their embeddings in **Pinecone (vector database)**, and query them using **Google Gemini API** to retrieve relevant answers with references.

## 🚀 Featuresfgh
- 📂 Upload **multiple PDFs**
- 🔎 Store and search document embeddings in **Pinecone**
- 🧠 Use **Gemini API** for answering queries
- 🔐 Full authentication (User Registration, Login, Logout, Password Reset)
- 🌐 **Flask-based** backend with **Streamlit** frontend


📚 Usage
Upload PDFs through the web interface.
The system extracts text and generates embeddings.
Store embeddings in Pinecone.
Ask questions, and the system retrieves the most relevant answers using the Gemini API.

🔧 Technologies Used
Python (Flask, Streamlit)
Google Gemini API (LLM)
Pinecone (Vector Database)
LangChain (for embedding generation)
PyMuPDF / pdfplumber (for PDF processing)
Authentication (Flask-Login, Flask-SQLAlchemy)

📌 TODO
 Improve accuracy with RAG (Retrieval-Augmented Generation)
 Add file deletion and management features
 Deploy on Render / AWS / Google Cloud
 
🤝 Contributing
Feel free to fork this repository and submit a pull request!

🔗 Connect with Me
📧 Email: amrizwan175@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/rizwan-ali-ai/
