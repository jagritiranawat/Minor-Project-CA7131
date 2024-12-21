# Minor-Project-CA7131
Project
MCA Curriculum Chatbot 📚
An intelligent chatbot system designed to help students navigate and understand MCA curriculum documents using Retrieval-Augmented Generation (RAG) technology.
Features

Interactive chat interface for curriculum-related queries
PDF document processing and intelligent information retrieval
Context-aware responses using RAG architecture
Efficient conversation history management
User-friendly Streamlit interface

Technology Stack

Frontend: Streamlit
PDF Processing: PyPDF2
Vector Storage: FAISS
Language Model: Google Gemini Pro
Embeddings: Google Generative AI Embeddings
Framework: LangChain
Environment Management: python-dotenv

System Requirements

Python 3.8+ (Recommended 3.9-3.11)
Minimum 8GB RAM
Stable internet connection
Google API Key for Generative AI

Installation

Clone the repository
Install required packages:

bashCopypip install -r requirements.txt

Create a .env file with your Google API key:

CopyGOOGLE_API_KEY=your_api_key_here
Usage

Run the application:

bashCopystreamlit run app.py

Upload MCA curriculum PDF(s) through the sidebar
Click "Submit & Process" to initialize the system
Start asking questions about the curriculum

Key Features

PDF Processing: Efficiently extracts and processes text from curriculum PDFs
Semantic Search: Uses FAISS for accurate information retrieval
Conversational Interface: Maintains context across multiple queries
Structured Responses: Provides organized, clear answers to curriculum queries
History Management: Maintains recent conversation history
Error Handling: Gracefully handles invalid inputs and processing errors

Limitations

Currently supports only PDF format
Requires well-formatted input documents
Limited to curriculum-related queries
Depends on internet connection for API access

Future Scope

Multi-document type support
Integration with Learning Management Systems
Mobile application development
Voice interface implementation
Multilingual support
Predictive analytics features
Enhanced visualization capabilities

Project Team

Jagriti (23FS20MCA00023)
Ashutosh Tripathi (23FS20MCA00022)

Institution
Department of Computer Applications
Manipal University Jaipur, Jaipur-303007 (RAJASTHAN) INDIA
Faculty Coordinators

Dr. Linesh Raja (Associate Professor)
Dr. Govind Murari Upadhyay (Assistant Professor, Senior Scale)

License
This project was developed as part of the MCA III Semester Minor Project.
Note
This chatbot is specifically designed for MCA curriculum queries. For best results, ensure that uploaded PDFs are well-formatted and contain clear curriculum information.
