# MCA Curriculum Chatbot 📚

An intelligent chatbot system designed to help students navigate and understand MCA curriculum documents using Retrieval-Augmented Generation (RAG) technology.

## Features

- Interactive chat interface for curriculum-related queries
- PDF document processing and intelligent information retrieval
- Context-aware responses using RAG architecture
- Efficient conversation history management
- User-friendly Streamlit interface

## Technology Stack

- **Frontend**: Streamlit
- **PDF Processing**: PyPDF2
- **Vector Storage**: FAISS
- **Language Model**: Google Gemini Pro
- **Embeddings**: Google Generative AI Embeddings
- **Framework**: LangChain
- **Environment Management**: python-dotenv

## System Requirements

- Python 3.8+ (Recommended 3.9-3.11)
- Minimum 8GB RAM
- Stable internet connection
- Google API Key for Generative AI

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```
2. Upload MCA curriculum PDF(s) through the sidebar
3. Click "Submit & Process" to initialize the system
4. Start asking questions about the curriculum

## Key Features

- **PDF Processing**: Efficiently extracts and processes text from curriculum PDFs
- **Semantic Search**: Uses FAISS for accurate information retrieval
- **Conversational Interface**: Maintains context across multiple queries
- **Structured Responses**: Provides organized, clear answers to curriculum queries
- **History Management**: Maintains recent conversation history
- **Error Handling**: Gracefully handles invalid inputs and processing errors

## Limitations

- Currently supports only PDF format
- Requires well-formatted input documents
- Limited to curriculum-related queries
- Depends on internet connection for API access

## Future Scope

- Multi-document type support
- Integration with Learning Management Systems
- Mobile application development
- Voice interface implementation
- Multilingual support
- Predictive analytics features
- Enhanced visualization capabilities

## Project Team

- Jagriti (23FS20MCA00023)
- Ashutosh Tripathi (23FS20MCA00022)

## Institution

Department of Computer Applications  
Manipal University Jaipur, Jaipur-303007 (RAJASTHAN) INDIA

## Faculty Coordinators

- Dr. Linesh Raja (Associate Professor)
- Dr. Govind Murari Upadhyay (Assistant Professor, Senior Scale)

## License

This project was developed as part of the MCA III Semester Minor Project.

## Note

This chatbot is specifically designed for MCA curriculum queries. For best results, ensure that uploaded PDFs are well-formatted and contain clear curriculum information.
