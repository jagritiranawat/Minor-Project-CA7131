import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from collections import deque

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store using the embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up the conversational chain using LangChain
def get_conversational_chain():
    prompt_template = """
You are a helpful assistant for the MCA curriculum. The curriculum is divided into semesters with different subjects and additional program-specific details. You should answer questions based on the uploaded PDF containing the MCA syllabus, including all sections, tables, course codes, subjects, and other program details. Follow these guidelines to answer questions more effectively:

1. **Tables and Lists**: For questions related to courses, semester details, or elective options, extract information from the provided tables and list format. Format your answers in a clear and structured way.
   - Example: For "What are the program electives in the second semester?", list the electives and their course codes.

2. **Program Specific Outcomes (PSOs)**: If the user asks about the PSOs or outcomes of the MCA program, refer to the program-specific outcomes and provide them in the answer.
   - Example: "What are the PSOs of the MCA program?" should list all the PSOs like:
     - PSO.1: To work productively as an IT professional in supportive and leadership roles.
     - PSO.2: To advance successfully in chosen career paths with technical abilities, leadership qualities, and communication skills.

3. **Year and Semester Information**: When the user asks about specific semesters or subjects, give a detailed explanation of the course structure and electives for that semester.
   - Example: For "What courses are in the first semester?", list all subjects and their corresponding course codes, credits, and lecture/lab hours.

4. **Course Details**: If asked about a specific course or subject, provide a summary of the course along with prerequisites, objectives, or any associated lab details.
   - Example: "What is the syllabus for the 'Discrete Mathematical Structures with Graph Theory' course?"

5. **Text Format**: For structured sections, like eligibility criteria, course prerequisites, and other program requirements, follow the same format and present the information clearly.
   - Example: "What is the eligibility criteria for the MCA program?"

6. **Handling Annexures and Special Sections**: If there are annexures or other special sections, extract relevant information and provide concise summaries for questions related to those sections.
   - Example: "What is the content of 'Annexure 1' in the syllabus?"

7. **Handling Bridge Courses**: For questions about bridge courses or additional prerequisites, refer to the specific details about the courses, topics, and conditions mentioned in the syllabus.
   - Example: "What are the details of the bridge course for students with no mathematics background?"

8. **General Information**: For general inquiries about the MCA program, its structure, and eligibility, provide comprehensive answers based on the sections like eligibility, program structure, and specific outcomes.

9. **Here - BASIC MATHEMATICS FOR COMPUTER APPLICATIONS (BRIDGE COURSE) [0 0 0 0]** - four 0 are Lecture, Tutorial, Practical, Credits so if you are showing a subject detail do write these explicitly for that subject like:
   Lecture - 0
   Tutorial - 0
   Practical - 0
   Credits - 0
   LTPC Format will be there always so show this accordingly and add this list as shown above

10. **Common Subjects like Maths**: Show all instances where it is found, including semester and syllabus.
Make sure to answer in detail using the context provided. If the question is not available in the context, just say "Answer not available in the context."

11.if asked for whole semester syllabus always return complete solid table along with program electives table separtely downward after table strictly return ###table.

12.Always Show Course Code like CA-7106 of that specific subject which is asked 

13.STRICTLY DO NOT GO BEYOND CONTEXT NEITHER GENERATE NEW CONTENT and fcus on providing correct syllabus if asked only correct syllabus

14.Always show program electives in tabular form if asked and start each subject from new row


Context: {context}

Question: {question}

Answer:
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to process user input and get response from the chatbot
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the FAISS index with embeddings
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question,k=5)

    chain = get_conversational_chain()

    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

# Chat History Function to maintain flow of conversation
def chat_history(messages, user_question, response):
    messages.append({"role": "user", "content": user_question})
    messages.append({"role": "assistant", "content": response})
    return messages

# Main function to set up the Streamlit interface
import streamlit as st
from collections import deque

# Function to set up the Streamlit interface
def main():
    st.set_page_config("MCA Curriculum Chatbot 📚")
    st.title("📚 MCA Curriculum Chatbot")

    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = deque(maxlen=10)  # Keep only last 10 messages

    # Sidebar for file upload and reset option
    with st.sidebar:
        st.title("Upload MCA Syllabus 📄")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, label_visibility="collapsed")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("MCA Curriculum PDF processed successfully! Ready to chat.")
        
        # Option to reset the chat history
        if st.button("Start New Chat"):
            st.session_state.messages.clear()  # Clear chat history

    # Display chat history in a flowing manner
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.chat_message("user").markdown(message['content'])
        else:
            st.chat_message("assistant").markdown(message['content'])

    # Create a new input box with a unique key
    user_question = st.text_input("Ask me anything about the MCA syllabus...", key=f"input_box_{len(st.session_state.messages)}")

    if user_question:
        with st.spinner("Finding answer..."):
            # Get the response from the assistant
            response = user_input(user_question)
            
            # Append new messages to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display the new messages in chat format
            st.chat_message("user").markdown(user_question)
            st.chat_message("assistant").markdown(response)

            # Automatically focus the input box for the next query by updating the key dynamically
            st.text_input("Ask me anything about the MCA syllabus...", key=f"input_box_{len(st.session_state.messages)}")

if __name__ == "__main__":
    main()
