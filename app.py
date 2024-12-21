import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key= GEMINI_API_KEY)
google_credentials = st.secrets["GOOGLE_CREDENTIALS"] 

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Add text only if it's not empty
                text += page_text

    if not text.strip():  # Check if no text is found in PDFs
        st.error("No readable text found in the uploaded PDFs. Please upload valid files.")
        return None

    return text

# Function to split extracted text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    if not chunks:  # Check if chunks are empty
        st.error("Failed to split text into chunks. Please try again.")
        return None
    return chunks

# Function to create and save the FAISS vector store
def get_vector_store(text_chunks):
    if not text_chunks:  # Ensure chunks are not empty
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    # Debugging: Check directory contents
    st.info(f"FAISS index saved at: {os.path.abspath('faiss_index')}")
    if os.path.exists("faiss_index"):
        st.info(f"Directory contents: {os.listdir('faiss_index')}")

# Function to set up the conversational chain
def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
        If the answer is not in the provided context, just say, "answer is not available in the context." Do not provide incorrect answers.\n\n
        Context:\n {context}?\n
        Question:\n {question}\n
        Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate responses
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if FAISS index exists
    if not os.path.exists("faiss_index/index.faiss"):
        st.error("FAISS index not found. Please process PDF files first.")
        return

    # Load the FAISS index
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    # Generate the conversational response
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply:", response.get("output_text", "No reply generated."))

# Main function for the Streamlit app
def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:  # Stop if no text is extracted
                    return

                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:  # Stop if text splitting fails
                    return

                get_vector_store(text_chunks)
                st.success("Processing complete! You can now ask questions.")

if __name__ == "__main__":
    main()
