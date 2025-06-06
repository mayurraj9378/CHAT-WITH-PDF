import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time

# Fix torch watcher bug (still relevant for HuggingFace embeddings)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Set page config
st.set_page_config(
    page_title="Fast PDF Chat with Gemini",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to list available models (for debugging and model selection)
def list_available_models():
    try:
        models = genai.list_models()
        return [model.name for model in models if 'generateContent' in model.supported_generation_methods]
    except Exception as e:
        st.error(f"Error listing models: {e}")
        return []

# Cache embeddings
@st.cache_resource
def get_cached_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Cache QA Chain
@st.cache_resource
def get_cached_chain(_vectorstore):
    # List available models to ensure compatibility
    available_models = list_available_models()
    model_name = "gemini-1.5-pro" if "gemini-1.5-pro" in available_models else "gemini-1.5-flash"
    if not available_models:
        st.error("No compatible models found. Check your API key or network connection.")
        return None

    # Initialize ChatGoogleGenerativeAI with a supported model
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
    )

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, reply with:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    return RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# Split text
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len
    )
    return splitter.split_text(text)

# Create vector store
def get_vector_store(text_chunks):
    embeddings = get_cached_embeddings()

    progress_bar = st.progress(0)
    status = st.empty()

    batch_size = 20
    vectorstore = None

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        progress = (i + batch_size) / len(text_chunks)
        progress_bar.progress(min(progress, 1.0))
        status.text(f"Embedding chunk {i+1} - {min(i+batch_size, len(text_chunks))} of {len(text_chunks)}")

        if i == 0:
            vectorstore = FAISS.from_texts(batch, embedding=embeddings)
        else:
            partial_vs = FAISS.from_texts(batch, embedding=embeddings)
            vectorstore.merge_from(partial_vs)

    progress_bar.empty()
    status.empty()

    vectorstore.save_local("faiss_index")
    return vectorstore

# Ask question
def user_input(question):
    try:
        embeddings = get_cached_embeddings()
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        chain = get_cached_chain(vectorstore)
        if chain is None:
            st.error("Failed to initialize QA chain. Check model availability.")
            return
        response = chain.invoke({"query": question})

        st.markdown("### 📘 Answer:")
        st.write(response['result'])

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Try reprocessing the PDF or check your Google API key.")

# Main App
def main():
    st.title("⚡ Fast PDF Chat using Google's Gemini Model 📚")
    st.markdown("*Now using FREE Hugging Face embeddings with Google's Gemini!*")

    st.sidebar.header("📁 Upload PDF")
    st.sidebar.markdown("🚀 Powered by **Gemini + Hugging Face** - Fast & Powerful!")

    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.sidebar.button("🚀 Process PDFs", type="primary"):
        if pdf_docs:
            with st.spinner("Reading and embedding your PDF..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("❌ No text found in uploaded PDFs.")
                    return
                chunks = get_text_chunks(raw_text)
                st.info(f"✅ Split into {len(chunks)} chunks")
                get_vector_store(chunks)
                st.success("✅ PDF processed successfully!")
                st.balloons()
        else:
            st.error("⚠️ Please upload PDF files first.")

    if os.path.exists("faiss_index"):
        st.sidebar.success("📊 Vector DB Ready!")

    st.markdown("### 💬 Ask a Question")

    user_question = st.text_input(
        "What would you like to know about your PDFs?",
        placeholder="e.g. What is the summary of this paper?"
    )

    if user_question and user_question.strip():
        if not os.path.exists("faiss_index"):
            st.warning("⚠️ Please process a PDF first.")
        else:
            with st.spinner("Generating answer..."):
                start = time.time()
                user_input(user_question)
                st.caption(f"🕒 Answer generated in {time.time() - start:.2f} seconds")

    if not os.path.exists("faiss_index"):
        st.markdown("### 📝 Example Questions:")
        st.markdown("- What are the key takeaways?")
        st.markdown("- What topics are covered?")
        st.markdown("- List important definitions.")
        st.markdown("- Give a summary of the document.")

    if not os.path.exists("faiss_index"):
        st.info("ℹ️ First-time use will download a model (~90MB) from Hugging Face for embeddings.")

if __name__ == "__main__":
    main()