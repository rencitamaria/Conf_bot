import streamlit as st
import os
import tempfile
import re
import pandas as pd
from typing import List, Dict
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------- PDF Loader ------------------
class SimplePDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        documents = []
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

                class Document:
                    def __init__(self, content, metadata=None):
                        self.page_content = content
                        self.metadata = metadata or {}

                documents.append(Document(text))
                return documents
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return []

# ---------------- Text Splitter ------------------
class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        chunks = []
        for doc in documents:
            text = doc.page_content
            sentences = re.split(r'[.!?]+', text)

            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < self.chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        class Document:
                            def __init__(self, content):
                                self.page_content = content.strip()
                        chunks.append(Document(current_chunk.strip()))
                    current_chunk = sentence + ". "

            if current_chunk.strip():
                class Document:
                    def __init__(self, content):
                        self.page_content = content.strip()
                chunks.append(Document(current_chunk.strip()))

        return chunks

# ---------------- Map Helper ------------------
def show_conference_map():
    latitude = 12.9666    # CMRIT AECS Layout latitude
    longitude = 77.7118   # CMRIT AECS Layout longitude

    df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
    st.map(df)

# ---------------- Q&A System ------------------
class SimpleQASystem:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        doc_texts = [doc.page_content for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(doc_texts)
        self.keywords = self._extract_keywords(doc_texts)

    def _extract_keywords(self, texts):
        all_text = " ".join(texts).lower()
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]

    def answer_question(self, question, top_k=3):
        question_lower = question.lower()
        if any(word in question_lower for word in ['where', 'location', 'venue']):
            st.info("📍 The conference is held at our college. Here's the map:")
            show_conference_map()
            return "The venue is our college campus. Please refer to the map above!"

        try:
            question_vector = self.vectorizer.transform([question_lower])
            similarities = cosine_similarity(question_vector, self.doc_vectors).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            relevant_docs = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    relevant_docs.append({
                        'content': self.documents[idx].page_content,
                        'score': similarities[idx]
                    })

            if not relevant_docs:
                return self._generate_fallback_response(question)

            return self._generate_answer(question, relevant_docs)

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def _generate_fallback_response(self, question):
        question_lower = question.lower()
        if any(word in question_lower for word in ['what', 'define', 'explain']):
            return "I couldn't find specific information about that topic. Try asking about the main topics or speakers."
        elif any(word in question_lower for word in ['when', 'time', 'date']):
            return "I couldn't find the schedule details. Please check the official program."
        elif any(word in question_lower for word in ['who', 'speaker', 'presenter']):
            return "I couldn't find the speaker list. Please refer to the conference brochure."
        else:
            return "I couldn't find relevant information. Try rephrasing or asking about a different topic."

    def _generate_answer(self, question, relevant_docs):
        combined_content = ""
        for doc in relevant_docs[:2]:
            combined_content += doc['content'][:500] + "\n\n"

        question_lower = question.lower()
        if any(word in question_lower for word in ['summary', 'summarize', 'overview']):
            return f"Summary:\n\n{combined_content[:600]}..."
        elif any(word in question_lower for word in ['main', 'key', 'important']):
            return f"Key points:\n\n{combined_content[:600]}..."
        elif any(word in question_lower for word in ['how', 'process', 'method']):
            return f"Here's how it works:\n\n{combined_content[:600]}..."
        else:
            return f"Answer to your question:\n\n{combined_content[:600]}..."

# ---------------- Streamlit UI ------------------
st.set_page_config(page_title="🤖 Conference Q&A Bot", layout="wide")
st.title("🎓 Conference Q&A Assistant (Offline)")
st.markdown("Ask questions about your conference document using basic AI magic!")

# Sidebar
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Upload Conference PDF", type=['pdf'])

    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Overlap", 0, 200, 100)
    top_k = st.slider("Top Results", 1, 5, 3)

# Session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load PDF
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("⏳ Processing..."):
        loader = SimplePDFLoader(pdf_path)
        documents = loader.load()

        if documents:
            splitter = SimpleTextSplitter(chunk_size, chunk_overlap)
            chunks = splitter.split_documents(documents)

            qa = SimpleQASystem(chunks)
            st.session_state.qa_system = qa

            st.success(f"✅ Loaded {len(chunks)} chunks")

# Q&A Section
if st.session_state.qa_system:
    st.header("💬 Ask Your Question")
    question = st.text_input("What do you want to know?", placeholder="e.g., Where is the venue?")
    col1, col2 = st.columns([4, 1])

    with col1:
        if st.button("🔍 Get Answer", disabled=not question):
            answer = st.session_state.qa_system.answer_question(question, top_k)
            st.subheader("📝 Answer")
            st.write(answer)
            st.session_state.chat_history.append((question, answer))

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # History
    if st.session_state.chat_history:
        st.subheader("💭 Chat History")
        for q, a in reversed(st.session_state.chat_history[-5:]):
            with st.expander(f"Q: {q}"):
                st.write(f"**Answer:** {a}")

else:
    st.info("👆 Upload a PDF file to begin!")

# Requirements
with st.expander("💻 Installation"):
    st.code("""
pip install streamlit PyPDF2 scikit-learn numpy pandas
""")
