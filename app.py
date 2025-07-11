import os
import streamlit as st
import streamlit.components.v1 as components

from utils import load_pdf, split_text
from retriever import create_vector_store
from chains import create_qa_chain

from langchain_core.documents import Document

# Load API key from .env
openai_api_key = st.secrets["api_keys"]["openai"]
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file!")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Streamlit App
st.set_page_config(page_title="📚 RAG Q&A App", layout="wide")
st.title("📄 PDF Q&A Chatbot")

# Custom CSS for enhanced text area, label, and hidden submit button
st.markdown(
    """
    <style>
    /* Container for the text area to control width and centering */
    .text-area-container {
        max-width: 800px; /* Constrain width for better readability */
        margin: 0 auto; /* Center the text area */
        padding: 10px;
    }
    /* Style the text area */
    .stTextArea textarea {
        font-size: 18px !important; /* Larger, readable font */
        font-weight: 500 !important; /* Subtle bold for clarity */
        color: #1A1A1A !important; /* Dark text for contrast */
        background-color: #F5F8FF !important; /* Soft blue background */
        border: 2px solid #4A90E2 !important; /* Vibrant blue border */
        border-radius: 12px !important; /* Rounded corners */
        padding: 15px !important; /* Spacious padding */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important; /* Subtle shadow */
        transition: all 0.3s ease !important; /* Smooth hover effect */
    }
    /* Hover effect for interactivity */
    .stTextArea textarea:hover {
        border-color: #357ABD !important; /* Darker blue on hover */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important; /* Enhanced shadow */
    }
    /* Style the label */
    .stTextArea > label {
        font-size: 22px !important; /* Larger, prominent label */
        font-weight: 700 !important; /* Bold label */
        color: #1A3C7A !important; /* Dark blue for professionalism */
        margin-bottom: 10px !important; /* Space below label */
    }
    /* Ensure placeholder text is clear */
    .stTextArea textarea::placeholder {
        color: #6B7280 !important; /* Gray placeholder text */
        font-weight: 400 !important; /* Normal weight for placeholder */
        opacity: 0.7 !important; /* Slightly transparent */
    }
    /* Hide the submit button */
    .hidden-submit-button {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# JavaScript to trigger form submission on Enter key (without Ctrl/Shift/Alt)
components.html(
    """
    <script>
    document.addEventListener('keydown', function(event) {
        const textarea = document.querySelector('textarea');
        if (event.key === 'Enter' && !event.ctrlKey && !event.shiftKey && !event.altKey && textarea) {
            event.preventDefault(); // Prevent newline in textarea
            const button = document.querySelector('.hidden-submit-button button');
            if (button) {
                button.click(); // Trigger the hidden submit button
            }
        }
    });
    </script>
    """,
    height=0  # Invisible component
)

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    try:
        st.info("📄 Loading and processing PDF...")
        text = load_pdf(uploaded_file)
        # st.success(f"✅ Loaded PDF with {len(text)} characters.")

        chunks = split_text(text)
        # st.success(f"✅ Split into {len(chunks)} chunks.")

        vector_store = create_vector_store(chunks)
        if not vector_store:
            st.error("❌ Failed to create vector store.")
        else:
            qa_chain = create_qa_chain(vector_store)
            st.success("✅ QA chain ready!")

            # Form for query input with Enter key submission
            with st.form(key="query_form"):
                with st.container():
                    query = st.text_area(
                        "💬 Ask a question about your PDF:",
                        height=200,
                        max_chars=1000,
                        placeholder="Type your question here",
                        key="query_input"
                    )
                # Wrap submit button in a div with class to hide it
                st.markdown('<div class="hidden-submit-button">', unsafe_allow_html=True)
                submit_button = st.form_submit_button("Submit")
                st.markdown('</div>', unsafe_allow_html=True)

            if submit_button and query:
                with st.spinner("🧠 Thinking..."):
                    result = qa_chain.invoke({"query": query})
                    st.markdown("### 🤖 Answer")
                    st.write(result["result"])

                    # Optional: show sources
                    if "source_documents" in result:
                        st.markdown("### 📚 Source Chunks")
                        for doc in result["source_documents"]:
                            st.markdown(f"• {doc.page_content[:300]}...")
    except Exception as e:
        st.error(f"❌ Error: {e}")
