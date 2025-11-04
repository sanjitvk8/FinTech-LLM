import streamlit as st
import requests

st.set_page_config(page_title="FinTech LLM Dashboard", layout="wide")

st.title("💹 FinTech LLM Document Processing")

st.markdown("### Upload a financial document or enter text to analyze using the backend LLM API")

# File upload section
uploaded_file = st.file_uploader("📄 Choose a file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

# Text input
user_input = st.text_area("Or, paste text content below:")

# API endpoint (your backend should expose this)
API_URL = "http://127.0.0.1:8000/process"  # change this if your backend runs elsewhere

if st.button("🚀 Process"):
    with st.spinner("Processing your request..."):
        if uploaded_file:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)
        elif user_input.strip():
            data = {"text": user_input}
            response = requests.post(API_URL, json=data)
        else:
            st.warning("Please upload a file or enter text.")
            st.stop()

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Processed Successfully!")
            st.json(result)
        else:
            st.error(f"❌ Error {response.status_code}: {response.text}")
