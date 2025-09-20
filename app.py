import streamlit as st
import requests

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="Document Ingestion", layout="wide")
st.title("Document Ingestion API Frontend")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Select chunking strategy
chunk_strategy = st.selectbox("Select chunking strategy", ["recursive", "simple"])

# Submit button
if st.button("Upload and Process"):
    if uploaded_file is not None:
        try:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {"chunk_strategy": chunk_strategy}
            
            # Send POST request to FastAPI
            response = requests.post("http://127.0.0.1:8000/upload/", files=files, data=data)
            
            if response.status_code == 200:
                res_json = response.json()
                st.success(f"Document uploaded successfully! Document ID: {res_json['doc_id']}")
                st.info(f"Number of chunks created: {res_json['num_chunks']}")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Exception occurred: {e}")
    else:
        st.warning("Please upload a file first.")
