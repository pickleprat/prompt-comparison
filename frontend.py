import streamlit as st
from prompts import meta_prompt
import base64

st.set_page_config(layout="wide") 

def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def pdf_viewer_page():
    st.header("PDF Viewer")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file is not None:
        display_pdf(pdf_file)
    else:
        st.info("Please upload a PDF file to view its content.")

def rag_page():
    st.header("RAG with PDF")
    
    st.subheader("Input Prompts")
    col1, col2 = st.columns(2)
    with col1:
        normal_prompt = st.text_area("User Prompt", height=400, placeholder="Enter your prompt here...")
    with col2:
        engineered_prompt = st.text_area("Engineered Prompt", height=400, placeholder="Engineered prompt will appear here...")
    
    if normal_prompt:
        engineered_prompt = normal_prompt.strip().upper()  
        st.session_state.engineered_prompt = engineered_prompt

    st.subheader("Outputs")
    out_col1, out_col2 = st.columns(2)
    with out_col1:
        st.markdown("**Output for User Prompt:**")
        if normal_prompt:
            st.write(f"Processed output for: {normal_prompt}")
        else:
            st.write("No user prompt provided.")
    with out_col2:
        st.markdown("**Output for Engineered Prompt:**")
        if 'engineered_prompt' in st.session_state or engineered_prompt:
            st.write(f"Processed output for: {engineered_prompt}")
        else:
            st.write("No engineered prompt provided.")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("PDF Viewer", "RAG with PDF"))

    if page == "PDF Viewer":
        pdf_viewer_page()
    elif page == "RAG with PDF":
        rag_page()

if __name__ == "__main__":
    main()
