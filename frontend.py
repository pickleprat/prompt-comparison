import streamlit as st
from prompts import meta_prompt
from streamlit_pdf_viewer import pdf_viewer
import openai 
import dotenv 
import os 
import re 
import json 

import pymupdf4llm
import tempfile 

dotenv.load_dotenv(override=True)

st.set_page_config(layout="wide") 

model: str = "gpt-4o-mini"
OPENAI_API_KEY : str = os.getenv("OPENAI_API_KEY") 

meta_prompt: str = """
You are a prompt generation bot. Your task is to read the user's instruction and generate an ENGINEERED PROMPT that is structured for subsequent language model processing.

### DEFINITIONS ###
- VAGUE REQUEST: An imprecise or unstructured demand from the user that lacks specific formatting or detailed instructions.
- Task: A clearly defined objective that must be achieved. Rephrase the user's idea into a precise, technical task statement.
- Inputs: The data provided by the user that is necessary to complete the task. Each input should be labeled as [INPUT VALUE N] and later replaced by its name (without brackets) followed by empty curly braces (e.g. `[curly-braces]`) to allow Python f-string formatting.
- Output: The final result demonstrating that the objective has been met.
- Expert-title: A creative, domain-specific title that establishes the LLM as an expert in the relevant field.

### ENGINEERED PROMPT FORMAT ###
```
You are a [expert-title]. Your goal is to [task].

### DEFINITIONS ###
[Define all relevant terms needed for the task.]

### INSTRUCTIONS ###
[Break down the task into a clear sequence of steps for the LLM.]

### OUTPUT FORMAT ### 
* Your output should be enclosed within <output></output> tags. 
* Within the output tag should be a stringifiable JSON dictionary. 
// additional output details of how the json should be structured. 

```

### INSTRUCTIONS ###
1. Replace `expert-title` with an imaginative title that positions the LLM as a subject matter expert.
2. Rephrase the user's vague request into a clearly defined technical task.
3. In the DEFINITIONS section, explain any key terms that the LLM must understand.
4. Break the task into detailed, step-by-step instructions in the INSTRUCTIONS section.

### USER TASK ###
{}

### OUTPUT THE PROMPT SHOULD PROVIDE ### 
The output should always be in JSON dictionary.  
"""

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def extract_markdown_per_page(pdf_path):
    page_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    markdown_list = [page_chunk['text'] for page_chunk in page_chunks]
    return markdown_list

def pdf_viewer_page():
    st.header("PDF Viewer")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        with st.spinner("Converting files to markdown..."): 
            file_content = pdf_file.read() 
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            try:
                markdown_pages = extract_markdown_per_page(tmp_file_path)
                
                st.session_state.markdown_pages = markdown_pages

            finally:
                os.remove(tmp_file_path)
    else:
        st.info("Please upload a PDF file to view its content.")

def rag_page():
    st.header("RAG with PDF")
    
    st.subheader("Input Prompts")
    col1, col2 = st.columns(2)

    with col1:
        normal_prompt = st.text_area("User Prompt", height=400, placeholder="Enter your prompt here...")
    
    if normal_prompt:
        engineered_prompt = meta_prompt.format(normal_prompt)  
        response = client.chat.completions.create(
            model=model, 
            messages=[{
                "role": "user", 
                "content": engineered_prompt, 
            }], 
        ) 

        # adding code to modify prompt here
        st.session_state.engineered_prompt = response.choices[0].message.content 

    with col2:
        engineered_prompt = st.text_area("Engineered Prompt", 
                                         disabled=True, 
                                         height=400, 
                                         placeholder=st.session_state.engineered_prompt)

    st.subheader("Outputs")
    out_col1, out_col2 = st.columns(2)
    with out_col1:
        st.markdown("**Output for User Prompt:**")
        if normal_prompt:
            st.write(f"Processed output for: {normal_prompt}")
            with st.spinner("Normal text output..."): 
                if "markdown_pages" in st.session_state: 
                    normal_prompt = (normal_prompt + "### TEXT CONTENT ###\n" +  
                        ".".join(st.session_state.markdown_pages) ) 

                    response = client.chat.completions.create(
                        model=model, 
                        temperature=0.1, 
                        messages=[{
                            "role": "user", 
                            "content": normal_prompt, 
                        }], 
                    ) 

                    st.markdown(response.choices[0].message.content) 

        else:
            st.write("No user prompt provided.")
    with out_col2:
        st.markdown("**Output for Engineered Prompt:**")
        if "markdown_pages" in st.session_state: 
            with st.spinner("Output for Engineered prompt..."): 
                response = client.chat.completions.create(
                    model=model, 
                    temperature=0.1, 
                    messages=[{
                        "role": "user", 
                        "content": engineered_prompt, 
                    }], 
                ) 
                try: 
                    engineered_prompt = (st.session_state.engineered_prompt + "### PDF CONTENT###\n" + ".".join(st.session_state.markdown_pages)) 

                    response_content = response.choices[0].message.content
                    if re.findall(r"<output>(.*?)</output>", response_content, re.DOTALL): 
                        json_content : str = re.findall(r"<output>(.*?)</output>", response_content, re.DOTALL)[0]
                        if json_content.startswith("```json"): 
                            json_content = json_content.split("```json")[1].split("```")[0]

                        js_dict : dict = json.loads(json_content) 
                        st.json(js_dict) 
                except Exception as e: 
                    st.markdown(response.choices[0].message.content)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("PDF Viewer", "RAG with PDF"))
    if ("engineered_prompt" not in st.session_state) : 
        st.session_state.engineered_prompt = "Engineered prompt will appear here..."

    if page == "PDF Viewer":
        pdf_viewer_page()
    elif page == "RAG with PDF":
        rag_page()

if __name__ == "__main__":
    main()
