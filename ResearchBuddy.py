import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import re
from langchain_core.prompts import PromptTemplate

st.title("Research Buddy")

#user_input = st.text_area("Enter your message:", height=100, placeholder="Type your message here...")
paper_input = st.selectbox("Select Research paper", 
                        ["Aerospace and Electronic Systems Magazine",
                        "Learning to Drift",
                        "A Lightweight LiDAR-Based AAV",
                        "Dual-Branch Structure-Aware",
                        "Learning Autonomous"
                        ])
style_input = st.selectbox("Select Explanation Style", [
    "Beginner Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short", "Medium", "Long"])

def format_response(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    text = re.sub(r'\[\s*(.*?)\s*\]', r'$$\1$$', text, flags=re.DOTALL)
    return text
# Template 
template = PromptTemplate(
    template="""
    Please summarize the research paper titled "{paper_input}" with the following specifications:
    Explanation Style: {style_input}  
    Explanation Length: {length_input}  
    1. Mathematical Details:  
    - Include relevant mathematical equations if present in the paper.  
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
    2. Analogies:  
    - Use relatable analogies to simplify complex ideas.  
    If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
    Ensure the summary is clear, accurate, and aligned with the provided style and length.
    """,
    input_variables=['paper_input', 'style_input', "length_Input"]

)

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

load_dotenv()
endpoint_config = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")
model = ChatHuggingFace(llm=endpoint_config)

if st.button("Summerize"):
    with st.spinner("Generating response..."):
        result = model.invoke(prompt)
        clean_response = format_response(result.content)
        st.write(clean_response)
