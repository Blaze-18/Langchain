import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import re

# Page config
st.set_page_config(page_title="Pet Chat", page_icon="🤖", layout="centered")

# Title
st.title("Petchat AI")

# Dictionary mapping pets to their breeds
pet_breeds = {
    "Dog": ["Labrador Retriever", "German Shepherd", "Golden Retriever", "Bulldog", "Beagle"],
    "Cat": ["Siamese", "Persian", "Maine Coon", "Ragdoll", "Bengal"],
    "Turtle": ["Red-Eared Slider", "Painted Turtle", "Box Turtle", "Snapping Turtle", "Russian Tortoise"],
    "Rabbit": ["Holland Lop", "Flemish Giant", "Mini Rex", "Lionhead", "Dutch Rabbit"]
}

# selectbox for pet type
pet_type = st.selectbox("Choose your pet buddy", list(pet_breeds.keys()))
breed = st.selectbox(f"Select {pet_type} breed", pet_breeds[pet_type])
age = st.selectbox("Select Pet age between", ["0 to 3 months", "3 months to 1 year", "1 to 3 years", "4 years+"])
response_length = st.selectbox("Select response length", ["Short", "Medium", "Long"])
user_input = st.text_area("Enter your message:", height=100, placeholder="Type your concerns here..")

template = PromptTemplate(
    template="""
        Pet Care Advice Request:
        - Pet Type: {pet_type}
        - Breed: {breed}
        - Age: {age}
        - User Input: {user_input}
        - Response Length: {response_length}

        Based on this information, provide detailed care guidelines including:
        1. Explain the overall situation. why has it occured and reassure based on risk level
        2. Dietary recommendations
        3. Exercise needs
        4. Common health concerns for this breed and age
        5. Grooming requirements (if applicable)
        6. Any special considerations for this pet
        
        Focus on providing actionable and specific advice tailored to the user's input.
        If the situation is an emergency, please advise the user to contact a veterinarian immediately.
    """,
    input_variables=["pet_type", "breed", "age", "user_input", "response_length"]
)
prompt = template.invoke({
    "pet_type": pet_type,
    "breed": breed,
    "age": age,
    "user_input": user_input,
    "response_length": response_length
})

def clean_response(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    return text

# Button
if st.button("Generate Response", type="primary"):
    if user_input:
        # Model initiation
        load_dotenv()
        endpoint_config = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")
        model = ChatHuggingFace(llm=endpoint_config)
        with st.spinner("Generating response..."):
            # This is where you'll add your model
            response = model.invoke(prompt)
            final_res = clean_response(response.content)
        
        # Output section
        st.subheader("Response:")
        st.write(final_res)
    else:
        st.warning("Please enter a message first.")