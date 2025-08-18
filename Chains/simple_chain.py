from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3",task="text-generation")
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="What is the capital of {country}?",
    input_variables=["country"],
    validate_template=True
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"country": "Bangladesh"})

print(result)