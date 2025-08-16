from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek/deepseek-r1:free",
    base_url='https://openrouter.ai/api/v1',

)

result = llm.invoke("Are you deep seek ?")

print(result)