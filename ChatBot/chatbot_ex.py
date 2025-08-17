from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from format_response import clean_response
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()


llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", 
    task="text-generation")
model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage(content="You are specialized therapist for children suffering from HFA")
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if(user_input) == "exit":
        break
    response = model.invoke(chat_history)
    result = clean_response(response.content)
    chat_history.append(AIMessage(content=result))
    print("AI: ", result)


print(chat_history)