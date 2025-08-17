from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Model Initailiaztion

messages = [
    SystemMessage(content="You are a professional Therapist for children suffering from HFA"),
    HumanMessage(content="I have anexiaty when someone appreciates me. What should I do")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
