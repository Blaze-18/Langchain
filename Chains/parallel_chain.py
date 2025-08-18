from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model1 = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.7
)

model2 = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template="Generate a short note on {topic}",
    input_variables=['topic'],
    validate_template=True
)

prompt2 = PromptTemplate(
    template="Generate 5 simple question answers from the following topic \n {topic}",
    input_variables=['topic'],
    validate_template=True
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document\n Notes -> {note} \n Quiz -> {quiz}",
    input_variables=["note", "quiz"],
    validate_template=True
)

parser = StrOutputParser()
p_chain = RunnableParallel({
    "note": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = p_chain | merge_chain

result = chain.invoke({"topic": "Neural Network"})

print(result)
