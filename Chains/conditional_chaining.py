from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda


# Customer Feedback response system:
# Receive Feed back from customer
# Infer the feed back sentiment using LLM
# Based on the Feedback sentiment Give response

#-------Model Initiation

load_dotenv()
model = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0.7
)



#-------Prompt template 
class FeedBack(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="Give the sentiment of the feedback")

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
    template="Classify the Sentiment of the feedback text into positive or negative\n" \
    "Feedback: {feedback}\n" \
    "Format instruction: {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Write an appropriate response to the negative feedback.\nFeedback: {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to the negative feedback.\nFeedback: {feedback}",
    input_variables=["feedback"]
)


#---------Chaining

classify_chain = prompt1 | model | parser2
conditional_chain = RunnableBranch(
    (
        lambda x:x.sentiment.casefold() == "Positive".casefold(),
        prompt2 | model | parser1
    ),
    (
        lambda x:x.sentiment.casefold() == "Negative".casefold(),
        prompt3 | model | parser1
    ),
    RunnableLambda(lambda x: "could not find sentiment")
)

merge_chain = classify_chain | conditional_chain

result = merge_chain.invoke({"feedback": "This is a bad quality smartphone cant use"})

print(result)

merge_chain.get_graph().print_ascii()