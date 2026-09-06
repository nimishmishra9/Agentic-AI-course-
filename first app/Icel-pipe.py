from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

prompt = ChatPromptTemplate.from_messages([("system","You're a tutor. Answer in {limit} words."),
    ("user","{question}")
])

model = ChatOpenAI( model="gpt-4o-mini", temperature=0)

parser = StrOutputParser()

chain = prompt | model | parser

print(
    chain.invoke({
        "limit": 60,
        "question": "What is an API?"
    })
)

print("_________________________")


translate = ChatPromptTemplate.from_messages([
    ("human", "Translate the following text into Punjabi:\n\n{text}")
]) | model | parser

 
print(translate.invoke({
    "text": "I'm nimish"
}))

