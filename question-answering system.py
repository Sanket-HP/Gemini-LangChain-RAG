from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import ContextualRetriever
from langchain.chains.runnable import RunnableMap

# Initialize the model
model = OpenAI(model="text-davinci-003", temperature=0.7)

# Define the prompt template
prompt = PromptTemplate(
    template="Question: {question}\n\nAnswer:",
    input_variables=["question"]
)

# Initialize the retriever
retriever = ContextualRetriever()

# Define the chain
chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser

# Example usage
response = chain.invoke({"question": "What is machine learning?"})
print(response)
