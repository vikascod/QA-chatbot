import os
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from typing import Union

load_dotenv()

google_api_key = os.environ["GOOGLE_API_KEY"]
# Create Google Palm LLM model
llm = GooglePalm(google_api_key=google_api_key, temperature=0.2)

# Initialize instructor embeddings using the Hugging Face model
model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                           model_kwargs=model_kwargs)

# Saving locally
vectordb_path = "faiss_vector"

def create_vector_db():
    # Load data from CSV
    loader = CSVLoader(file_path='data\QA_data.csv', source_column="question")
    data = loader.load()
    data = data[:150]

    # Create a FAISS instance for the vector database
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=embeddings)

    # Save the vector database locally
    vectordb.save_local(vectordb_path)

def get_qa_chain():
    # Efficient in similarity search
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_path, embeddings)

    # Retriever to query the vector database
    # Threshold value for similarity scores
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Utilizing the provided context and a given question, generate a response that draws upon the 'response' section in the source document context.
    Strive to incorporate as much text as feasible from the 'response' section without significant alterations.
    If the answer cannot be located in the context, explicitly state 'I don't know.' Avoid fabricating an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Add ConversationBufferMemory with a capacity for 5 previous conversations
    conversation_memory = ConversationBufferMemory()

    # Add ConversationBufferMemory to the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        retriever=retriever,
                                        memory=conversation_memory)

    return chain

if __name__ == "__main__":
    # create_vector_db()
    chain = get_qa_chain()
    while True:
        user_input = input("Ask a question (type 'exit' to end): ")
        if user_input.lower() == 'exit':
            break
        else:
            response = chain(user_input)
            print(response)
