import os
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

google_api_key = os.environ["GOOGLE_API_KEY"]
# Create Google Palm LLM model
llm = GooglePalm(google_api_key=google_api_key, temperature=0.1)

# # Initialize instructor embeddings using the Hugging Face model
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                           model_kwargs=model_kwargs)
vectordb_path = "faiss_vector"

def create_vector_db():
    # Load data from CSV
    loader = CSVLoader(file_path='data\QA_data.csv', source_column="question")
    data = loader.load()

    # Create a FAISS instance for vector database
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_path)


def get_qa_chain():
    # Loading the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_path, embeddings)

    # Retriever for query the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Utilizing the provided context and a given question, generate a response that draws upon the 'response' section in the source document context.
    Strive to incorporate as much text as feasible from the 'response' section without significant alterations.
    If the answer cannot be located in the context, explicitly state 'I don't know.' Avoid fabricating an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    # create_vector_db()
    chain = get_qa_chain()
    while True:
        user_input = input("Ask a question (type 'exit' to end): ")
        if user_input.lower() == 'exit':
            break
        else:
            print(chain(user_input))