from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = GoogleGenerativeAI(model="models/gemini-1.5-pro", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.7)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path="DataSet.csv", source_column='prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever()
    prompt_template = """Given the following context and a question, generate an answer based on the context provided.
    In the answer, try to include as much text as possible from the "response" section of the source document context without altering it significantly.
    If the answer cannot be found in the context, kindly state "I don't know." Do not make up any answer or provide information not mentioned in the context.

    CONTEXT: {context}

    QUESTION: {question}

    ANSWER (if available, based strictly on context):"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    
    chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs)
    
    return chain

if __name__ == "__main__":
    # create_vector_db()
    chain = get_qa_chain()
    # print(chain("Do you have javascript course?"))
    
