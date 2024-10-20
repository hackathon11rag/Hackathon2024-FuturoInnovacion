from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser

# dataset = load_dataset("/home/jfr317/.llama/checkpoints/Llama3.2-1B")
# documents = []
# for row in dataset:
#    documents.append(Document(page_content=(row['description']), 
#    metadata={'product_name': row['product_name']}
    ))
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
["How do I get a replacement Medicare card?",
 "What is the monthly premium for Medicare Part B?",
 "How do I terminate my Medicare Part B (medical insurance)?",
 "How do I sign up for Medicare?",
 "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
 "How do I sign up for Medicare Part B if I already have Part A?",
 "What are Medicare late enrollment penalties?",
 "What is Medicare and who can get it?",
 "How can I get help with my Medicare Part A and Part B premiums?",
 "What are the different parts of Medicare?",
 "Will my Medicare premiums be higher because of my higher income?",
 "What is TRICARE ?",
 "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]
processed_documents = text_splitter.split_documents()

embedder = HuggingFaceEmbeddings(model_name="/home/jfr317/.llama/checkpoints/Llama3.2-1B")
vector_store = FAISS.from_documents(processed_documents, embedder)
retriever = vector_store.as_retriever()

general_prompt = """You are an assistant specialized in technology and cibersecurity. Given a question about the subjects, you have to answer it in the simplest way possible, in order to even people who doesn't have much background in IT can comproehend. Also, you will try to give an answer which will intend to be the least expensive posibble but still feasible to solve the problem if there is one

Question: {input}

Answer: """

prompt = PromptTemplate.from_template(general_prompt)

llm = HuggingFacePipeline.from_model_id(
    model_id="/home/jfr317/.llama/checkpoints/Llama3.2-1B",
    task="chatbot"
)

def format_docs(docs)
    return "\n\n".join(doc.page_content for doc in docs)
generator_chain = (
    RunnablePassthrough.assign(input=(lambda x: format_docs(x[""])))
    | prompt
    | llm
    | StrOutputParser()
)

retrieve_dcos = ()

# /home/jfr317/.llama/checkpoints/Llama3.2-1B