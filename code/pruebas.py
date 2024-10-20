from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from datasets import load_dataset
# from langchain.docstore.document import Document
from langchain_core.runnables.passthrough import RunnablePassthrough
import pprint

dataset = load_dataset("/home/jfr317/.llama/checkpoints/Llama3.2-1B", split='train')
                       
pprint.pprint(dataset[0])