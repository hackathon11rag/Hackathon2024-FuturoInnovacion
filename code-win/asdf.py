import requests
import pandas as pd
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

texts = ["How do I get a replacement Medicare card?",
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

output = query(texts)

embeddings = pd.DataFrame(output)

print(embeddings)

## dataset = load_dataset("/home/jfr317/.llama/checkpoints/Llama3.2-1B", split='train')

general_prompt1 = """Eres un asistente especializado en tecnología y ciberseguridad. Dada una pregunta sobre alguno de esos temas, la responderás de la manera más simple posible, para que incluso gente que no tiene bagaje sobre esos tópicos pueda entenderte. También tu respuesta estará deberá considerar que el usuario no tiene acceso a una gran cantidad de dinero"""
