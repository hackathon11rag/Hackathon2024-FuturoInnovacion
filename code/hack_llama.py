# -*- coding: utf-8 -*-
"""
hack_llama.py

This script processes vulnerability data from an Excel file
and performs text classification, translation, and sentence embedding using ChromaDB.
"""

import openpyxl
import pandas as pd
from transformers import pipeline
from chromadb import Client
from sentence_transformers import SentenceTransformer

# Load the Excel file
df = pd.read_excel('Vulnerabilidades.xlsx')

# Display the shape and a sample of the DataFrame
print(f"DataFrame shape: {df.shape}")
print(df.head())

# Display the description of the DataFrame
print(df.describe())

# Display DataFrame info
print(df.info())

# Load the text classification pipeline
classifier = pipeline("text-classification")
classification_result = classifier("we are very sad to learn about Transformer")
print("Classification result:", classification_result)

# Load the translation pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
translation_result = translator("Hola, ¿cómo estás?")
print("Translation result:", translation_result)

# Initialize ChromaDB client
chroma_client = Client()

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Extract descriptions from the DataFrame and fill NaN values
df["Description"].fillna("No especificado", inplace=True)
sentences = df["Description"].tolist()
print("Sample sentences from Description:", sentences[:5])

# Encode sentences to get embeddings
embeddings = model.encode(sentences)

# Save embeddings to ChromaDB
for idx, embedding in enumerate(embeddings):
    chroma_client.insert(
        collection_name="vulnerability_embeddings",
        document={"description": sentences[idx]},
        embedding=embedding.tolist(),  # Ensure embedding is a list
    )

# Display the embeddings
print("Sentence embeddings:")
print(embeddings)

# Optionally, retrieve and display stored embeddings from ChromaDB
stored_embeddings = chroma_client.get_all(collection_name="vulnerability_embeddings")
print("Stored embeddings in ChromaDB:")
print(stored_embeddings)
