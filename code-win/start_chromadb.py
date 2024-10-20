import chromadb
from chromadb import Client
from chromadb.config import Settings

# Create a new ChromaDB client instance
client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="db"))

# Create a new collection (if you need one)
client.create_collection("my_collection")

# Start the ChromaDB server
print("Starting ChromaDB server...")
client.start()
print("ChromaDB server is running.")
