import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from chromadb.auth.token_authn import TokenAuthenticationClientProvider

# Configuración de autenticación
auth_credentials = "test-token"
auth_provider = TokenAuthenticationClientProvider(token=auth_credentials)
token_transport_header = "X-Chroma-Token"

# Conectar al servidor remoto de ChromaDB
client = chromadb.Client(Settings(
    chroma_api_impl="chromadb.api.fastapi.FastAPI",
    chroma_server_host="127.0.0.1",
    chroma_server_http_port="8000",
    chroma_auth_token_transport_header=token_transport_header,
    chroma_auth_client_provider=auth_provider
))

# Crear una colección de ejemplo
collection = client.create_collection("text_collection")

# Inicializar el modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Puedes usar cualquier modelo de sentence-transformers

# Ejemplo de cadenas
texts = [
    "El gato está en el tejado.",
    "Me gusta programar en Python.",
    "El cielo está despejado hoy."
]

# Convertir las cadenas en vectores (embeddings)
vectors = model.encode(texts)

# Añadir las cadenas y sus vectores a la colección
collection.add(
    documents=texts,  # Las cadenas de texto originales
    embeddings=vectors,  # Los vectores generados
    ids=[str(i) for i in range(len(texts))]  # Identificadores únicos para cada documento
)

# Verificar los documentos almacenados
results = collection.get(ids=["0", "1", "2"])
print("Documentos almacenados:", results["documents"])