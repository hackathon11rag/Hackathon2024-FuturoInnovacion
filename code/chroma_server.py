import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv('.chroma_env')
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
        chroma_client_auth_credentials=os.getenv("CHROMA_CLIENT_AUTH_CREDENTIALS")
    )
)
chroma_client.heartbeat()
