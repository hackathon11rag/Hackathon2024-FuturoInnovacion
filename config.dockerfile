docker run  -p 8000:8000  -e CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER="chromadb.auth.token.TokenConfigServerAuthCredentialsProvider"  -e CHROMA_SERVER_AUTH_PROVIDER="chromadb.auth.token.TokenAuthServerProvider"  -e CHROMA_SERVER_AUTH_TOKEN_TRANSPORT_HEADER="X_CHROMA_TOKEN"  -e CHROMA_SERVER_AUTH_CREDENTIALS="test-token"  -v /path/to/local/db/:/chroma/chroma  chromadb/chroma


