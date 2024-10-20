import requests
import json
import uuid  # Importa el módulo UUID

# Define constantes
CHROMA_SERVER_URL = 'http://localhost:8000'  # Ajusta si tu servidor corre en un host/puerto diferente
AUTH_TOKEN = 'test-token'  # Usa tu token real si es diferente
DATASET_FILE = 'dataset.json'  # El archivo que contiene tus datos de embeddings
COLLECTION_NAME = "hackcollection"  # Nombre de la colección fija

#DATASET_FILE = 'demo.json'  # El archivo que contiene tus datos de embeddings
#COLLECTION_NAME = "demo2"  # Nombre de la colección fija

def load_embeddings_data():
    """Cargar embeddings desde un archivo JSON."""
    with open(DATASET_FILE, 'r') as file:
        data = json.load(file)
    return data

def clean_embeddings_data(data):
    """Limpiar los datos de embeddings."""
    for item in data:
        if 'metadata' not in item or item['metadata'] is None:
            item['metadata'] = {"placeholder": "default"}  # Establece un metadata predeterminado si no se proporciona
        
        if 'embedding' not in item or item['embedding'] is None:
            item['embedding'] = [0.0]  # Establece un embedding predeterminado si no se proporciona
            
        if 'document' not in item or item['document'] is None:
            item['document'] = "No document provided"  # Proporciona un documento predeterminado si no se da
            
    return data

def ensure_unique_embeddings(data):
    """Asegurar que los embeddings sean únicos."""
    embeddings_set = set()
    for item in data:
        embedding_tuple = tuple(item['embedding'])  # Convertir a tupla para inmutabilidad
        if embedding_tuple in embeddings_set:
            item['embedding'] = [0.0]  # Establecer a predeterminado si es duplicado
        else:
            embeddings_set.add(embedding_tuple)
    return data


def list_collections():
    """Listar todas las colecciones existentes."""
    url = f"{CHROMA_SERVER_URL}/api/v1/collections"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()  # Devolver la lista de colecciones
    else:
        return None  # Retornar None si hay un error

def delete_collection(collection_name):
    """Eliminar una colección específica."""
    url = f"{CHROMA_SERVER_URL}/api/v1/collections/{collection_name}"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    
    response = requests.delete(url, headers=headers)
    return response

def delete_all_collections():
    """Eliminar todas las colecciones existentes."""
    collections = list_collections()
    
    if collections is None:
        print("Failed to retrieve collections.")
        return
    
    for collection in collections:
        collection_name = collection['name']
        response = delete_collection(collection_name)
        
        if response.status_code == 200:
            print(f"Collection '{collection_name}' deleted successfully.")
        else:
            print(f"Failed to delete collection '{collection_name}': {response.status_code}, Response: {response.json()}")


def collection_exists(collection_name):
    """Verificar si una colección existe."""
    collections = list_collections()
    if collections is not None:
        for collection in collections:
            if collection['name'] == collection_name:
                return True
    return False

def create_collection(collection_name):
    """Crear una nueva colección y devolver su ID."""
    url = f"{CHROMA_SERVER_URL}/api/v1/collections"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "name": collection_name  # Proporcionar el nombre de la colección
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        return response.json()['id'], response  # Devolver el ID de la colección y la respuesta
    else:
        return None, response  # Retornar None y la respuesta si hay un error

def add_embeddings(collection_id, embeddings_data):
    """Enviar solicitud POST para agregar embeddings a la colección."""
    url = f"{CHROMA_SERVER_URL}/api/v1/collections/{collection_id}/add"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Envío de la carga útil correctamente estructurada
    payload = {
        "embeddings": [item['embedding'] for item in embeddings_data],
        "metadatas": [item['metadata'] for item in embeddings_data],
        "documents": [item['document'] for item in embeddings_data],
        "ids": [str(i) for i in range(len(embeddings_data))]
    }

    response = requests.post(url, headers=headers, json=payload)
    return response





def main():
    # Cargar y limpiar datos de embeddings
    embeddings_data = load_embeddings_data()
    embeddings_data = clean_embeddings_data(embeddings_data)
    embeddings_data = ensure_unique_embeddings(embeddings_data)
    
    # Verificar si la colección existe antes de crearla
    if collection_exists(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} already exists.")
        collection_id = next((c['id'] for c in list_collections() if c['name'] == COLLECTION_NAME), None)
    else:
        # Crear la colección
        collection_id, create_response = create_collection( COLLECTION_NAME)
        if collection_id:
            print(f"Collection {COLLECTION_NAME} created successfully with ID: {collection_id}.")
        else:
            print(f"Failed to create collection: {create_response.status_code}, Response: {create_response.json()}")
            return  # Si falla la creación, salir

    # Enviar la solicitud para agregar embeddings
    response = add_embeddings(collection_id, embeddings_data)
    
    if response.status_code == 201:
        print("Embeddings added successfully.")
    else:
        print(f"Failed to add embeddings: {response.status_code}, Response: {response.json()}")


if __name__ == "__main__":
    main()
