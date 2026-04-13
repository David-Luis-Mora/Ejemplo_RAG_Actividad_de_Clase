from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# Carpeta donde tienes las imágenes
IMAGE_FOLDER = "./pokemon_images"

# Directorio persistente de Chroma
CHROMA_DIR = "./chroma_db_imagenes"

# Nombre de la colección
COLLECTION_NAME = "pokemon_images"

# Modelo OpenCLIP más ligero que ViT-H-14
OPEN_CLIP_MODEL_NAME = "ViT-B-32"
OPEN_CLIP_CHECKPOINT = "laion2b_s34b_b79k"


def obtener_rutas_imagenes(carpeta: str):
    extensiones_validas = {".png", ".jpg", ".jpeg", ".webp"}
    rutas = []

    for fichero in Path(carpeta).iterdir():
        if fichero.is_file() and fichero.suffix.lower() in extensiones_validas:
            rutas.append(str(fichero.resolve()))

    return sorted(rutas)


def main():
    rutas_imagenes = obtener_rutas_imagenes(IMAGE_FOLDER)

    if not rutas_imagenes:
        print("No se encontraron imágenes en la carpeta.")
        return

    print(f"Imágenes encontradas: {len(rutas_imagenes)}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    image_loader = ImageLoader()

    embedding_function = OpenCLIPEmbeddingFunction(
        model_name=OPEN_CLIP_MODEL_NAME,
        checkpoint=OPEN_CLIP_CHECKPOINT
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        data_loader=image_loader
    )

    # Evitamos duplicados simples comprobando si ya hay algo
    existentes = collection.count()
    print(f"Documentos actuales en la colección: {existentes}")

    if existentes == 0:
        ids = [f"img_{i}" for i in range(len(rutas_imagenes))]
        metadatas = [
            {
                "filename": Path(ruta).name,
                "categoria": "pokemon"
            }
            for ruta in rutas_imagenes
        ]

        collection.add(
            ids=ids,
            uris=rutas_imagenes,
            metadatas=metadatas
        )

        print("Imágenes indexadas correctamente.")
        print(f"Total final en colección: {collection.count()}")
    else:
        print("La colección ya tenía imágenes indexadas.")


if __name__ == "__main__":
    main()