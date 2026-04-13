from io import BytesIO
from pathlib import Path

import chromadb
import matplotlib.pyplot as plt
from PIL import Image
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

CHROMA_DIR = "./chroma_db_imagenes"
COLLECTION_NAME = "pokemon_images"

OPEN_CLIP_MODEL_NAME = "ViT-B-32"
OPEN_CLIP_CHECKPOINT = "laion2b_s34b_b79k"


def mostrar_imagen_desde_ruta(ruta: str, titulo: str):
    imagen = Image.open(ruta)
    plt.figure(figsize=(4, 4))
    plt.imshow(imagen)
    plt.title(titulo)
    plt.axis("off")
    plt.show()


def main():
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

    consulta = input("Escribe tu búsqueda de imagen: ").strip()

    resultados = collection.query(
        query_texts=[consulta],
        n_results=3
    )

    uris = resultados.get("uris", [[]])[0]
    metadatas = resultados.get("metadatas", [[]])[0]
    distances = resultados.get("distances", [[]])[0]

    if not uris:
        print("No se encontraron resultados.")
        return

    print("\nResultados encontrados:\n")

    for i, ruta in enumerate(uris):
        metadata = metadatas[i] if i < len(metadatas) else {}
        distancia = distances[i] if i < len(distances) else None

        nombre = metadata.get("filename", Path(ruta).name)

        print(f"Resultado {i + 1}")
        print(f"Archivo: {nombre}")
        print(f"Ruta: {ruta}")
        if distancia is not None:
            print(f"Distancia: {distancia}")
        print("-" * 50)

        mostrar_imagen_desde_ruta(ruta, f"{nombre}")
    

if __name__ == "__main__":
    main()