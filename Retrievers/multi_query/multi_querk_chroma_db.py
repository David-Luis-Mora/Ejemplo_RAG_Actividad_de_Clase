from pathlib import Path
import shutil

from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db_arxiv"
COLLECTION_NAME = "arxiv_rag"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"

QUERY = "retrieval augmented generation"


def crear_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )


def cargar_documentos_arxiv(query: str, max_docs: int = 4):
    print("Cargando documentos de arXiv...")

    loader = ArxivLoader(
        query=query,
        load_max_docs=max_docs
    )

    docs = loader.load()

    for doc in docs:
        doc.metadata["tema_busqueda"] = query

    print(f"Documentos descargados: {len(docs)}")
    return docs


def dividir_documentos(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documentos)
    print(f"Chunks generados: {len(chunks)}")
    return chunks


def crear_vectorstore(chunks, reiniciar=False):
    if reiniciar and Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=crear_embeddings(),
    )

    total = vectorstore._collection.count()
    print(f"Documentos actuales en la colección: {total}")

    if total == 0:
        print(f"Guardando {len(chunks)} fragmentos en Chroma...")
        vectorstore.add_documents(chunks)
    else:
        print(f"La colección ya existe con {total} fragmentos.")

    return vectorstore


def main():
    docs = cargar_documentos_arxiv(QUERY, max_docs=4)

    if len(docs) == 0:
        print("No se descargó ningún documento.")
        return

    chunks = dividir_documentos(docs)
    crear_vectorstore(chunks)

    print("Proceso terminado correctamente")


if __name__ == "__main__":
    main()