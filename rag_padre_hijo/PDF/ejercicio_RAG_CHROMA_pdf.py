from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
# from langchain.storage import LocalFileStore
# from langchain_classic.retrievers import ParentDocumentRetriever
# from langchain_classic.storage import create_kv_docstore

from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever

CHROMA_DIR = "./chroma_db_pdf"
COLLECTION_NAME = "recetario_canario"
COLLECTION_NAME2 = "vegano_sin_indice"


def cargar_documentos(fichero):
    loader = PyPDFLoader(fichero)
    documentos = loader.load()
    return documentos


def crear_embeddings():
    return OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434",
    )


def crear_vectorstore(documentos, embeddings, collection_name_db, ruta_docstore):
    """
    Crea el retriever padre-hijo y añade los documentos.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )

    # Almacén persistente para documentos padre
    fs = LocalFileStore(ruta_docstore)
    # docstore = create_kv_docstore(fs)

    # Vectorstore persistente para chunks hijo
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name_db,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=create_kv_docstore(fs),
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # IMPORTANTE: indexar los documentos
    retriever.add_documents(documentos)

    return retriever


# -------- PDF 1 --------
file_path = "../../ficheros/recetario_canario.pdf"
documento1 = cargar_documentos(file_path)
embeddings = crear_embeddings()
retriever1 = crear_vectorstore(
    documento1,
    embeddings,
    COLLECTION_NAME,
    "./documento_papa_recetario",
)

# # -------- PDF 2 --------
# file_path = "../../ficheros/vegano_sin_indice.pdf"
# documento2 = cargar_documentos(file_path)
# embeddings = crear_embeddings()
# retriever2 = crear_vectorstore(
#     documento2,
#     embeddings,
#     COLLECTION_NAME2,
#     "./documento_papa_vegano",
# )