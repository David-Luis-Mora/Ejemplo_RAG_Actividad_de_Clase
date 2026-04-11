from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import WebBaseLoader

CHROMA_DIR = "./chroma_db_web"

COLLECTION_NAME = "3djuegos"
COLLECTION_NAME2 = "vandal.elespanol"
COLLECTION_NAME3 = "hobbyconsolas"
COLLECTION_NAME4 = "eurogamer"

# https://www.3djuegos.com/
# https://vandal.elespanol.com/
# https://www.hobbyconsolas.com/
# https://www.eurogamer.es/


def cargar_documentos(titulo):
    docs = WebBaseLoader(query=titulo, load_max_docs=2).load()
    return docs


def partir_documentos(documentos):
    spiltter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunck_overlaop=200
    )
    chunks = spiltter.split_documents(documentos)
    return chunks



"""
Creamos los embeddings
"""
def crear_embeddings():

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large", # El modelo LLM a usar
        base_url="http://localhost:11434", # Esta es la URL de Ollama (local)
    )
    return embeddings


"""
Añadimos los embeddings a Chroma

"""
def crear_vectorstore(embeddings,chunks = None,collection_name_db = None):
    """
    Si la colección ya existe en disco, la reutiliza.
    Si no existe, indexa los documentos.
    """

    # Podéis usar este nétodo también, pero con menos control.
    # Si ya existe la colección la va a duplicar, así que
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name_db
    )

    num_docs = vectorstore._collection.count()

    if num_docs == 0:
        print("Guardamos documentos en Chroma")
        vectorstore.add_documents(chunks)
    else:
        print(f"Ya tenemos este número de documentos: {num_docs}")

    return vectorstore




titulo = "https://www.3djuegos.com/"
documento1=cargar_documentos(titulo)
documento1_cortado=partir_documentos(documento1)
embeddings = crear_embeddings()
crear_vectorstore(embeddings, documento1_cortado,COLLECTION_NAME)


titulo2 = "https://vandal.elespanol.com/"
documento2=cargar_documentos(titulo2)
documento2_cortado=partir_documentos(documento2)
embeddings = crear_embeddings()
crear_vectorstore(embeddings, documento2_cortado,COLLECTION_NAME2)

titulo3 = "https://vandal.elespanol.com/"
documento3=cargar_documentos(titulo3)
documento3_cortado=partir_documentos(documento3)
embeddings = crear_embeddings()
crear_vectorstore(embeddings, documento3_cortado,COLLECTION_NAME3)


titulo4 = "https://www.eurogamer.es/"
documento4=cargar_documentos(titulo4)
documento4_cortado=partir_documentos(documento4)
embeddings = crear_embeddings()
crear_vectorstore(embeddings, documento4_cortado,COLLECTION_NAME4)





