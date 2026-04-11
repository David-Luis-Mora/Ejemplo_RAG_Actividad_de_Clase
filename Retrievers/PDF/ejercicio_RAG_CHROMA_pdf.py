from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


CHROMA_DIR = "./chroma_db_pdf"
COLLECTION_NAME = "recetario_canario"
COLLECTION_NAME2 = "vegano_sin_indice"



def cargar_documentos(fichero):
    loader= PyPDFLoader(fichero)
    documentos=loader.load()
    
    return documentos


def partir_documentos(documentos):
    spiltter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        # chunck_overlaop=200,
        chunk_overlap=200,
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
def crear_vectorstore(embeddings,chunks,collection_name_db):
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
        collection_name= collection_name_db
    )

    num_docs = vectorstore._collection.count()

    if num_docs == 0:
        print("Guardamos documentos en Chroma")
        vectorstore.add_documents(chunks)
    else:
        print(f"Ya tenemos este número de documentos: {num_docs}")

    return vectorstore




file_path = "../../ficheros/recetario_canario.pdf"
documento1=cargar_documentos(file_path)
documento1_cortado=partir_documentos(documento1)
embeddings = crear_embeddings()
crear_vectorstore(embeddings, documento1_cortado, COLLECTION_NAME)


file_path = "../../ficheros/vegano_sin_indice.pdf"
documento2=cargar_documentos(file_path)
documento2_cortado=partir_documentos(documento2)
embeddings = crear_embeddings()
crear_vectorstore(embeddings, documento2_cortado,COLLECTION_NAME2)