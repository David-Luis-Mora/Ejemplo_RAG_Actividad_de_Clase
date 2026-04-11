from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, SystemMessage
import requests
from langchain.tools import tool
from langgraph.types import Command

CHROMA_DIR = "./chroma_db_wikipedia"
COLLECTION_NAME = "Jujutsu Kaisen"
COLLECTION_NAME2 = "Kimetsu no Yaiba"


LLM_MODEL = "gemma4:e2b"  


config = {
"configurable": {
    "thread_id": "usuario_consulta_informacion"
}
}



def crear_embeddings():

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large", # El modelo LLM a usar. Que sea el mismo con el que vectorizamos los documentos!
        base_url="http://localhost:11434", # Esta es la URL de Ollama (local)
    )
    return embeddings


def crear_retriever(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever




modelo = ChatOllama(
    model="gemma4:e4b",
    base_url="http://192.168.117.48:11434"
)
PROMPT_SISTEMA="Eres un agente que le devuelve informacion al usuario de lo que pide con lenguaje natural y bien explicado"


@tool
def obtener_datos_del_anime(promp_usuario):
    """
    El usuario te pide alguna informacion de algun anime en relacion a Jujutsu Kaisen y Kimetsu no Yaiba usa esta herramientas
    """
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=crear_embeddings(),
    )
    
    print("...Chromadb integrado...")
    print(promp_usuario)
    print("-------------------------")
    resultado = vectorstore.similarity_search_with_score(promp_usuario, k = 10)


    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME2,
        embedding_function=crear_embeddings(),
    )

    print("...Chromadb integrado...")
    print(promp_usuario)
    print("-------------------------")
    resultado2 = vectorstore.similarity_search_with_score(promp_usuario, k = 10)

    return resultado, resultado2
   

    

   

agente = create_agent(
    model=modelo,
    tools=[obtener_datos_del_anime],
    system_prompt=PROMPT_SISTEMA,
)
promppy_usuario=input("¿Que quieres preguntar al modelo?: ")

respuesta = agente.invoke(
    {
        "messages": [
            {"role": "user", "content": promppy_usuario}
        ]
    },
    config=config
)



for msg in respuesta["messages"]:
    msg.pretty_print()







