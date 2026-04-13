from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent

CHROMA_DIR = "./chroma_db_arxiv"
COLLECTION_NAME = "arxiv_rag"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "gemma4:e2b"


def crear_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )


def cargar_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=crear_embeddings(),
    )


def formatear_resultados(resultados):
    salida = []

    for i, (doc, score) in enumerate(resultados, start=1):
        titulo = doc.metadata.get("Title", "Sin título")
        autores = doc.metadata.get("Authors", "Desconocidos")
        fecha = doc.metadata.get("Published", "Desconocida")
        fuente = doc.metadata.get("Entry ID", "Sin enlace")
        contenido = doc.page_content.replace("\n", " ")

        salida.append(
            f"[Resultado {i}]\n"
            f"Título: {titulo}\n"
            f"Autores: {autores}\n"
            f"Fecha: {fecha}\n"
            f"Fuente: {fuente}\n"
            f"Score: {score:.4f}\n"
            f"Contenido: {contenido}\n"
        )

    return "\n".join(salida)


@tool
def buscar_en_arxiv(pregunta: str) -> str:
    """
    Busca información en documentos académicos descargados desde arXiv
    e indexados en una base vectorial Chroma.
    """
    vectorstore = cargar_vectorstore()
    resultados = vectorstore.similarity_search_with_score(pregunta, k=4)
    return formatear_resultados(resultados)


def main():
    modelo = ChatOllama(
        model=LLM_MODEL,
        # base_url=OLLAMA_BASE_URL
    )

    prompt_sistema = """
Eres un asistente académico.
Responde usando solo la información recuperada de la herramienta.
No inventes datos.
Si falta información, dilo claramente.
Cita siempre el título del paper y el enlace de origen si aparece.
"""

    agente = create_agent(
        model=modelo,
        tools=[buscar_en_arxiv],
        system_prompt=prompt_sistema,
    )

    pregunta = input("Haz tu pregunta: ").strip()

    respuesta = agente.invoke(
        {
            "messages": [
                {"role": "user", "content": pregunta}
            ]
        }
    )

    for msg in respuesta["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    main()