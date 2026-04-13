from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

CHROMA_DIR = "./chroma_db_arxiv"
COLLECTION_NAME = "arxiv_rag"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "gemma4:e2b"

# ¿Qué es retrieval augmented generation?
# ¿Qué ventajas tiene RAG según los artículos?
# ¿Qué problemas intenta resolver RAG?
# ¿Cómo mejora RAG a los modelos de lenguaje?


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


def crear_retriever_base(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )


def crear_modelo():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )


def mostrar_documentos(documentos, titulo):
    print("\n" + "=" * 90)
    print(titulo)
    print("=" * 90)

    if not documentos:
        print("No se recuperaron documentos.")
        return

    for i, doc in enumerate(documentos, start=1):
        titulo_doc = doc.metadata.get("Title", "Sin título")
        autores = doc.metadata.get("Authors", "Desconocidos")
        fuente = doc.metadata.get("Entry ID", "Sin fuente")
        contenido = doc.page_content[:500].replace("\n", " ")

        print(f"\nDocumento {i}")
        print(f"Título: {titulo_doc}")
        print(f"Autores: {autores}")
        print(f"Fuente: {fuente}")
        print(f"Contenido: {contenido}...")


def generar_respuesta_natural(pregunta, documentos, llm):
    if not documentos:
        return "No he encontrado contexto suficiente para responder."

    contexto = []
    for i, doc in enumerate(documentos, start=1):
        titulo_doc = doc.metadata.get("Title", "Sin título")
        fuente = doc.metadata.get("Entry ID", "Sin fuente")
        contexto.append(
            f"Fragmento {i}\n"
            f"Título: {titulo_doc}\n"
            f"Fuente: {fuente}\n"
            f"Contenido: {doc.page_content}\n"
        )

    contexto_final = "\n\n".join(contexto)

    prompt = f"""
Eres un asistente académico.
Responde usando solo la información del contexto.
Si no hay información suficiente, dilo claramente.
Cita el título o la fuente cuando sea posible.

Pregunta:
{pregunta}

Contexto recuperado:
{contexto_final}
"""

    respuesta = llm.invoke(prompt)
    return respuesta.content


def main():
    vectorstore = cargar_vectorstore()
    retriever_base = crear_retriever_base(vectorstore)
    llm = crear_modelo()

    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever_base,
        llm=llm
    )

    pregunta = input("Haz tu pregunta: ").strip()

    docs_base = retriever_base.invoke(pregunta)
    mostrar_documentos(docs_base, "RESULTADOS DEL RETRIEVER BASE")

    docs_multi = multiquery_retriever.invoke(pregunta)
    mostrar_documentos(docs_multi, "RESULTADOS DEL MULTIQUERY RETRIEVER")

    print("\n" + "=" * 90)
    print("RESPUESTA FINAL CON MULTIQUERY")
    print("=" * 90)

    respuesta = generar_respuesta_natural(pregunta, docs_multi, llm)
    print(respuesta)


if __name__ == "__main__":
    main()
