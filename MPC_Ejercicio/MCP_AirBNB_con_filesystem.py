import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json
from textwrap import indent

# Función para mostrar información de cada herramienta
def pretty_tool(tool):
    print(f"\n🛠️ TOOL: {tool.name}")
    print("-" * 50)
    print("📄 Descripción:")
    print(indent(tool.description.strip(), "  "))
    print("\n📥 Argumentos (JSON Schema):")
    try:
        schema = tool.args_schema
        print(indent(json.dumps(schema, indent=2, ensure_ascii=False), "  "))
    except Exception:
        print("  No disponible")
    print("\n📌 Campos requeridos:")
    try:
        required = tool.args_schema.get("required", [])
        print(f"  {required}")
    except Exception:
        print("  No disponible")
    print("\n⚙️ Response format:")
    print(f"  {getattr(tool, 'response_format', 'N/A')}")
    print("\n🔧 Tipo:")
    print(f"  {type(tool)}")
    print("-" * 50)

async def main():

    # Configuración de ambos MCP servers
    client = MultiServerMCPClient(
        {
            "airbnb": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@openbnb/mcp-server-airbnb","--ignore-robots-txt"]
            },
            "local_mcp": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem",
                         "/home/inta@informatica.edu/Escritorio/EjemplosRAG/MPC_Ejercicio"]
            }
        }
    )

    # Descargamos herramientas de ambos servidores
    tools = await client.get_tools()

    # Mostrar herramientas
    # for tool in tools:
    #     pretty_tool(tool)

    # Función para ejecutar herramientas de Airbnb y guardar resultado
    # async def run_airbnb_tool(tool_name, args):
    #     tool = next((t for t in tools if t.name == tool_name and "airbnb" in t.name.lower()), None)
    #     if not tool:
    #         return f"No encontré la herramienta {tool_name} en Airbnb."

    #     result = await tool.run(args)

    #     # Guardamos en un txt
    #     with open("airbnb_results.txt", "a", encoding="utf-8") as f:
    #         f.write(f"\n--- Resultado {tool_name} ---\n")
    #         f.write(json.dumps(result, indent=2, ensure_ascii=False))
    #         f.write("\n---------------------------\n")

    #     return result

    # Crear agente LangChain
    agente = create_agent(
        model=ChatOllama(
            model="gemma4:26b",
            base_url="http://192.168.117.48:11434",
            reasoning=True,
            num_ctx=16384
        ),
        tools=tools,
        system_prompt=(
    "Eres un asistente experto que llama a herramientas. Tienes acceso a dos MCP: Airbnb y un MCP local de archivos.\n\n"
    "Reglas de comportamiento:\n"
    "1. Cuando el usuario mencione 'Airbnb':\n"
    "   - Usa únicamente las herramientas del MCP de Airbnb.\n"
    "   - Devuelve solo la información más relevante: nombre de la propiedad, precio y fechas disponibles.\n"
    "   - Muestra solo las 5 mejores opciones.\n"
    "   - Explica la información en lenguaje natural en español.\n"
    "2. Si el usuario indica que quiere hacer una reserva:\n"
    "   - Usa la herramienta correspondiente para guardar los datos en un archivo .txt.\n"
    "   - El archivo se llamará info_airbnb.txt.\n"
    "   - Dentro del archivo guarda: nombre de la propiedad, precio, fechas y cualquier detalle adicional de la reserva.\n"
    "   - Confirma al usuario que la reserva se ha guardado.\n"
    "3. Para cualquier otra consulta que no sea de Airbnb, usa las herramientas del MCP local normalmente.\n"
    "4. Siempre responde en español y no preguntes nuevamente por qué herramienta usar.\n"
    "5. Si la información del MCP excede el contexto, resume los datos para mostrar solo lo esencial."
    "Guadar la informacion en la carpeta llamado airbnb"
            
        )
    )

    # Bucle principal de interacción
    while (prompt := input("> ")) != "end":
        # Detectar si la consulta es sobre Airbnb
        if "airbnb" in prompt.lower():
            # Por ejemplo, si el usuario pide buscar listados
            # Ajusta la herramienta y los argumentos según tu MCP de Airbnb
            # resultado = await run_airbnb_tool("search_listing", {"location": "Madrid"})
            # print("\n=== Airbnb Resultado ===")
            # print(resultado)
            continue

        # Si no es Airbnb, pasar al agente normal
        async for paso in agente.astream({
            "messages": [HumanMessage(prompt)]
        }, stream_mode="values"):
            ultimo_mensaje = paso["messages"][-1]

            hayRazonamiento = ""
            if hasattr(ultimo_mensaje, "additional_kwargs"):
                hayRazonamiento = ultimo_mensaje.additional_kwargs.get("reasoning_content", "")

            if hayRazonamiento:
                print("\n=== PENSANDO ===")
                print(hayRazonamiento)

            print("\n=== MENSAJE ===")
            ultimo_mensaje.pretty_print()

# Lanzar main
asyncio.run(main())