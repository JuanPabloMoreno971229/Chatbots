import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import ConsultaBase

warnings.filterwarnings("ignore")

# 🔐 Cargar variables desde el archivo .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ No se encontró OPENAI_API_KEY en el archivo .env")

# 🤖 Configuración del modelo OpenRouter
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model_name="mistralai/mistral-7b-instruct",
    temperature=0.7,
)

def load_config(path: str) -> Dict:
    """Carga y valida el archivo de configuración JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


Meta_prompt = (
    f"{system_instruction}\n\n"
    f"-----------------------------------------------------------------"
    f"Contexto más relevante (similitud={score:.4f}):\n{best_text}\n\n"
    f"-----------------------------------------------------------------"
    f"Pregunta del usuario:\n{user_input}"
)
Memory = ""
Last_Request = "Dame un resumen de esto"


def SummaryPromt(Memory):
    try:
        prompt = "Resume el siguiente texto en un limite de 250 palabras:" + Memory
        response = llm.invoke([HumanMessage(content=prompt)])
        return response
    except Exception as e:
        return f"Error al resumir: {e}"

cfg = ConsultaBase.load_config("config.json")

while True:
    user_input = input("👤 Tú: ")
    if user_input.lower() in ["salir", "exit", "quit"]:
        print("👋 Hasta luego.")
        break

    results = ConsultaBase.query_database(cfg, user_input)
    context_texts = ""
    for item in results:
        if isinstance(item, tuple) and hasattr(item[0], "page_content"):
            context_texts += item[0].page_content + "\n"
        elif hasattr(item, "page_content"):
            context_texts += item.page_content + "\n"

    prompt_parts = [
        Meta_Promt,
        "\nContexto relevante:\n" + (context_texts if context_texts else "No se encontraron datos relevantes."),
        "\nPregunta del usuario: " + user_input,
        "\nResponde de forma clara y completa."
    ]
    prompt = "\n".join(prompt_parts)

    try:
        response = llm.invoke([HumanMessage(content=str(prompt))])

        content = ""
        try:
            content = response.content.strip()
        except AttributeError:
            if isinstance(response, dict) and "content" in response:
                content = response["content"].strip()
            elif isinstance(response, list) and len(response) > 0:
                content = response[0].content.strip()

        print(f"🤖 Bot: {content}\n")
        Memory += "\n" + content
        Memory = SummaryPromt(Memory)

    except Exception as e:
        print(f"❌ Error: {e}\n")
