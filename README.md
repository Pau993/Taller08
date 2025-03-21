# Arquitectura del Proyecto

Este proyecto utiliza **LangChain** y **Pinecone** para gestionar una base de datos vectorial y realizar búsquedas semánticas. La estructura principal incluye:

- **LangChain**: Para manejar modelos de lenguaje y abstracciones.
- **Pinecone**: Como base de datos vectorial para almacenar embeddings.
- **OpenAI**: Para generar embeddings y utilizar modelos de lenguaje.
- **FastAPI** o cualquier otro framework de backend opcional.

---

## Instalación y Configuración

### 1. Requisitos previos
Antes de comenzar, asegúrate de tener instalado:
- Python 3.8 o superior
- pip actualizado (`pip install --upgrade pip`)

### 2. Instalación de dependencias
Ejecuta el siguiente comando para instalar las bibliotecas necesarias:
```bash
pip install -qU langchain-openai pinecone-client
```

Si necesitas el paquete completo de LangChain:
```bash
pip install -qU "langchain[openai]"
```

### 3. Configuración de API Keys
El proyecto requiere una API key de OpenAI. Puedes configurar la clave de la siguiente manera en tu entorno:
```python
import os
os.environ["OPENAI_API_KEY"] = "tu_api_key"
```
Si no deseas almacenarla en el código, puedes ingresarla manualmente:
```python
import getpass
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
```

---

## Inicialización del Índice en Pinecone

```python
import time
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="tu_pinecone_api_key")
index_name = "langchain-test-index"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
```

---

## Manejo del Almacenamiento Vectorial

### 1. Crear documentos y añadirlos a Pinecone
```python
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from uuid import uuid4

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

documents = [
    Document(page_content="LangChain es un framework poderoso para LLMs", metadata={"source": "tweet"}),
    Document(page_content="El clima estará nublado mañana", metadata={"source": "news"}),
]
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)
```

### 2. Consultas semánticas a la base de datos
```python
results = vector_store.similarity_search("¿Cómo será el clima mañana?", k=1, filter={"source": "news"})
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```

---

## Implementación de un LLM Simple con Chat Models

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model("gpt-4o-mini", model_provider="openai")
messages = [
    SystemMessage("Traduce lo siguiente del inglés al italiano"),
    HumanMessage("Hola!"),
]
response = model.invoke(messages)
print(response.content)  # Output: 'Ciao!'
```

---

## Capturas de Pantalla o Ejecución
Puedes ejecutar estos scripts en un entorno como **Jupyter Notebook**, **Google Colab**, o un script local de Python y verificar los resultados en la consola.

Para cualquier error o ajuste, asegúrate de que las API keys están correctamente configuradas y que tienes acceso a **Pinecone** y **OpenAI**.

---

## Conclusión
Este proyecto permite manejar consultas semánticas utilizando embeddings y modelos de lenguaje, integrando Pinecone y LangChain. Puedes expandirlo para crear asistentes virtuales, sistemas de recomendación y más.

