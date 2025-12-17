🔍 Sistema de Consulta Documental Inteligente (RAG con Ollama)
Implementación práctica de Retrieval-Augmented Generation (RAG) con modelos de lenguaje locales para consultas documentales automatizadas.

📋 Descripción
Este proyecto implementa un sistema completo de RAG (Retrieval-Augmented Generation) que permite realizar consultas inteligentes sobre documentos de texto. Combina embeddings semánticos con modelos de lenguaje locales (via Ollama) para proporcionar respuestas precisas basadas en el contenido de los documentos.

🚀 Características Principales
🔍 Búsqueda Semántica: Indexación de documentos usando embeddings (all-MiniLM-L6-v2)

🤖 Modelos Locales: Integración con Ollama para ejecutar modelos LLM localmente (Llama3, Mistral, etc.)

📚 Procesamiento Multilingüe: Soporte para documentos en español con manejo robusto de encoding

⚡ Rápido y Eficiente: Búsqueda por similitud con FAISS para recuperación rápida

🧠 Contexto Relevante: Recupera solo los fragmentos más relevantes para cada consulta

🔧 Configurable: Parámetros ajustables para fragmentación, modelos y resultados

🛠️ Requisitos Previos
1. Instalar Python 3.8+
bash
python --version
2. Instalar Ollama
bash
# Descargar e instalar desde: https://ollama.com/
# Luego, descargar un modelo:
ollama pull llama3
ollama pull mistral
3. Instalar Dependencias de Python
bash
pip install sentence-transformers faiss-cpu numpy tqdm
📦 Instalación
Clonar o descargar el proyecto

bash
git clone <repository-url>
cd rag-ollama-system
Instalar dependencias

bash
pip install -r requirements.txt
Preparar documentos

Colocar los documentos .txt en la carpeta especificada en RUTA_DOCUMENTOS

Por defecto: E:/maestria en bd y bi/Text Mining y procesamiento de lenguaje natural/tarea4/docs

🎯 Uso
Ejecución Básica
bash
python rag_ollama.py
Opciones de Línea de Comandos
bash
# Especificar modelo diferente
python rag_ollama.py --modelo mistral

# Cambiar número de resultados
python rag_ollama.py --resultados 6

# Especificar ruta de documentos personalizada
python rag_ollama.py --docs "ruta/a/mis/documentos"
Ejemplo de Sesión Interactiva
bash
SISTEMA DE CONSULTA DOCUMENTAL - Modelo: llama3
Escriba su pregunta o 'salir' para terminar
======================================================================

💬 CONSULTA > ¿Qué es Webhomo?

🔍 Buscando información relevante...
🤖 Consultando modelo llama3...

📄 RESPUESTA:
Según Documento 1, Webhomo es una herramienta altamente calificada entre 78 herramientas 
para anotación manual de documentos en NLP y text mining.

💬 CONSULTA > salir
📁 Estructura del Proyecto
text
rag-ollama-system/
├── rag_ollama.py          # Script principal del sistema
├── requirements.txt       # Dependencias de Python
├── docs/                  # Carpeta con documentos .txt
│   ├── documento1.txt
│   ├── documento2.txt
│   └── ...
└── README.md             # Este archivo
🏗️ Arquitectura Técnica
Componentes del Sistema
Cargador de Documentos: Lee archivos .txt con manejo de encoding (UTF-8, Latin-1)

Fragmentador: Divide documentos en fragmentos manejables (~800 caracteres)

Motor de Embeddings: Usa Sentence Transformers para crear representaciones vectoriales

Índice FAISS: Almacena y busca embeddings por similitud coseno

Integración Ollama: Conecta con modelos LLM locales para generación

Sistema RAG: Combina recuperación y generación con plantilla estructurada

Flujo del Sistema
text
Documentos → Fragmentación → Embeddings → FAISS Index → Búsqueda → Contexto → Ollama → Respuesta
⚙️ Configuración
Variables Principales en el Código
python
# Modelo de embeddings (puede cambiarse por otros de Sentence Transformers)
MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"

# Modelo de Ollama por defecto
MODELO_LLM_PREDETERMINADO = "llama3"

# Ruta a documentos (ajustar según tu sistema)
RUTA_DOCUMENTOS = "ruta/a/tus/documentos"

# Tamaño máximo de fragmentos (caracteres)
tamano_fragmento = 800

# Número de fragmentos a recuperar
resultados_maximos = 4
📊 Ejemplos de Consultas
El sistema puede manejar diversos tipos de consultas:

Consultas Factuales

text
¿Cuántas herramientas fueron evaluadas en el estudio?
Consultas Comparativas

text
¿Cuáles son las diferencias entre GPT y GPT-3?
Consultas de Síntesis

text
Resume los criterios de evaluación de herramientas de anotación
Consultas Específicas

text
¿Qué tareas puede realizar GPT-3 según los documentos?
⚠️ Limitaciones Conocidas
Dependencia de Calidad de Documentos: Respuestas precisas requieren documentos bien estructurados

Tamaño de Contexto: Limitado por el tamaño de ventana del modelo LLM

Modelos Locales: Requieren hardware adecuado (CPU/GPU) para buen rendimiento

Lenguaje: Optimizado para español, pero funciona con múltiples idiomas

Validación: No incluye validación automática de hechos en respuestas

🚀 Mejoras Futuras
Interfaz web con Gradio o Streamlit

Soporte para más formatos (PDF, DOCX, HTML)

Cache de embeddings para documentos grandes

Métricas de evaluación de precisión

Soporte para múltiples modelos de embeddings

Sistema de historial de conversaciones
