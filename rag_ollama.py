"""
Sistema de Consulta Documental Inteligente
Implementación práctica de RAG con modelos locales
Dr. Alberto Verdecia Cabrera - 2025
"""
import os
import argparse
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Configuración del sistema
MODELO_EMBEDDINGS = "all-MiniLM-L6-v2"
MODELO_LLM_PREDETERMINADO = "llama3"
# ✅ CORRECTO - Usa barras normales
RUTA_DOCUMENTOS = "E:/maestria en bd y bi/Text Mining y procesamiento de lenguaje natural/tarea4/docs"

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def leer_archivos_texto(carpeta: str) -> List[Tuple[str, str]]:
    """Carga todos los archivos .txt desde la carpeta especificada"""
    documentos = []
    ruta = Path(carpeta)
    
    if not ruta.exists():
        logger.error(f"La carpeta {carpeta} no existe")
        return documentos
    
    archivos_txt = list(ruta.glob("**/*.txt"))
    
    if not archivos_txt:
        logger.warning(f"No se encontraron archivos .txt en {carpeta}")
        return documentos

    for archivo in sorted(archivos_txt):
        try:
            contenido = archivo.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            try:
                contenido = archivo.read_text(encoding="latin-1", errors="ignore").strip()
                logger.warning(f"Archivo {archivo} leído con encoding latin-1")
            except Exception as e:
                logger.error(f"Error leyendo {archivo}: {e}")
                continue
        except Exception as e:
            logger.error(f"Error inesperado leyendo {archivo}: {e}")
            continue

        if contenido:
            documentos.append((str(archivo), contenido))
            logger.info(f"Cargado: {archivo.name} ({len(contenido)} caracteres)")

    return documentos


def dividir_texto(texto: str, max_caracteres: int = 800) -> List[str]:
    """Divide el texto en fragmentos manejables"""
    if not texto or not texto.strip():
        return []
    
    lineas = [linea.strip() for linea in texto.split("\n") if linea.strip()]
    
    if not lineas:
        return []
    
    fragmentos = []
    fragmento_actual = ""

    for linea in lineas:
        if len(fragmento_actual) + len(linea) + 1 <= max_caracteres:
            fragmento_actual = (fragmento_actual + " " + linea).strip()
        else:
            if fragmento_actual:
                fragmentos.append(fragmento_actual)
            fragmento_actual = linea

    if fragmento_actual:
        fragmentos.append(fragmento_actual)

    # Asegurar que ningún fragmento sea demasiado largo
    resultado_final = []
    for fragmento in fragmentos:
        if len(fragmento) <= max_caracteres:
            resultado_final.append(fragmento)
        else:
            for i in range(0, len(fragmento), max_caracteres):
                sub_fragmento = fragmento[i:i + max_caracteres]
                if sub_fragmento.strip():
                    resultado_final.append(sub_fragmento)

    return resultado_final


class MotorBusqueda:
    """Gestiona la indexación y búsqueda de documentos"""

    def __init__(self, modelo_embeddings: str = MODELO_EMBEDDINGS):
        logger.info(f"Inicializando modelo de embeddings: {modelo_embeddings}")
        self.modelo_embeddings = SentenceTransformer(modelo_embeddings)
        self.dimension = self.modelo_embeddings.get_sentence_embedding_dimension()
        self.indice = None
        self.textos = []
        self.metadatos = []

    def construir_indice(self, documentos: List[Tuple[str, str]], tamano_fragmento: int = 800):
        """Construye el índice de búsqueda a partir de los documentos"""
        fragmentos = []
        metadatos = []

        logger.info("Dividiendo documentos en fragmentos...")
        for ruta, texto in documentos:
            fragmentos_doc = dividir_texto(texto, max_caracteres=tamano_fragmento)
            for i, fragmento in enumerate(fragmentos_doc):
                fragmentos.append(fragmento)
                metadatos.append({
                    "fuente": ruta,
                    "id_fragmento": i,
                    "longitud": len(fragmento)
                })

        if not fragmentos:
            raise ValueError("No se encontraron textos para indexar")

        logger.info(f"Generando {len(fragmentos)} embeddings...")
        embeddings = self.modelo_embeddings.encode(
            fragmentos,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )

        # Verificar que los embeddings no sean cero antes de normalizar
        embeddings = embeddings.astype('float32')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Evitar división por cero
        embeddings = embeddings / norms
        
        self.indice = faiss.IndexFlatIP(self.dimension)
        self.indice.add(embeddings)
        self.textos = fragmentos
        self.metadatos = metadatos

        logger.info(f"Índice creado exitosamente: {len(fragmentos)} fragmentos")

    def consultar(self, pregunta: str, resultados_maximos: int = 20) -> List[Dict]:
        """Realiza búsqueda de fragmentos relevantes"""
        if self.indice is None:
            raise ValueError("El índice no ha sido construido. Ejecute construir_indice() primero.")
        
        if not pregunta or not pregunta.strip():
            raise ValueError("La pregunta no puede estar vacía")
        
        embedding_pregunta = self.modelo_embeddings.encode(
            [pregunta], 
            convert_to_numpy=True
        ).astype('float32')
        
        # Normalizar
        norm = np.linalg.norm(embedding_pregunta)
        if norm > 0:
            embedding_pregunta = embedding_pregunta / norm

        # Asegurar que no se soliciten más resultados de los disponibles
        k = min(resultados_maximos, len(self.textos))
        similitudes, indices = self.indice.search(embedding_pregunta, k)

        resultados = []
        for similitud, indice in zip(similitudes[0], indices[0]):
            if indice < len(self.textos):  # Validación adicional
                resultados.append({
                    "similitud": float(similitud),
                    "texto": self.textos[indice],
                    "metadatos": self.metadatos[indice]
                })

        return resultados


def ejecutar_ollama(modelo: str, instruccion: str, tiempo_espera: int = 60) -> str:
    """Ejecuta el modelo de lenguaje local a través de la CLI de Ollama"""
    try:
        comando = ["ollama", "run", modelo]
        proceso = subprocess.run(
            comando,
            input=instruccion,
            capture_output=True,
            text=True,
            timeout=tiempo_espera,
            encoding='utf-8'
        )

        if proceso.returncode != 0:
            error_msg = proceso.stderr.strip()
            raise RuntimeError(f"Error en Ollama: {error_msg}")

        return proceso.stdout.strip()

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout: el modelo tardó más de {tiempo_espera} segundos")
    except FileNotFoundError:
        raise RuntimeError(
            "No se encontró el comando 'ollama'. "
            "Verifique que Ollama esté instalado y en el PATH del sistema."
        )
    except Exception as e:
        raise RuntimeError(f"Error ejecutando Ollama: {str(e)}")


PLANTILLA_INSTRUCCION = """
Como asistente especializado, analice la siguiente información recuperada de documentos 
y responda la pregunta del usuario de manera precisa. 

Si utiliza información específica de algún documento, indique la fuente correspondiente.
Si la información necesaria para responder no se encuentra en los documentos proporcionados, 
responda claramente indicando que no cuenta con esa información.

DOCUMENTOS RECUPERADOS:
{documentos}

PREGUNTA DEL USUARIO:
{pregunta}

RESPUESTA:
"""


def construir_instruccion_rag(
        resultados: List[Dict],
        pregunta: str,
        max_caracteres_por_documento: int = 800
) -> str:
    """Construye la instrucción completa para el modelo de lenguaje"""
    textos_documentos = []

    for idx, resultado in enumerate(resultados, 1):
        fuente = resultado["metadatos"].get("fuente", "desconocida")
        # Extraer solo el nombre del archivo
        nombre_archivo = Path(fuente).name
        texto = resultado["texto"][:max_caracteres_por_documento].replace("\n", " ")
        similitud = resultado.get("similitud", 0.0)
        
        textos_documentos.append(
            f"[Documento {idx}] FUENTE: {nombre_archivo} (relevancia: {similitud:.2f})\n"
            f"CONTENIDO: {texto}"
        )

    documentos_unidos = "\n\n---\n\n".join(textos_documentos)

    return PLANTILLA_INSTRUCCION.format(
        documentos=documentos_unidos,
        pregunta=pregunta
    )


def main():
    """Función principal del sistema"""
    parser = argparse.ArgumentParser(
        description="Sistema de consulta documental inteligente con RAG"
    )
    parser.add_argument(
        "--modelo",
        default=MODELO_LLM_PREDETERMINADO,
        help="Modelo de Ollama a utilizar (ej: llama3, mistral)"
    )
    parser.add_argument(
        "--resultados",
        type=int,
        default=4,
        help="Número de fragmentos a recuperar para el contexto"
    )
    parser.add_argument(
        "--docs",
        default=RUTA_DOCUMENTOS,
        help="Ruta a la carpeta de documentos"
    )

    args = parser.parse_args()

    try:
        # Cargar y validar documentos
        print(f"\n{'='*70}")
        print(f"Cargando documentos desde: {args.docs}")
        print(f"{'='*70}\n")
        
        documentos = leer_archivos_texto(args.docs)

        if not documentos:
            print(f"\n❌ Error: No se encontraron archivos .txt en '{args.docs}'")
            print("   Asegúrese de que:")
            print("   1. La carpeta existe")
            print("   2. Contiene archivos con extensión .txt")
            print("   3. Los archivos tienen contenido")
            return

        print(f"\n✓ Se cargaron {len(documentos)} documentos correctamente\n")

        # Construir índice de búsqueda
        motor_busqueda = MotorBusqueda()
        motor_busqueda.construir_indice(documentos)

        # Bucle principal de interacción
        print("\n" + "=" * 70)
        print(f"SISTEMA DE CONSULTA DOCUMENTAL - Modelo: {args.modelo}")
        print("Escriba su pregunta o 'salir' para terminar")
        print("=" * 70)

        while True:
            try:
                pregunta_usuario = input("\n💬 CONSULTA > ").strip()

                if pregunta_usuario.lower() in ['salir', 'exit', 'quit', 'q']:
                    print("\n👋 Finalizando sesión. ¡Hasta pronto!")
                    break

                if not pregunta_usuario:
                    continue

                # Recuperar fragmentos relevantes
                print(f"\n🔍 Buscando información relevante...")
                resultados = motor_busqueda.consultar(
                    pregunta_usuario, 
                    resultados_maximos=args.resultados
                )

                # Construir instrucción para el modelo
                instruccion_completa = construir_instruccion_rag(resultados, pregunta_usuario)

                # Ejecutar modelo de lenguaje
                print(f"🤖 Consultando modelo {args.modelo}...")
                respuesta = ejecutar_ollama(args.modelo, instruccion_completa)

                # Mostrar resultados
                print("\n" + "-" * 70)
                print("📄 RESPUESTA:")
                print(respuesta)
                print("-" * 70)

            except KeyboardInterrupt:
                print("\n\n👋 Finalizando sesión...")
                break
            except ValueError as e:
                logger.error(f"Error de validación: {e}")
                print(f"\n❌ Error: {e}")
            except RuntimeError as e:
                logger.error(f"Error de ejecución: {e}")
                print(f"\n❌ Error: {e}")
            except Exception as e:
                logger.error(f"Error inesperado: {e}", exc_info=True)
                print(f"\n❌ Error inesperado: {e}")
                break

    except Exception as e:
        logger.critical(f"Error crítico en main: {e}", exc_info=True)
        print(f"\n❌ Error crítico: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())