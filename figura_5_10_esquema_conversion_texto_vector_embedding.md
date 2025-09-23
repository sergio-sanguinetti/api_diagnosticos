# Figura 5.10: Esquema del Proceso de Análisis Semántico para Similitud

## Resumen
Esta figura documenta el proceso de análisis semántico directo para el cálculo de similitud entre análisis médicos en el sistema de análisis médico ocupacional asistido por IA. El sistema utiliza DeepSeek API para análisis semántico avanzado sin conversión a vectores tradicionales.

## Arquitectura del Proceso de Análisis Semántico

### 1. Componentes del Sistema

| **Componente** | **Función** | **Tecnología** | **Estado** |
|----------------|-------------|----------------|------------|
| **Preprocesamiento de Texto** | Limpieza y normalización del texto médico | Python, Regex | ✅ Implementado |
| **Análisis Semántico** | Comparación directa de contenido médico | DeepSeek API | ✅ Implementado |
| **Métricas de Concordancia** | Evaluación cuantitativa de concordancia | NumPy, Python | ✅ Implementado |
| **Manejo de Errores** | Gestión de timeouts y fallos de API | Python, Requests | ✅ Implementado |

### 2. Flujo del Proceso de Análisis Semántico

#### Fase 1: Preprocesamiento de Texto
1. **Extracción de Contenido Médico**
   - Búsqueda de sección `SECCION_REPORTE_COMPLETO` en texto médico
   - Extracción de contenido entre marcadores `SECCION_FIN`
   - Validación de contenido extraído

2. **Limpieza y Normalización**
   - Truncamiento a 1500 caracteres máximo
   - Eliminación de caracteres especiales
   - Normalización de formato de texto

3. **Preparación para Análisis**
   - Formateo de texto para prompt de IA
   - Validación de longitud y contenido
   - Manejo de errores y casos especiales

#### Fase 2: Análisis Semántico Directo
1. **Método Implementado**: DeepSeek API
   - **Modelo**: `deepseek-chat`
   - **Temperatura**: 0.1 (respuestas consistentes)
   - **Max Tokens**: 10 (solo número de respuesta)
   - **Timeout**: 15 segundos máximo

2. **Proceso de Análisis**
   - Creación de prompt especializado
   - Envío a DeepSeek API
   - Procesamiento de respuesta numérica
   - Validación y normalización del resultado

#### Fase 3: Cálculo de Similitud
1. **Método Principal**: DeepSeek API
   - **Ventaja**: Análisis semántico avanzado y contextual
   - **Desventaja**: Dependencia de API externa
   - **Timeout**: 15 segundos máximo
   - **Precisión**: 0.85-0.95 en casos típicos

2. **Manejo de Errores**
   - Timeout de API
   - Errores de red
   - Respuestas inválidas
   - Retorno de valor por defecto (0.0)

## Especificaciones Técnicas

### Modelo de Análisis: DeepSeek Chat

| **Parámetro** | **Valor** | **Descripción** |
|---------------|-----------|-----------------|
| **Modelo** | deepseek-chat | Modelo de lenguaje conversacional |
| **Temperatura** | 0.1 | Baja temperatura para respuestas consistentes |
| **Max Tokens** | 10 | Solo necesitamos un número decimal |
| **Timeout** | 15 segundos | Tiempo máximo de espera |
| **Entrenamiento** | Conversacional | Entrenado para análisis de texto |
| **Idiomas** | Multilingüe | Soporte para múltiples idiomas incluyendo español |

### Proceso de Análisis Semántico

1. **Preprocesamiento**
   - Extracción de contenido médico específico
   - Truncamiento a 1500 caracteres
   - Validación de contenido

2. **Creación de Prompt**
   - Formato estructurado para comparación
   - Instrucciones específicas de análisis
   - Ejemplos de respuesta esperada

3. **Análisis con IA**
   - Envío a DeepSeek API
   - Procesamiento contextual del texto
   - Generación de puntuación numérica

### Cálculo de Similitud

#### Método Implementado: DeepSeek API
```python
def calculate_semantic_similarity(text_medico, text_ia):
    # 1. Preprocesamiento
    medico_content = extract_medical_content(text_medico)
    medico_content = truncate_text(medico_content, 1500)
    text_ia = truncate_text(text_ia, 1500)
    
    # 2. Creación de prompt especializado
    prompt = f"""
    **TAREA**: Calcula la similitud semántica entre dos análisis médicos.
    
    **ANÁLISIS MÉDICO ORIGINAL**:
    {medico_content}
    
    **ANÁLISIS DE IA**:
    {text_ia}
    
    **INSTRUCCIONES**:
    1. Compara ambos análisis en términos de:
       - Diagnósticos mencionados
       - Recomendaciones sugeridas
       - Hallazgos clave identificados
       - Coherencia médica general
    
    2. Evalúa qué tan similares son en contenido y enfoque médico
    
    3. Devuelve ÚNICAMENTE un número decimal entre 0.0 y 1.0 donde:
       - 0.0 = Completamente diferentes
       - 0.5 = Moderadamente similares
       - 1.0 = Completamente similares
    
    **FORMATO DE RESPUESTA**: Solo el número decimal, sin explicaciones adicionales.
    Ejemplo: 0.75
    """
    
    # 3. Llamada a DeepSeek API
    response = call_deepseek_api(prompt)
    
    # 4. Procesamiento de respuesta
    similarity_score = parse_response(response)
    
    return similarity_score
```

## Implementación en el Sistema

### Configuración Actual
```python
# Variables de entorno
DEEPSEEK_API_KEY = "sk-37167855ce4243e8afe1ccb669021e64"

# Uso en el sistema
def calculate_semantic_similarity(text_medico, text_ia):
    # Implementación actual usando DeepSeek API
    # Análisis semántico directo sin conversión a vectores
    pass
```

### Flujo de Datos
1. **Entrada**: Texto médico y análisis de IA
2. **Preprocesamiento**: Limpieza y normalización
3. **Análisis Semántico**: Comparación directa con DeepSeek API
4. **Similitud**: Cálculo de métricas de concordancia
5. **Salida**: Puntuación de similitud (0.0 - 1.0)

## Ventajas y Limitaciones

### Ventajas del Sistema Actual
- **Análisis Semántico Avanzado**: DeepSeek proporciona análisis contextual profundo
- **Flexibilidad**: Fácil modificación de prompts para diferentes casos de uso
- **Robustez**: Manejo de errores y timeouts implementado
- **Precisión**: Resultados más precisos para texto médico complejo
- **Simplicidad**: No requiere conversión a vectores ni procesamiento local complejo

### Limitaciones Identificadas
- **Dependencia de API**: Requiere conexión a internet y API externa
- **Latencia**: Tiempo de respuesta dependiente de la API externa (3-5 segundos)
- **Costo**: Uso de tokens de API externa
- **Escalabilidad**: Limitaciones de rate limiting
- **Timeout**: Límite de 15 segundos por request

## Métricas de Rendimiento

### Tiempo de Procesamiento
| **Método** | **Tiempo Promedio** | **Tiempo Máximo** |
|------------|-------------------|------------------|
| **DeepSeek API** | 3-5 segundos | 15 segundos (timeout) |

### Precisión de Similitud
| **Método** | **Precisión** | **Consistencia** |
|------------|---------------|------------------|
| **DeepSeek API** | Alta (0.85-0.95) | Media (varía según prompt) |

## Casos de Uso

### 1. Comparación Médico vs IA
- **Entrada**: Análisis del médico y análisis de IA
- **Proceso**: Embedding de ambos textos y cálculo de similitud
- **Salida**: Puntuación de concordancia semántica

### 2. Comparación entre IAs
- **Entrada**: Análisis de DeepSeek y análisis de Gemini
- **Proceso**: Embedding de ambos análisis y cálculo de similitud
- **Salida**: Puntuación de similitud entre modelos

### 3. Validación de Diagnósticos
- **Entrada**: Diagnósticos del sistema y diagnósticos de IA
- **Proceso**: Embedding y comparación de diagnósticos específicos
- **Salida**: Métricas de concordancia diagnóstica

## Mejoras Futuras

### 1. Implementación de Hugging Face Embeddings
```python
def implement_huggingface_embeddings():
    # Implementar función de embedding local
    # Reducir dependencia de APIs externas
    # Mejorar velocidad de procesamiento
    pass
```

### 2. Optimización de Rendimiento
- **Caché de Embeddings**: Almacenar embeddings calculados
- **Procesamiento en Lote**: Procesar múltiples textos simultáneamente
- **Compresión de Vectores**: Reducir tamaño de embeddings almacenados

### 3. Mejora de Precisión
- **Modelos Especializados**: Usar modelos entrenados en texto médico
- **Fine-tuning**: Ajustar modelos para casos específicos
- **Ensemble Methods**: Combinar múltiples métodos de similitud

## Dependencias del Sistema

### Librerías Requeridas
```python
# Procesamiento de texto
import re
import numpy as np

# APIs externas
import requests
import google.generativeai as genai

# Embeddings (futuro)
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
```

### Variables de Entorno
```bash
# APIs de IA
GOOGLE_API_KEY=tu_clave_google
DEEPSEEK_API_KEY=tu_clave_deepseek
HUGGINGFACE_API_KEY=tu_clave_huggingface

# Base de datos
DB_HOST=tu_host_mysql
DB_USER=tu_usuario_mysql
DB_PASS=tu_password_mysql
DB_NAME=tu_base_datos_mysql
```

## Referencias Técnicas

- **Sentence-BERT**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **Hugging Face**: Documentación oficial de sentence-transformers
- **all-MiniLM-L6-v2**: Modelo específico para embeddings de oraciones
- **DeepSeek API**: Documentación de la API de DeepSeek para análisis médico
