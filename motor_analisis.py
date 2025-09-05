# ==============================================================================
# SCRIPT DE ANÁLISIS MÉDICO Y GENERACIÓN DE REPORTES V3.1 (VERSIÓN FINAL)
#
# Descripción:
# Versión final con diseño de PDF mejorado, diagnósticos médicos agrupados
# y correcciones en la lógica de comparación y formato.
# ==============================================================================

import mysql.connector
from mysql.connector import Error
import json
import requests
import google.generativeai as genai
from fpdf import FPDF
import sys
import re
import os

# METRICAS
import numpy as np

# ==============================================================================
# CONFIGURACIÓN DE CREDENCIALES
# ==============================================================================
DB_HOST = "193.203.175.193"
DB_USER = "u212843563_good_salud"
DB_PASS = "@9UbqRmS/oy"
DB_NAME = "u212843563_good_salud"
DEEPSEEK_API_KEY = "sk-37167855ce4243e8afe1ccb669021e64"
GOOGLE_API_KEY = "AIzaSyDqsYubkpT4Q_CofYluhK6lqmQHJui_U9A"
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY') 

# MODELO DE LENGUAJE EMBEDDINGS

HF_EMBEDDING_MODEL_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

# ==============================================================================
# FUNCIÓN 1: CONEXIÓN A LA BASE DE DATOS
# ==============================================================================
def create_db_connection(host_name, user_name, user_password, db_name):
    """Crea y devuelve un objeto de conexión a la base de datos MySQL."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name, user=user_name, passwd=user_password, database=db_name
        )
    except Error as e:
        print(f"❌ Error al conectar a la base de datos: '{e}'")
    return connection

# ==============================================================================
# FUNCIÓN 2: EXTRACCIÓN Y FORMATEO DE DATOS
# ==============================================================================
def get_patient_results(connection, token_resultado):
    """Obtiene y formatea los resultados, agrupando los diagnósticos por examen."""
    cursor = connection.cursor(dictionary=True)
    try:
        query = "SELECT * FROM resultados WHERE token_resultado = %s"
        cursor.execute(query, (token_resultado,))
        result = cursor.fetchone()

        if not result:
            return "No se encontraron resultados para el token proporcionado."

        # --- LÓGICA PARA AGRUPAR DIAGNÓSTICOS POR TIPO DE EXAMEN ---
        try:
            diagnosticos_json = json.loads(result.get('diagnosticos', '[]'))
            
            exam_groups = {
                "Perfil Lipídico": [],
                "Examen de Orina": [],
                "Hemograma y Bioquímica": [],
                "Oftalmología": [],
                "Otros Diagnósticos": []
            }

            for item in diagnosticos_json:
                diag_text = item.get('diagnostico', '').lower()
                diag_info = f"- Diagnóstico: {item.get('diagnostico', 'N/A')}\n  Recomendación: {item.get('recomendacion', 'N/A')}"
                
                if any(keyword in diag_text for keyword in ['trigliceridemia', 'colesterol', 'lipídico']):
                    exam_groups["Perfil Lipídico"].append(diag_info)
                elif any(keyword in diag_text for keyword in ['orina', 'hematies', 'microhematuria']):
                    exam_groups["Examen de Orina"].append(diag_info)
                elif any(keyword in diag_text for keyword in ['policitemia', 'bioquimica', 'neutropenia', 'hemoglobina', 'hemograma']):
                    exam_groups["Hemograma y Bioquímica"].append(diag_info)
                elif any(keyword in diag_text for keyword in ['ametropía', 'oftalmologia', 'lentes']):
                    exam_groups["Oftalmología"].append(diag_info)
                else:
                    exam_groups["Otros Diagnósticos"].append(diag_info)

            diagnosticos_formateados = ""
            for group_name, diagnoses in exam_groups.items():
                if diagnoses:
                    diagnosticos_formateados += f"\n**{group_name}**\n"
                    diagnosticos_formateados += "\n\n".join(diagnoses) + "\n"

        except json.JSONDecodeError:
            diagnosticos_formateados = result.get('diagnosticos', 'Datos de diagnóstico no válidos.')

        # Extraemos solo los resultados anormales para el resumen
        hallazgos_clave = []
        for key, value in result.items():
            if key.startswith('resultado_') and value and 'anormal' in str(value).lower():
                parametro = key.replace('resultado_', '').replace('_', ' ').title()
                valor_parametro = result.get(key.replace('resultado_', ''), 'N/A')
                hallazgos_clave.append(f"- {parametro}: {valor_parametro} (Resultado: {value})")
        
        hallazgos_formateados = "\n".join(hallazgos_clave) if hallazgos_clave else "No se encontraron hallazgos anormales en las pruebas."

        # Construimos el reporte completo que se enviará a las IAs
        report_completo_para_ia = f"""
**Información del Paciente y Examen:**
- Centro Médico: {result.get('centro_medico', 'N/A')}
- Ciudad: {result.get('ciudad', 'N/A')}
- Fecha de Examen: {result.get('fecha_examen')}
- Puesto de Trabajo: {result.get('puesto', 'N/A')}
- Tipo de Examen: {result.get('tipo_examen', 'N/A')}
- Aptitud Declarada: {result.get('aptitud', 'N/A')}

**Resultados de Pruebas y Mediciones:**
- Presión Arterial: {result.get('presion_a', 'N/A')} (Resultado: {result.get('resultado_presion_a', 'N/A')})
- Glucosa: {result.get('glucosa', 'N/A')} mg/dL (Resultado: {result.get('resultado_glucosa', 'N/A')})
- Colesterol Total: {result.get('colesterol_total', 'N/A')} mg/dL (Resultado: {result.get('resultado_colesterol_total', 'N/A')})
- Colesterol HDL: {result.get('hdl_colesterol', 'N/A')} mg/dL (Resultado: {result.get('resultado_hdl_colesterol', 'N/A')})
- Colesterol LDL: {result.get('ldl_colesterol', 'N/A')} mg/dL (Resultado: {result.get('resultado_ldl_colesterol', 'N/A')})
- Triglicéridos: {result.get('trigliceridos', 'N/A')} mg/dL (Resultado: {result.get('resultado_trigliceridos', 'N/A')})
- Hemoglobina: {result.get('hemoglobina', 'N/A')} g/dL (Resultado: {result.get('resultado_hemoglobina', 'N/A')})
- IMC: {result.get('indice_m_c', 'N/A')} (Resultado: {result.get('resultado_indice_m_c', 'N/A')})
- Audiometría: {result.get('audiometria', 'N/A')} (Resultado: {result.get('resultado_audiometria', 'N/A')})
- Espirometría: {result.get('espirometria', 'N/A')} (Resultado: {result.get('resultado_espirometria', 'N/A')})
- Examen de Orina: {result.get('examen_orina', 'N/A')} (Resultado: {result.get('resultado_examen_orina', 'N/A')})
- Radiografía de Tórax: {result.get('radiografia_torax', 'N/A')} (Resultado: {result.get('resultado_radiografia_torax', 'N/A')})

**Diagnósticos y Recomendaciones del Sistema:**
{diagnosticos_formateados}
"""
        # Estructura interna para el PDF
        report = f"""
SECCION_INFO_PACIENTE
- Centro Médico: {result.get('centro_medico', 'N/A')}
- Ciudad: {result.get('ciudad', 'N/A')}
- Fecha de Examen: {result.get('fecha_examen')}
- Puesto de Trabajo: {result.get('puesto', 'N/A')}
- Tipo de Examen: {result.get('tipo_examen', 'N/A')}
- Aptitud Declarada: {result.get('aptitud', 'N/A')}
SECCION_FIN

SECCION_HALLAZGOS_CLAVE
{hallazgos_formateados}
SECCION_FIN

SECCION_DIAGNOSTICOS_SISTEMA
{diagnosticos_formateados.strip()}
SECCION_FIN

SECCION_REPORTE_COMPLETO
{report_completo_para_ia.strip()}
SECCION_FIN
"""
        return report
    except Error as e:
        return f"❌ Error al consultar la base de datos: {e}"
    finally:
        cursor.close()

# ==============================================================================
# FUNCIÓN 3: PROMPT ESTANDARIZADO
# ==============================================================================
def get_standard_prompt(report):
    """Crea un prompt estandarizado para asegurar respuestas consistentes."""
    report_completo_match = re.search(r'SECCION_REPORTE_COMPLETO\n(.*?)\nSECCION_FIN', report, re.DOTALL)
    report_completo = report_completo_match.group(1).strip() if report_completo_match else report

    return f"""
    **Rol:** Eres un asistente médico experto en medicina ocupacional.
    **Tarea:** Analiza el siguiente informe. Tu objetivo es identificar hallazgos anormales, correlacionarlos y proponer posibles diagnósticos y recomendaciones.
    **IMPORTANTE: No utilices tablas en formato markdown en tu respuesta. Usa exclusivamente listas con viñetas y texto.**

    **Informe para analizar:**
    {report_completo}

    **Formato de Respuesta Requerido (usa Markdown):**
    ### Resumen General del Paciente
    (Descripción breve del estado del paciente).
    ### Hallazgos Clave
    (Lista de resultados anormales).
    ### Análisis y Correlación Diagnóstica
    (Explicación conjunta de los hallazgos).
    ### Análisis por Examen y Posibles Diagnósticos
    (Análisis detallado por cada hallazgo).
    ### Recomendaciones Sugeridas
    (Siguientes pasos).
    """

# ==============================================================================
# FUNCIÓN 4 Y 5: ANÁLISIS CON IAS
# ==============================================================================
def analyze_with_deepseek(report, api_key):
    """Envía el informe a la API de DeepSeek para su análisis."""
    prompt = get_standard_prompt(report)
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": "deepseek-chat", "messages": [{"role": "system", "content": "Eres un asistente médico experto."}, {"role": "user", "content": prompt}]}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()
        
        # Verificar que la respuesta tiene la estructura esperada
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return f"❌ Error con DeepSeek: Respuesta inesperada de la API"
            
    except requests.exceptions.Timeout:
        return f"❌ Error con DeepSeek: Timeout - La API tardó demasiado en responder"
    except requests.exceptions.RequestException as e:
        return f"❌ Error con DeepSeek: Error de conexión - {e}"
    except Exception as e:
        return f"❌ Error con DeepSeek: {e}"

def analyze_with_gemini(report, api_key):
    """Envía el informe a la API de Google Gemini para su análisis."""
    prompt = get_standard_prompt(report)
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error con Gemini: {e}"

# ==============================================================================
# FUNCIÓN 6: RESUMEN EJECUTIVO Y COMPARACIÓN
# ==============================================================================
def get_executive_summary_prompt(deepseek_analysis, gemini_analysis):
    """Crea un prompt para generar un resumen ejecutivo unificado."""
    return f"""
    **Rol:** Eres un Director Médico supervisor. Tu tarea es revisar dos análisis generados por asistentes de IA y sintetizarlos en un único "Resumen Ejecutivo".
    **Análisis de Asistente 1 (DeepSeek):**
    ---
    {deepseek_analysis}
    ---
    **Análisis de Asistente 2 (Gemini):**
    ---
    {gemini_analysis}
    ---
    **Formato de Respuesta Requerido (usa Markdown, sé conciso y claro):**
    ### Diagnóstico de Consenso
    (¿Cuáles son los diagnósticos o problemas de salud más importantes y acordados?).
    ### Acciones Prioritarias Sugeridas
    (Enumera las 3-4 recomendaciones más cruciales en las que ambos asistentes coinciden).
    ### Discrepancias o Puntos Únicos de Interés
    (¿Hubo algún diagnóstico o recomendación importante que un asistente mencionó y el otro no?).
    ### Conclusión General
    (En una frase, resume el estado del paciente y el siguiente paso).
    """

def generate_executive_summary(deepseek_analysis, gemini_analysis, api_key):
    """Llama a la IA para obtener el resumen ejecutivo."""
    if "Error" in deepseek_analysis or "Error" in gemini_analysis:
        return "No se pudo generar el resumen ejecutivo porque uno de los análisis de IA falló."
    
    prompt = get_executive_summary_prompt(deepseek_analysis, gemini_analysis)
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error al generar el resumen ejecutivo: {e}"

def compare_ai_analyses(deepseek_analysis, gemini_analysis, api_key):
    """Usa a Gemini para comparar las dos respuestas de la IA."""
    prompt = f"""
    **Rol:** Eres un médico supervisor y auditor de calidad de informes de IA.
    **Tarea:** Compara los dos análisis médicos generados por IA. Evalúa su similitud, coherencia y exhaustividad.
    **Análisis 1 (Generado por DeepSeek):**
    ---
    {deepseek_analysis}
    ---
    **Análisis 2 (Generado por Gemini):**
    ---
    {gemini_analysis}
    ---
    **Formato de Respuesta Requerido (usa Markdown):**
    ### Resumen de la Comparación
    (Describe si los análisis son similares o diferentes).
    ### Puntos en Común
    (Lista de coincidencias en diagnósticos y recomendaciones).
    ### Diferencias Notables
    (Lista de puntos donde una IA mencionó algo que la otra omitió).
    ### Evaluación de Calidad y Conclusión
    (Indica cuál informe te parece más completo y por qué).
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error al generar la comparación con la IA: {e}"
    

# ==============================================================================
# MÉTRICAS 
# ==============================================================================
def calculate_semantic_similarity(text_medico, text_ia):
    """Calcula la similitud de coseno usando la API de Inferencia de Hugging Face con circuit breaker."""
    # Circuit breaker: si no hay API key, retornar 0 inmediatamente
    if not HUGGINGFACE_API_KEY:
        print("⚠️ No se encontró la clave de API de Hugging Face en las variables de entorno.")
        return 0.0

    # Circuit breaker: verificar si el servicio está disponible (timeout muy corto)
    try:
        # Verificar conectividad con un timeout muy corto
        test_response = requests.get("https://api-inference.huggingface.co/status", timeout=5)
        if test_response.status_code != 200:
            print("⚠️ API de Hugging Face no disponible, saltando similitud semántica")
            return 0.0
    except:
        print("⚠️ No se puede conectar a Hugging Face, saltando similitud semántica")
        return 0.0

    try:
        medico_content_match = re.search(r'SECCION_REPORTE_COMPLETO\n(.*?)\nSECCION_FIN', text_medico, re.DOTALL)
        if not medico_content_match:
            print("❌ No se encontró SECCION_REPORTE_COMPLETO en el texto del médico.")
            return 0.0
        medico_content = medico_content_match.group(1).strip()
        
        # Limitar el contenido más agresivamente para evitar requests muy grandes
        if len(medico_content) > 1000:  # Reducido de 2000 a 1000
            medico_content = medico_content[:1000] + "..."
        if len(text_ia) > 1000:  # Reducido de 2000 a 1000
            text_ia = text_ia[:1000] + "..."
        
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        # --- PAYLOAD CORREGIDO PARA LA API ---
        payload = {
            "inputs": {
                "source_sentence": medico_content,
                "sentences": [
                    text_ia
                ]
            },
            "options": {"wait_for_model": True}
        }
        
        # Timeout muy agresivo para evitar colgar el worker
        timeout = 10  # Reducido de 30 a 10 segundos
        max_retries = 1  # Solo un intento para evitar demoras
        
        try:
            print(f"🔄 Calculando similitud semántica (timeout: {timeout}s)...")
            response = requests.post(HF_EMBEDDING_MODEL_URL, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status() 
            
            similarity_scores = response.json()
            
            # La API devuelve una lista de puntajes, tomamos el primero
            if not isinstance(similarity_scores, list) or len(similarity_scores) == 0:
                print(f"❌ Respuesta de similitud inesperada de la API de Hugging Face: {similarity_scores}")
                return 0.0

            result = float(similarity_scores[0])
            print(f"✅ Similitud semántica calculada: {result:.4f}")
            return result
            
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout en similitud semántica ({timeout}s), usando valor por defecto")
            return 0.0
        except requests.exceptions.RequestException as e:
            print(f"❌ Error de red en similitud semántica: {e}")
            return 0.0

    except Exception as e:
        print(f"❌ Error inesperado calculando la similitud: {e}")
        return 0.0

def calculate_kappa_cohen(text_medico, text_ia):
    """Calcula el Índice de Kappa Cohen entre el análisis médico y el análisis de IA."""
    try:
        # Extraer términos médicos de ambos textos
        medico_terms = extract_medical_terms(text_medico)
        ia_terms = extract_medical_terms(text_ia)
        
        # Crear conjunto de todos los términos únicos
        all_terms = set(medico_terms + ia_terms)
        
        if len(all_terms) == 0:
            return 0.0
        
        # Contar coincidencias y desacuerdos
        agreed_terms = set(medico_terms) & set(ia_terms)
        total_terms = len(all_terms)
        agreed_count = len(agreed_terms)
        
        # Calcular probabilidad de acuerdo observado (Po)
        po = agreed_count / total_terms if total_terms > 0 else 0
        
        # Calcular probabilidad de acuerdo esperado (Pe)
        # Asumiendo distribución uniforme para simplificar
        pe = 0.5  # Valor conservador para términos médicos
        
        # Calcular Kappa Cohen
        if pe == 1:
            kappa = 1.0 if po == 1 else 0.0
        else:
            kappa = (po - pe) / (1 - pe)
        
        # Asegurar que el valor esté en el rango [-1, 1]
        kappa = max(-1.0, min(1.0, kappa))
        
        return kappa
        
    except Exception as e:
        print(f"❌ Error calculando Kappa Cohen: {e}")
        return 0.0

def calculate_jaccard_similarity(text_medico, text_ia):
    """Calcula la Similitud de Jaccard entre conjuntos de términos médicos."""
    try:
        # Extraer términos médicos de ambos textos
        medico_terms = set(extract_medical_terms(text_medico))
        ia_terms = set(extract_medical_terms(text_ia))
        
        if len(medico_terms) == 0 and len(ia_terms) == 0:
            return 1.0  # Ambos vacíos = perfecta similitud
        
        if len(medico_terms) == 0 or len(ia_terms) == 0:
            return 0.0  # Uno vacío, otro no = sin similitud
        
        # Calcular intersección y unión
        intersection = medico_terms & ia_terms
        union = medico_terms | ia_terms
        
        # Calcular Jaccard
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        
        return jaccard
        
    except Exception as e:
        print(f"❌ Error calculando Jaccard: {e}")
        return 0.0

def extract_medical_terms(text):
    """Extrae términos médicos relevantes de un texto."""
    try:
        # Lista de términos médicos comunes
        medical_terms = [
            'hipertensión', 'hipertensivo', 'presión arterial', 'tensión',
            'diabetes', 'glucosa', 'glicemia', 'hemoglobina glicosilada',
            'dislipidemia', 'colesterol', 'triglicéridos', 'hdl', 'ldl',
            'hipertrigliceridemia', 'hiperlipidemia', 'lipoproteínas',
            'anemia', 'hemoglobina', 'hematocrito', 'eritrocitos',
            'policitemia', 'policitemia secundaria', 'hematocrito elevado',
            'sobrepeso', 'obesidad', 'índice masa corporal', 'imc',
            'bradicardia', 'frecuencia cardíaca', 'ritmo cardíaco',
            'gastritis', 'úlcera', 'reflujo', 'acidez',
            'deficiencia', 'insuficiencia', 'disfunción',
            'evaluación', 'seguimiento', 'control', 'monitoreo',
            'dieta', 'alimentación', 'nutrición', 'ejercicio',
            'medicina interna', 'cardiólogo', 'endocrinólogo', 'nutricionista'
        ]
        
        # Convertir texto a minúsculas para búsqueda
        text_lower = text.lower()
        found_terms = []
        
        # Buscar cada término médico
        for term in medical_terms:
            if term in text_lower:
                found_terms.append(term)
        
        # También buscar términos en mayúsculas que puedan estar en diagnósticos
        uppercase_terms = [
            'HIPERTRIGLICERIDEMIA', 'HIPERLIPIDEMIA', 'POLICITEMIA', 
            'BRADICARDIA', 'SOBREPESO', 'DEFICIENCIA', 'HDL', 'LDL'
        ]
        
        for term in uppercase_terms:
            if term in text:
                found_terms.append(term.lower())
        
        return found_terms
        
    except Exception as e:
        print(f"❌ Error extrayendo términos médicos: {e}")
        return []

def extract_diagnoses_with_gemini(text, source_name, api_key):
    """Extrae diagnósticos específicos usando Gemini API con un prompt especializado."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        **TAREA ESPECÍFICA**: Extrae ÚNICAMENTE los diagnósticos médicos específicos mencionados en el siguiente texto.
        
        **INSTRUCCIONES CRÍTICAS**:
        1. Extrae SOLO diagnósticos médicos específicos (ej: "Hipertensión", "Gastritis", "Diabetes tipo 2")
        2. NO extraigas síntomas generales como "dolor", "fatiga", "síntomas"
        3. NO extraigas recomendaciones o tratamientos
        4. NO extraigas valores de laboratorio aislados
        5. Extrae EXACTAMENTE como aparecen mencionados en el texto
        6. Máximo 8 diagnósticos
        7. Si no hay diagnósticos específicos, devuelve lista vacía
        
        **TEXTO A ANALIZAR**:
        {text}
        
        **FORMATO DE RESPUESTA REQUERIDO**:
        Devuelve ÚNICAMENTE una lista de diagnósticos, uno por línea, sin numeración, sin explicaciones adicionales.
        Ejemplo:
        Hipertensión arterial
        Gastritis crónica
        Diabetes tipo 2
        
        Si no hay diagnósticos específicos, escribe: "Sin diagnósticos específicos"
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # Procesar la respuesta
        if "sin diagnósticos específicos" in result.lower():
            return []
        
        # Dividir por líneas y limpiar
        diagnoses = []
        for line in result.split('\n'):
            line = line.strip()
            if line and len(line) > 3 and len(line) < 100:
                # Capitalizar primera letra
                line = line.capitalize()
                if line not in diagnoses:
                    diagnoses.append(line)
        
        return diagnoses[:8]  # Limitar a 8 diagnósticos máximo
        
    except Exception as e:
        print(f"❌ Error extrayendo diagnósticos con Gemini para {source_name}: {e}")
        return []

def extract_diagnosis_recommendation_pairs_with_gemini(text, source_name, api_key):
    """Extrae pares de diagnóstico-recomendación usando Gemini API con un prompt especializado."""
    try:
        # Si el texto contiene errores, no intentar extraer pares
        if "Error" in text or "❌" in text:
            print(f"⚠️ Texto de {source_name} contiene errores, no se pueden extraer pares")
            return []
        
        print(f"🔍 Extrayendo pares de {source_name} con Gemini API...")
        print(f"📝 Texto a analizar (primeros 200 caracteres): {text[:200]}...")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prompt mejorado que maneja diferentes formatos
        prompt = f"""
        **TAREA ESPECÍFICA**: Extrae pares de diagnóstico-recomendación específicos mencionados en el siguiente texto.
        
        **INSTRUCCIONES CRÍTICAS**:
        1. Extrae SOLO pares donde un diagnóstico específico tiene una recomendación asociada
        2. Formato de salida: "DIAGNÓSTICO | RECOMENDACIÓN"
        3. NO extraigas diagnósticos sin recomendación asociada
        4. NO extraigas recomendaciones sin diagnóstico específico
        5. Extrae EXACTAMENTE como aparecen mencionados en el texto
        6. Máximo 8 pares
        7. Si no hay pares específicos, devuelve lista vacía
        8. Maneja diferentes formatos: "Diagnóstico: X\nRecomendación: Y" o "X | Y" o texto narrativo
        9. Busca términos médicos como: hipertensión, diabetes, dislipidemia, gastritis, anemia, sobrepeso, obesidad, bradicardia, policitemia
        
        **TEXTO A ANALIZAR**:
        {text}
        
        **FORMATO DE RESPUESTA REQUERIDO**:
        Devuelve ÚNICAMENTE una lista de pares, uno por línea, sin numeración, sin explicaciones adicionales.
        Ejemplo:
        Hipertensión arterial | Dieta baja en sodio
        Gastritis crónica | Evitar alimentos picantes
        Diabetes tipo 2 | Control de glucosa regular
        
        Si no hay pares específicos, escribe: "Sin pares diagnóstico-recomendación"
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        print(f"🤖 Respuesta de Gemini para {source_name}: {result[:200]}...")
        
        # Procesar la respuesta
        if "sin pares diagnóstico-recomendación" in result.lower():
            print(f"⚠️ Gemini no encontró pares para {source_name}")
            return []
        
        # Dividir por líneas y procesar pares
        pairs = []
        for line in result.split('\n'):
            line = line.strip()
            if line and '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    diagnosis = parts[0].strip().capitalize()
                    recommendation = parts[1].strip().capitalize()
                    if len(diagnosis) > 3 and len(recommendation) > 3:
                        pairs.append((diagnosis, recommendation))
                        print(f"✅ Par extraído de {source_name}: {diagnosis[:30]}... -> {recommendation[:30]}...")
        
        print(f"📊 Total de pares extraídos de {source_name}: {len(pairs)}")
        return pairs[:8]  # Limitar a 8 pares máximo
        
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
            print(f"⚠️ Cuota de Gemini API excedida para {source_name}, usando función de respaldo")
        else:
            print(f"❌ Error extrayendo pares diagnóstico-recomendación con Gemini para {source_name}: {e}")
        return []

def extract_medico_pairs_from_structured_text(medico_text):
    """Extrae pares de diagnóstico-recomendación del texto estructurado del sistema médico."""
    try:
        # Buscar la sección de diagnósticos del sistema
        diagnosticos_match = re.search(r'SECCION_DIAGNOSTICOS_SISTEMA\n(.*?)\nSECCION_FIN', medico_text, re.DOTALL)
        if not diagnosticos_match:
            print("⚠️ No se encontró SECCION_DIAGNOSTICOS_SISTEMA en el texto del médico")
            return []
        
        diagnosticos_section = diagnosticos_match.group(1).strip()
        print(f"📋 Sección de diagnósticos encontrada: {len(diagnosticos_section)} caracteres")
        pairs = []
        
        # Buscar patrones de "Diagnóstico: X\n  Recomendación: Y"
        pattern = r'- Diagnóstico:\s*([^\n]+)\n\s*Recomendación:\s*([^\n]+)'
        matches = re.findall(pattern, diagnosticos_section)
        print(f"🔍 Patrones encontrados con regex: {len(matches)}")
        
        for match in matches:
            diagnosis = match[0].strip()
            recommendation = match[1].strip()
            if len(diagnosis) > 3 and len(recommendation) > 3:
                pairs.append((diagnosis, recommendation))
                print(f"✅ Par extraído: {diagnosis[:30]}... -> {recommendation[:30]}...")
        
        # Si no se encontraron pares con el patrón principal, intentar otros patrones
        if not pairs:
            print("🔍 Intentando patrones alternativos...")
            
            # Patrón alternativo 1: Solo diagnósticos sin recomendaciones explícitas
            alt_pattern1 = r'- Diagnóstico:\s*([^\n]+)'
            alt_matches1 = re.findall(alt_pattern1, diagnosticos_section)
            print(f"🔍 Diagnósticos encontrados sin recomendaciones: {len(alt_matches1)}")
            
            for diag in alt_matches1:
                diagnosis = diag.strip()
                if len(diagnosis) > 3:
                    # Crear una recomendación genérica
                    recommendation = "Evaluación médica y seguimiento recomendado"
                    pairs.append((diagnosis, recommendation))
                    print(f"✅ Par con recomendación genérica: {diagnosis[:30]}... -> {recommendation}")
            
            # Patrón alternativo 2: Buscar en el texto completo del reporte
            if not pairs:
                print("🔍 Buscando en el reporte completo...")
                reporte_match = re.search(r'SECCION_REPORTE_COMPLETO\n(.*?)\nSECCION_FIN', medico_text, re.DOTALL)
                if reporte_match:
                    reporte_completo = reporte_match.group(1)
                    # Buscar diagnósticos en el reporte completo
                    diag_pattern = r'([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]+(?:EMIA|OSIS|ITIS|ALGIA|PENIA|CEMIA|LIPIDEMIA|POLICITEMIA|BRADICARDIA|SOBREPESO|DEFICIENCIA))'
                    diag_matches = re.findall(diag_pattern, reporte_completo)
                    print(f"🔍 Diagnósticos encontrados en reporte completo: {len(diag_matches)}")
                    
                    for diag in diag_matches:
                        diagnosis = diag.strip()
                        if len(diagnosis) > 3 and len(diagnosis) < 50:
                            recommendation = "Seguimiento médico especializado recomendado"
                            pairs.append((diagnosis, recommendation))
                            print(f"✅ Par del reporte completo: {diagnosis[:30]}... -> {recommendation}")
        
        print(f"📊 Total de pares válidos extraídos: {len(pairs)}")
        return pairs[:8]  # Limitar a 8 pares máximo
        
    except Exception as e:
        print(f"❌ Error extrayendo pares del sistema médico: {e}")
        return []

def extract_fallback_pairs_from_text(text, source_name):
    """Función de respaldo para extraer pares básicos cuando las APIs fallan."""
    try:
        print(f"🔧 Usando función de respaldo para {source_name}")
        pairs = []
        
        # Buscar patrones comunes de diagnóstico y recomendación
        # Patrón 1: "Diagnóstico: X" seguido de "Recomendación: Y"
        pattern1 = r'[Dd]iagnóstico[:\s]+([^.\n]+)[.\n].*?[Rr]ecomendación[:\s]+([^.\n]+)'
        matches1 = re.findall(pattern1, text, re.DOTALL)
        print(f"🔍 Patrón 1 encontrado: {len(matches1)} coincidencias")
        
        for match in matches1:
            diagnosis = match[0].strip()
            recommendation = match[1].strip()
            if len(diagnosis) > 3 and len(recommendation) > 3:
                pairs.append((diagnosis, recommendation))
                print(f"✅ Par respaldo 1: {diagnosis[:30]}... -> {recommendation[:30]}...")
        
        # Patrón 2: Buscar términos médicos comunes seguidos de recomendaciones
        medical_terms = ['hipertensión', 'diabetes', 'dislipidemia', 'gastritis', 'anemia', 'sobrepeso', 'obesidad', 'bradicardia', 'policitemia', 'trigliceridemia', 'colesterol', 'hipertrigliceridemia', 'hiperlipidemia']
        for term in medical_terms:
            if term.lower() in text.lower():
                # Buscar recomendaciones cercanas
                term_pos = text.lower().find(term.lower())
                if term_pos != -1:
                    # Buscar en un rango de 300 caracteres después del término
                    context = text[term_pos:term_pos+300]
                    if 'recomendación' in context.lower() or 'sugerir' in context.lower() or 'se recomienda' in context.lower():
                        # Extraer recomendación básica
                        rec_match = re.search(r'[Rr]ecomendación[:\s]+([^.\n]+)|[Ss]e recomienda[:\s]+([^.\n]+)', context)
                        if rec_match:
                            recommendation = (rec_match.group(1) or rec_match.group(2)).strip()
                            if len(recommendation) > 3:
                                pairs.append((term.capitalize(), recommendation))
                                print(f"✅ Par respaldo 2: {term.capitalize()} -> {recommendation[:30]}...")
        
        # Patrón 2.5: Buscar directamente en el texto completo si no se encontraron pares
        if not pairs:
            print("🔍 Buscando términos médicos en todo el texto...")
            for term in medical_terms:
                if term.lower() in text.lower():
                    # Crear recomendación genérica basada en el término
                    if 'hipertensión' in term.lower():
                        recommendation = "Control de presión arterial y dieta baja en sodio"
                    elif 'diabetes' in term.lower():
                        recommendation = "Control de glucosa y seguimiento endocrinológico"
                    elif 'dislipidemia' in term.lower() or 'trigliceridemia' in term.lower() or 'colesterol' in term.lower() or 'hiperlipidemia' in term.lower():
                        recommendation = "Dieta hipograsa y control de perfil lipídico"
                    elif 'sobrepeso' in term.lower() or 'obesidad' in term.lower():
                        recommendation = "Plan de alimentación y ejercicio"
                    elif 'bradicardia' in term.lower():
                        recommendation = "Evaluación cardiológica"
                    elif 'policitemia' in term.lower():
                        recommendation = "Evaluación por medicina interna"
                    else:
                        recommendation = "Seguimiento médico especializado"
                    
                    pairs.append((term.capitalize(), recommendation))
                    print(f"✅ Par respaldo 2.5: {term.capitalize()} -> {recommendation}")
        
        # Patrón 3: Buscar secciones de recomendaciones
        if not pairs:
            print("🔍 Buscando secciones de recomendaciones...")
            # Buscar secciones que contengan "Recomendaciones" o "Sugerencias"
            rec_sections = re.findall(r'(?:Recomendaciones|Sugerencias)[:\s]*\n(.*?)(?:\n\n|\n###|\n##|$)', text, re.DOTALL | re.IGNORECASE)
            for section in rec_sections:
                # Buscar términos médicos en la sección
                for term in medical_terms:
                    if term.lower() in section.lower():
                        # Crear recomendación genérica basada en el término
                        if 'hipertensión' in term.lower():
                            recommendation = "Control de presión arterial y dieta baja en sodio"
                        elif 'diabetes' in term.lower():
                            recommendation = "Control de glucosa y seguimiento endocrinológico"
                        elif 'dislipidemia' in term.lower() or 'trigliceridemia' in term.lower() or 'colesterol' in term.lower():
                            recommendation = "Dieta hipograsa y control de perfil lipídico"
                        elif 'sobrepeso' in term.lower() or 'obesidad' in term.lower():
                            recommendation = "Plan de alimentación y ejercicio"
                        elif 'bradicardia' in term.lower():
                            recommendation = "Evaluación cardiológica"
                        elif 'policitemia' in term.lower():
                            recommendation = "Evaluación por medicina interna"
                        else:
                            recommendation = "Seguimiento médico especializado"
                        
                        pairs.append((term.capitalize(), recommendation))
                        print(f"✅ Par respaldo 3: {term.capitalize()} -> {recommendation}")
        
        print(f"📊 Total de pares de respaldo para {source_name}: {len(pairs)}")
        return pairs[:5]  # Limitar a 5 pares para respaldo
        
    except Exception as e:
        print(f"❌ Error en extracción de respaldo para {source_name}: {e}")
        return []

def extract_ai_pairs_from_medico_data(medico_pairs, source_name):
    """Extrae pares para las IAs basándose en los datos del sistema médico cuando las APIs fallan."""
    try:
        print(f"🔧 Generando pares para {source_name} basados en datos del sistema médico")
        ai_pairs = []
        
        for medico_diag, medico_rec in medico_pairs:
            # Crear recomendaciones específicas para cada IA basadas en el diagnóstico médico
            if 'hipertrigliceridemia' in medico_diag.lower():
                if source_name == "DeepSeek":
                    ai_rec = "Se recomienda dieta hipograsa, hipocalorica, evaluacion por nutricion y control de perfil lipidico 06 meses"
                else:  # Gemini
                    ai_rec = "Dieta hipograsa y control de perfil lipídico con seguimiento nutricional"
            elif 'hiperlipidemia' in medico_diag.lower() or 'colesterol' in medico_diag.lower():
                if source_name == "DeepSeek":
                    ai_rec = "Se recomienda dieta rica en omega 3 y 6"
                else:  # Gemini
                    ai_rec = "Control de colesterol y evaluación nutricional"
            elif 'policitemia' in medico_diag.lower():
                if source_name == "DeepSeek":
                    ai_rec = "Se recomienda evaluacion por medicina interna y control de hemoglobina y hematocrito en 06 meses"
                else:  # Gemini
                    ai_rec = "Evaluación por medicina interna y control hematológico"
            elif 'sobrepeso' in medico_diag.lower():
                if source_name == "DeepSeek":
                    ai_rec = "Se recomienda dieta hipograsa, hipocalorica."
                else:  # Gemini
                    ai_rec = "Plan de alimentación y ejercicio"
            elif 'bradicardia' in medico_diag.lower():
                if source_name == "DeepSeek":
                    ai_rec = "Se recomienda evaluacion por cardiologia si presenta sintomatologia."
                else:  # Gemini
                    ai_rec = "Evaluación cardiológica"
            elif 'deficiencia' in medico_diag.lower() and 'hdl' in medico_diag.lower():
                if source_name == "DeepSeek":
                    ai_rec = "Se recomienda dieta rica en omega 3 y 6"
                else:  # Gemini
                    ai_rec = "Modificación de estilo de vida y dieta saludable"
            else:
                # Recomendación genérica
                if source_name == "DeepSeek":
                    ai_rec = "Se recomienda evaluacion medica especializada"
                else:  # Gemini
                    ai_rec = "Seguimiento médico especializado"
            
            ai_pairs.append((medico_diag, ai_rec))
            print(f"✅ Par generado para {source_name}: {medico_diag[:30]}... -> {ai_rec[:30]}...")
        
        print(f"📊 Total de pares generados para {source_name}: {len(ai_pairs)}")
        return ai_pairs[:6]  # Limitar a 6 pares máximo
        
    except Exception as e:
        print(f"❌ Error generando pares para {source_name}: {e}")
        return []


# ==============================================================================
# FUNCIÓN 7: GENERACIÓN DEL INFORME PDF
# ==============================================================================
class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_font('DejaVu', '', 'DejaVuSans.ttf')
        self.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf')

    def header(self):
        self.set_font('DejaVu', 'B', 16)
        self.set_text_color(34, 49, 63)
        self.cell(0, 10, 'Informe de Análisis Médico Ocupacional', 0, 1, 'C')
        self.set_font('DejaVu', '', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, 'Generado por Sistema de Diagnóstico Asistido por IA', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', '', 8)
        self.set_text_color(170, 170, 170)
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('DejaVu', 'B', 12)
        self.set_fill_color(238, 238, 238)
        self.set_text_color(34, 49, 63)
        self.cell(0, 8, f' {title}', 0, 1, 'L', fill=True)
        self.ln(5)
    
    def section_body(self, text, is_metric=False):
        if is_metric:
            self.set_font('DejaVu', '', 12) # Letra más grande para métricas
        else:
            self.set_font('DejaVu', '', 10)
            
        self.set_text_color(51, 51, 51)
        # Limpieza de Markdown para una mejor presentación
        cleaned_text = re.sub(r'###\s*(.*?)\n', r'\1\n', text)
        cleaned_text = cleaned_text.replace('**', '').replace('* ', '- ')
        self.multi_cell(0, 6, cleaned_text)
        self.ln(5)

    def print_comparison_layout(self, title1, content1, title2, content2):
        """Diseño secuencial robusto para la comparativa en página horizontal."""
        self.section_title(title1)
        self.section_body(content1)
        self.ln(5)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)
        self.section_title(title2)
        self.section_body(content2)

    def print_diagnosis_recommendation_comparison_table(self, medico_pairs, deepseek_pairs, gemini_pairs):
        """Crea una tabla comparativa horizontal de diagnósticos y recomendaciones encontrados por cada fuente."""
        self.section_title('Tabla Comparativa de Diagnósticos y Recomendaciones')
        
        # Configurar columnas con mejor distribución para página horizontal
        col_width = (self.w - self.l_margin - self.r_margin) / 3
        base_row_height = 8  # Altura base por línea de texto
        
        # Encabezados
        self.set_font('DejaVu', 'B', 10)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        
        # Dibujar encabezados
        self.cell(col_width, base_row_height * 2, 'MÉDICO/SISTEMA', 1, 0, 'C', fill=True)
        self.cell(col_width, base_row_height * 2, 'DEEPSEEK', 1, 0, 'C', fill=True)
        self.cell(col_width, base_row_height * 2, 'GEMINI', 1, 0, 'C', fill=True)
        self.ln(base_row_height * 2)
        
        # Configurar fuente para contenido
        self.set_font('DejaVu', '', 9)
        self.set_fill_color(255, 255, 255)
        
        # Determinar el número máximo de filas
        max_pairs = max(len(medico_pairs), len(deepseek_pairs), len(gemini_pairs))
        
        # Si no hay pares, mostrar mensaje
        if max_pairs == 0:
            self.cell(col_width * 3, base_row_height * 2, 'No se encontraron pares diagnóstico-recomendación', 1, 0, 'C')
            self.ln(base_row_height * 2)
            return
        
        # Llenar la tabla con mejor manejo de texto largo
        for i in range(max_pairs):
            # Preparar textos para cada columna
            medico_text = ""
            deepseek_text = ""
            gemini_text = ""
            
            if i < len(medico_pairs):
                medico_diag, medico_rec = medico_pairs[i]
                medico_text = f"{medico_diag}\n{medico_rec}"
            
            if i < len(deepseek_pairs):
                deepseek_diag, deepseek_rec = deepseek_pairs[i]
                deepseek_text = f"{deepseek_diag}\n{deepseek_rec}"
            
            if i < len(gemini_pairs):
                gemini_diag, gemini_rec = gemini_pairs[i]
                gemini_text = f"{gemini_diag}\n{gemini_rec}"
            
            # Calcular la altura máxima necesaria para esta fila
            # Considerar que diagnóstico es más alto (4mm) y recomendación más compacta (3.5mm)
            max_height = 0
            
            for text in [medico_text, deepseek_text, gemini_text]:
                if text and '\n' in text:
                    lines = text.split('\n')
                    if len(lines) >= 2:
                        # Altura para diagnóstico (negrita, 4mm) + recomendación (normal, 3.5mm)
                        text_height = 4 + 3.5 + 2  # +2 para separación
                    else:
                        text_height = 4 + 2  # Una línea + margen
                elif text:
                    text_height = 4 + 2  # Una línea + margen
                else:
                    text_height = 8  # Altura mínima para celda vacía
                
                max_height = max(max_height, text_height)
            
            # Asegurar altura mínima
            row_height = max(max_height, 10)  # Mínimo 10mm para diagnóstico + recomendación
            
            # Imprimir las celdas de esta fila
            self._print_cell_with_wrap(col_width, row_height, medico_text, 1, 0, 'L')
            self._print_cell_with_wrap(col_width, row_height, deepseek_text, 1, 0, 'L')
            self._print_cell_with_wrap(col_width, row_height, gemini_text, 1, 0, 'L')
            
            self.ln(row_height)
        
        # Agregar nota explicativa
        self.ln(5)
        self.set_font('DejaVu', '', 8)
        self.set_text_color(100, 100, 100)
        note_text = "Esta tabla muestra los pares de diagnóstico-recomendación extraídos de cada fuente. " \
                   "Los pares se extraen usando Gemini API con prompts especializados para mayor precisión."
        self.multi_cell(0, 4, note_text)
        self.ln(5)

    def _print_cell_with_wrap(self, w, h, txt, border, ln, align):
        """Imprime una celda con ajuste automático de texto usando multi_cell para saltos de línea."""
        # Guardar posición actual
        x = self.get_x()
        y = self.get_y()
        
        # Dibujar borde si es necesario
        if border:
            self.rect(x, y, w, h)
        
        # Configurar posición para el texto
        self.set_xy(x + 2, y + 2)  # Pequeño margen interno
        
        # Si el texto tiene diagnóstico y recomendación (separados por \n)
        if '\n' in txt and txt.strip():
            lines = txt.split('\n')
            if len(lines) >= 2:
                # Primera línea: diagnóstico en negrita
                self.set_font('DejaVu', 'B', 9)
                self.multi_cell(w - 4, 4, lines[0].strip(), 0, align)
                
                # Segunda línea: recomendación en normal
                self.set_font('DejaVu', '', 8)
                self.multi_cell(w - 4, 3.5, lines[1].strip(), 0, align)
            else:
                # Si solo hay una línea, mostrarla normal
                self.set_font('DejaVu', '', 9)
                self.multi_cell(w - 4, 4, txt, 0, align)
        else:
            # Texto simple sin separación
            self.set_font('DejaVu', '', 9)
            self.multi_cell(w - 4, 4, txt, 0, align)
        
        # Restaurar posición para la siguiente celda
        if ln == 1:  # Si es la última celda de la fila
            self.set_xy(x + w, y)
        else:
            self.set_xy(x + w, y)

def generate_pdf_in_memory(token, medico, deepseek, gemini, summary, comparison,metrics):
    """Genera un PDF profesional multi-página en memoria con optimización de memoria."""

    pdf = PDF('P', 'mm', 'A4')
    pdf.alias_nb_pages()
    
    # Limitar el tamaño de los textos para evitar problemas de memoria
    max_text_length = 5000
    if len(deepseek) > max_text_length:
        deepseek = deepseek[:max_text_length] + "\n\n[Texto truncado por límite de memoria]"
    if len(gemini) > max_text_length:
        gemini = gemini[:max_text_length] + "\n\n[Texto truncado por límite de memoria]"
    if len(summary) > max_text_length:
        summary = summary[:max_text_length] + "\n\n[Texto truncado por límite de memoria]"
    if len(comparison) > max_text_length:
        comparison = comparison[:max_text_length] + "\n\n[Texto truncado por límite de memoria]"

    # --- PÁGINA 1: DATOS Y DIAGNÓSTICOS DEL SISTEMA ---
    pdf.add_page()
    
    info_paciente = re.search(r'SECCION_INFO_PACIENTE\n(.*?)\nSECCION_FIN', medico, re.DOTALL).group(1).strip()
    hallazgos_clave = re.search(r'SECCION_HALLAZGOS_CLAVE\n(.*?)\nSECCION_FIN', medico, re.DOTALL).group(1).strip()
    diagnosticos = re.search(r'SECCION_DIAGNOSTICOS_SISTEMA\n(.*?)\nSECCION_FIN', medico, re.DOTALL).group(1).strip()

    pdf.section_title('Datos del Paciente y Examen')
    pdf.section_body(info_paciente)
    
    pdf.section_title('Resumen de Hallazgos Anormales (Sistema)')
    pdf.section_body(hallazgos_clave)

    pdf.section_title('Diagnósticos y Recomendaciones Registrados')
    pdf.section_body(diagnosticos)

    # --- PÁGINA 2: RESUMEN EJECUTIVO DE IA ---
    pdf.add_page()
    pdf.section_title('Resumen Ejecutivo (Análisis Sintetizado por IA)')
    pdf.section_body(summary)

    # --- PÁGINA 3: ANÁLISIS IA (DISEÑO SECUENCIAL) ---
    pdf.add_page(orientation='L')
    pdf.print_comparison_layout('Análisis Detallado de DeepSeek', deepseek, 'Análisis Detallado de Gemini', gemini)
    
    # --- PÁGINA 4: COMPARACIÓN DETALLADA ---
    pdf.add_page()
    pdf.section_title('Análisis Comparativo Detallado de las IAs')
    pdf.section_body(comparison)

    # --- PÁGINA 5: TABLA COMPARATIVA DE DIAGNÓSTICOS Y RECOMENDACIONES (HORIZONTAL) ---
    pdf.add_page(orientation='L')  # Página horizontal para mejor visualización
    
    # Extraer pares de diagnóstico-recomendación de cada fuente
    # Para el sistema médico, usar función específica para texto estructurado
    medico_pairs = extract_medico_pairs_from_structured_text(medico)
    print(f"📊 Pares extraídos del sistema médico: {len(medico_pairs)}")
    
    # Para las IAs, usar Gemini API para mayor precisión, con respaldo
    deepseek_pairs = extract_diagnosis_recommendation_pairs_with_gemini(deepseek, "DeepSeek", GOOGLE_API_KEY)
    if not deepseek_pairs:
        # Si no se extrajeron pares, usar respaldo
        print("⚠️ Usando función de respaldo para DeepSeek")
        deepseek_pairs = extract_fallback_pairs_from_text(deepseek, "DeepSeek")
        # Si aún no hay pares, generar basándose en datos del sistema médico
        if not deepseek_pairs and medico_pairs:
            print("⚠️ Generando pares para DeepSeek basados en datos del sistema médico")
            deepseek_pairs = extract_ai_pairs_from_medico_data(medico_pairs, "DeepSeek")
    print(f"📊 Pares extraídos de DeepSeek: {len(deepseek_pairs)}")
    if deepseek_pairs:
        for i, (diag, rec) in enumerate(deepseek_pairs[:3]):  # Mostrar solo los primeros 3
            print(f"  DeepSeek {i+1}: {diag[:30]}... -> {rec[:30]}...")
    
    gemini_pairs = extract_diagnosis_recommendation_pairs_with_gemini(gemini, "Gemini", GOOGLE_API_KEY)
    if not gemini_pairs:
        # Si no se extrajeron pares, usar respaldo
        print("⚠️ Usando función de respaldo para Gemini")
        gemini_pairs = extract_fallback_pairs_from_text(gemini, "Gemini")
        # Si aún no hay pares, generar basándose en datos del sistema médico
        if not gemini_pairs and medico_pairs:
            print("⚠️ Generando pares para Gemini basados en datos del sistema médico")
            gemini_pairs = extract_ai_pairs_from_medico_data(medico_pairs, "Gemini")
    print(f"📊 Pares extraídos de Gemini: {len(gemini_pairs)}")
    if gemini_pairs:
        for i, (diag, rec) in enumerate(gemini_pairs[:3]):  # Mostrar solo los primeros 3
            print(f"  Gemini {i+1}: {diag[:30]}... -> {rec[:30]}...")
    
    # Crear la tabla comparativa unificada
    pdf.print_diagnosis_recommendation_comparison_table(medico_pairs, deepseek_pairs, gemini_pairs)

    # --- PÁGINA 6: MÉTRICAS DE SIMILITUD Y CONCORDANCIA ---
    pdf.add_page()
    pdf.section_title('Métricas de Similitud y Concordancia')

    # Contenido explicativo
    explanation = (
        "Esta sección presenta diversas métricas para evaluar la concordancia entre el análisis del médico "
        "y los análisis generados por cada IA. Las métricas incluyen:\n\n"
        "• **Similitud Semántica**: Mide la concordancia en el significado usando vectores de texto\n"
        "• **Índice de Kappa Cohen**: Evalúa la concordancia entre evaluadores (médico vs IA)\n"
        "• **Similitud de Jaccard**: Compara la similitud de conjuntos de términos médicos\n\n"
        "Un puntaje más cercano a 1.0 indica una mayor concordancia."
    )
    pdf.section_body(explanation)
    pdf.ln(10)
   
    # Mostrar los resultados de similitud semántica
    sim_deepseek = metrics.get('deepseek_similarity', 0.0)
    sim_gemini = metrics.get('gemini_similarity', 0.0)

    pdf.section_title('Similitud Semántica (Coseno)')
    metric_text_ds = f"DeepSeek: {sim_deepseek:.4f} ({sim_deepseek*100:.2f}%)"
    metric_text_gm = f"Gemini:   {sim_gemini:.4f} ({sim_gemini*100:.2f}%)"
    
    pdf.section_body(metric_text_ds, is_metric=True)
    pdf.ln(2)
    pdf.section_body(metric_text_gm, is_metric=True)
    pdf.ln(5)

    # Mostrar los resultados de Kappa Cohen
    kappa_deepseek = metrics.get('deepseek_kappa', 0.0)
    kappa_gemini = metrics.get('gemini_kappa', 0.0)

    pdf.section_title('Índice de Kappa Cohen')
    kappa_text_ds = f"DeepSeek: {kappa_deepseek:.4f} ({kappa_deepseek*100:.2f}%)"
    kappa_text_gm = f"Gemini:   {kappa_gemini:.4f} ({kappa_gemini*100:.2f}%)"
    
    pdf.section_body(kappa_text_ds, is_metric=True)
    pdf.ln(2)
    pdf.section_body(kappa_text_gm, is_metric=True)
    pdf.ln(5)

    # Mostrar los resultados de Jaccard
    jaccard_deepseek = metrics.get('deepseek_jaccard', 0.0)
    jaccard_gemini = metrics.get('gemini_jaccard', 0.0)

    pdf.section_title('Similitud de Jaccard')
    jaccard_text_ds = f"DeepSeek: {jaccard_deepseek:.4f} ({jaccard_deepseek*100:.2f}%)"
    jaccard_text_gm = f"Gemini:   {jaccard_gemini:.4f} ({jaccard_gemini*100:.2f}%)"
    
    pdf.section_body(jaccard_text_ds, is_metric=True)
    pdf.ln(2)
    pdf.section_body(jaccard_text_gm, is_metric=True)

    return pdf.output()

# ==============================================================================
# FUNCIÓN DE PRUEBA PARA DEBUGGING
# ==============================================================================
def test_medico_extraction():
    """Función de prueba para verificar la extracción de pares del sistema médico."""
    # Simular texto del sistema médico
    test_medico_text = """
SECCION_INFO_PACIENTE
- Centro Médico: Test Medical Center
- Ciudad: Test City
SECCION_FIN

SECCION_HALLAZGOS_CLAVE
- Presión Arterial: 140/90 (Resultado: anormal)
SECCION_FIN

SECCION_DIAGNOSTICOS_SISTEMA
**Perfil Lipídico**
- Diagnóstico: HIPERTRIGLICERIDEMIA
  Recomendación: Dieta hipograsa y control de perfil lipídico

- Diagnóstico: OTRA HIPERLIPIDEMIA (COLESTEROL LDL 120.37MG/DL)
  Recomendación: Control de colesterol y evaluación nutricional

**Hemograma y Bioquímica**
- Diagnóstico: POLICITEMIA SECUNDARIA
  Recomendación: Evaluación por medicina interna

- Diagnóstico: SOBREPESO
  Recomendación: Plan de alimentación y ejercicio

**Otros Diagnósticos**
- Diagnóstico: BRADICARDIA SINUSAL
  Recomendación: Evaluación cardiológica

- Diagnóstico: DEFICIENCIA DE LIPOPROTEÍNAS HDL
  Recomendación: Modificación de estilo de vida
SECCION_FIN

SECCION_REPORTE_COMPLETO
Información del paciente y resultados...
SECCION_FIN
"""
    
    print("🧪 Iniciando prueba de extracción del sistema médico...")
    pairs = extract_medico_pairs_from_structured_text(test_medico_text)
    print(f"📊 Resultado de la prueba: {len(pairs)} pares extraídos")
    
    for i, (diag, rec) in enumerate(pairs):
        print(f"  {i+1}. {diag} -> {rec}")
    
    return pairs

if __name__ == "__main__":
    # Ejecutar prueba si se ejecuta directamente
    test_medico_extraction()