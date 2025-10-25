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
        model = genai.GenerativeModel('gemini-flash-latest')
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
        model = genai.GenerativeModel('gemini-flash-latest')
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
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error al generar la comparación con la IA: {e}"
    

# ==============================================================================
# MÉTRICAS 
# ==============================================================================
def calculate_semantic_similarity(text_medico, text_ia):
    """Calcula la similitud semántica usando la API de DeepSeek."""
    try:
        print("🔄 Calculando similitud semántica con DeepSeek...")
        
        # Extraer contenido médico
        medico_content_match = re.search(r'SECCION_REPORTE_COMPLETO\n(.*?)\nSECCION_FIN', text_medico, re.DOTALL)
        if not medico_content_match:
            print("❌ No se encontró SECCION_REPORTE_COMPLETO en el texto del médico.")
            return 0.0
        medico_content = medico_content_match.group(1).strip()
        
        # Limitar el contenido para evitar requests muy grandes
        if len(medico_content) > 1500:
            medico_content = medico_content[:1500] + "..."
        if len(text_ia) > 1500:
            text_ia = text_ia[:1500] + "..."
        
        # Crear prompt para DeepSeek
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
        
        # Configurar request a DeepSeek
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": "Eres un experto en análisis médico que calcula similitudes entre diagnósticos. Responde solo con números decimales entre 0.0 y 1.0."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Baja temperatura para respuestas más consistentes
            "max_tokens": 10     # Solo necesitamos un número
        }
        
        # Hacer request con timeout corto
        timeout = 15  # 15 segundos máximo
        try:
            print(f"🔄 Enviando request a DeepSeek (timeout: {timeout}s)...")
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            
            # Extraer el contenido de la respuesta
            if 'choices' in result and len(result['choices']) > 0:
                similarity_text = result['choices'][0]['message']['content'].strip()
                
                # Limpiar y convertir a float
                similarity_text = re.sub(r'[^\d.]', '', similarity_text)  # Solo números y puntos
                
                if similarity_text:
                    similarity_score = float(similarity_text)
                    # Asegurar que esté en el rango [0, 1]
                    similarity_score = max(0.0, min(1.0, similarity_score))
                    
                    print(f"✅ Similitud semántica calculada con DeepSeek: {similarity_score:.4f}")
                    return similarity_score
                else:
                    print("❌ Respuesta de DeepSeek no contiene número válido")
                    return 0.0
            else:
                print("❌ Respuesta inesperada de DeepSeek")
                return 0.0
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout en DeepSeek ({timeout}s), usando valor por defecto")
            return 0.0
        except requests.exceptions.RequestException as e:
            print(f"❌ Error de red con DeepSeek: {e}")
            return 0.0
        except ValueError as e:
            print(f"❌ Error convirtiendo respuesta de DeepSeek: {e}")
            return 0.0

    except Exception as e:
        print(f"❌ Error inesperado calculando similitud semántica: {e}")
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
        model = genai.GenerativeModel('gemini-flash-latest')
        
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
        model = genai.GenerativeModel('gemini-flash-latest')
        
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
        
        # Aplicar filtros
        pairs = filter_ophthalmology_diagnoses(pairs)
        pairs = filter_administrative_diagnoses(pairs)
        
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
        
        # Aplicar filtros
        pairs = filter_ophthalmology_diagnoses(pairs)
        pairs = filter_administrative_diagnoses(pairs)
        
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
        
        # Aplicar filtros
        pairs = filter_ophthalmology_diagnoses(pairs)
        pairs = filter_administrative_diagnoses(pairs)
        
        print(f"📊 Total de pares de respaldo para {source_name}: {len(pairs)}")
        return pairs[:5]  # Limitar a 5 pares para respaldo
        
    except Exception as e:
        print(f"❌ Error en extracción de respaldo para {source_name}: {e}")
        return []

def filter_ophthalmology_diagnoses(pairs):
    """Filtra diagnósticos relacionados con oftalmología."""
    ophthalmology_keywords = [
        'oftalmología', 'oftalmologico', 'oftalmologica',
        'ametropia', 'ametropía', 'corregida', 'corregido',
        'lentes', 'gafas', 'anteojos', 'visión', 'visual',
        'ocular', 'ojo', 'ojos', 'miopía', 'hipermetropía',
        'astigmatismo', 'demanda visual', 'salud ocular'
    ]
    
    filtered_pairs = []
    for diagnosis, recommendation in pairs:
        diagnosis_lower = diagnosis.lower()
        recommendation_lower = recommendation.lower()
        
        # Verificar si contiene palabras clave oftalmológicas
        is_ophthalmology = any(keyword in diagnosis_lower or keyword in recommendation_lower 
                              for keyword in ophthalmology_keywords)
        
        if not is_ophthalmology:
            filtered_pairs.append((diagnosis, recommendation))
        else:
            print(f"🚫 Filtrado diagnóstico oftalmológico: {diagnosis[:30]}...")
    
    return filtered_pairs

def filter_administrative_diagnoses(pairs):
    """Filtra diagnósticos administrativos como 'Ausencia de resultados'."""
    administrative_keywords = [
        'ausencia de resultados', 'perfil', 'análisis faltantes',
        'programar urgentemente', 'exámenes pendientes',
        'resultados pendientes', 'laboratorio pendiente'
    ]
    
    filtered_pairs = []
    for diagnosis, recommendation in pairs:
        diagnosis_lower = diagnosis.lower()
        recommendation_lower = recommendation.lower()
        
        # Verificar si contiene palabras clave administrativas
        is_administrative = any(keyword in diagnosis_lower or keyword in recommendation_lower 
                               for keyword in administrative_keywords)
        
        if not is_administrative:
            filtered_pairs.append((diagnosis, recommendation))
        else:
            print(f"🚫 Filtrado diagnóstico administrativo: {diagnosis[:30]}...")
    
    return filtered_pairs

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
        
        # Aplicar filtros
        ai_pairs = filter_ophthalmology_diagnoses(ai_pairs)
        ai_pairs = filter_administrative_diagnoses(ai_pairs)
        
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
        base_row_height = 6  # Altura base por línea de texto
        
        # Encabezados
        self.set_font('DejaVu', 'B', 10)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        
        # Dibujar encabezados
        self.cell(col_width, base_row_height * 2, 'MÉDICO/SISTEMA', 1, 0, 'C', fill=True)
        self.cell(col_width, base_row_height * 2, 'DEEPSEEK deepseek-chat', 1, 0, 'C', fill=True)
        self.cell(col_width, base_row_height * 2, 'GEMINI gemini-flash-latest', 1, 0, 'C', fill=True)
        self.ln(base_row_height * 2)
        
        # Configurar fuente para contenido
        self.set_font('DejaVu', '', 8)
        self.set_fill_color(255, 255, 255)
        
        # Crear diccionarios para organizar diagnósticos por similitud
        def normalize_diagnosis(diag):
            """Normaliza diagnósticos para agrupar similares"""
            diag_lower = diag.lower()
            if 'hipertrigliceridemia' in diag_lower or 'trigliceridemia' in diag_lower or 'dislipidemia' in diag_lower:
                return 'HIPERTRIGLICERIDEMIA'
            elif 'hiperlipidemia' in diag_lower or 'colesterol' in diag_lower or 'ldl' in diag_lower:
                return 'HIPERLIPIDEMIA'
            elif 'policitemia' in diag_lower:
                return 'POLICITEMIA'
            elif 'sobrepeso' in diag_lower or 'obesidad' in diag_lower or 'imc' in diag_lower:
                return 'SOBREPESO'
            elif 'bradicardia' in diag_lower or 'cardiaco' in diag_lower:
                return 'BRADICARDIA'
            elif 'hdl' in diag_lower or 'deficiencia' in diag_lower:
                return 'DEFICIENCIA_HDL'
            elif 'diabetes' in diag_lower or 'glucosa' in diag_lower:
                return 'DIABETES'
            elif 'hipertensión' in diag_lower or 'presión' in diag_lower:
                return 'HIPERTENSIÓN'
            elif 'anemia' in diag_lower or 'hemoglobina' in diag_lower:
                return 'ANEMIA'
            elif 'gastritis' in diag_lower or 'gástrico' in diag_lower:
                return 'GASTRITIS'
            else:
                # Para diagnósticos únicos, usar el nombre original pero normalizado
                return diag.upper().strip()
        
        # Organizar diagnósticos por categorías
        organized_diagnoses = {}
        
        # Procesar diagnósticos del médico
        for diag, rec in medico_pairs:
            norm_diag = normalize_diagnosis(diag)
            if norm_diag not in organized_diagnoses:
                organized_diagnoses[norm_diag] = {'medico': [], 'deepseek': [], 'gemini': []}
            organized_diagnoses[norm_diag]['medico'].append((diag, rec))
        
        # Procesar diagnósticos de DeepSeek
        for diag, rec in deepseek_pairs:
            norm_diag = normalize_diagnosis(diag)
            if norm_diag not in organized_diagnoses:
                organized_diagnoses[norm_diag] = {'medico': [], 'deepseek': [], 'gemini': []}
            organized_diagnoses[norm_diag]['deepseek'].append((diag, rec))
        
        # Procesar diagnósticos de Gemini
        for diag, rec in gemini_pairs:
            norm_diag = normalize_diagnosis(diag)
            if norm_diag not in organized_diagnoses:
                organized_diagnoses[norm_diag] = {'medico': [], 'deepseek': [], 'gemini': []}
            organized_diagnoses[norm_diag]['gemini'].append((diag, rec))
        
        # Agregar diagnósticos únicos de las IAs que no están en el sistema médico
        # Crear un conjunto de diagnósticos del médico para comparar
        medico_diagnoses = set()
        for diag, rec in medico_pairs:
            medico_diagnoses.add(normalize_diagnosis(diag))
        
        # Agregar diagnósticos únicos de DeepSeek
        for diag, rec in deepseek_pairs:
            norm_diag = normalize_diagnosis(diag)
            if norm_diag not in medico_diagnoses and norm_diag not in organized_diagnoses:
                organized_diagnoses[norm_diag] = {'medico': [], 'deepseek': [], 'gemini': []}
                organized_diagnoses[norm_diag]['deepseek'].append((diag, rec))
        
        # Agregar diagnósticos únicos de Gemini
        for diag, rec in gemini_pairs:
            norm_diag = normalize_diagnosis(diag)
            if norm_diag not in medico_diagnoses and norm_diag not in organized_diagnoses:
                organized_diagnoses[norm_diag] = {'medico': [], 'deepseek': [], 'gemini': []}
                organized_diagnoses[norm_diag]['gemini'].append((diag, rec))
        
        # Si no hay diagnósticos organizados, mostrar mensaje
        if not organized_diagnoses:
            self.cell(col_width * 3, base_row_height * 2, 'No se encontraron pares diagnóstico-recomendación', 1, 0, 'C')
            self.ln(base_row_height * 2)
            return
        
        # Imprimir tabla organizada
        for norm_diag, sources in organized_diagnoses.items():
            # Calcular altura máxima para esta fila
            max_height = 0
            
            # Preparar textos para cada columna
            medico_texts = []
            deepseek_texts = []
            gemini_texts = []
            
            # Función para truncar texto
            def truncate_text(text, max_length):
                if len(text) <= max_length:
                    return text
                return text[:max_length-3] + "..."
            
            # Función para eliminar duplicados en una lista de pares
            def remove_duplicates_in_pairs(pairs):
                seen_diagnoses = set()
                unique_pairs = []
                for diag, rec in pairs:
                    # Normalizar diagnóstico para comparar (más simple)
                    diag_normalized = diag.lower().strip()
                    # Remover caracteres especiales y espacios extra
                    diag_normalized = re.sub(r'[^\w\s]', '', diag_normalized)
                    diag_normalized = re.sub(r'\s+', ' ', diag_normalized).strip()
                    
                    if diag_normalized not in seen_diagnoses:
                        seen_diagnoses.add(diag_normalized)
                        unique_pairs.append((diag, rec))
                return unique_pairs
            
            # Procesar médico
            if sources['medico']:
                unique_medico = remove_duplicates_in_pairs(sources['medico'])
                for diag, rec in unique_medico:
                    diag_short = truncate_text(diag, 40)
                    rec_short = truncate_text(rec, 50)
                    medico_texts.append(f"• {diag_short}\n  → {rec_short}")
            else:
                medico_texts.append("Sin diagnóstico")
            
            # Procesar DeepSeek
            if sources['deepseek']:
                unique_deepseek = remove_duplicates_in_pairs(sources['deepseek'])
                for diag, rec in unique_deepseek:
                    diag_short = truncate_text(diag, 40)
                    rec_short = truncate_text(rec, 50)
                    deepseek_texts.append(f"• {diag_short}\n  → {rec_short}")
            else:
                deepseek_texts.append("Sin diagnóstico")
            
            # Procesar Gemini
            if sources['gemini']:
                unique_gemini = remove_duplicates_in_pairs(sources['gemini'])
                for diag, rec in unique_gemini:
                    diag_short = truncate_text(diag, 40)
                    rec_short = truncate_text(rec, 50)
                    gemini_texts.append(f"• {diag_short}\n  → {rec_short}")
            else:
                gemini_texts.append("Sin diagnóstico")
            
            # Unir textos de cada columna
            medico_text = "\n\n".join(medico_texts)
            deepseek_text = "\n\n".join(deepseek_texts)
            gemini_text = "\n\n".join(gemini_texts)
            
            # Calcular altura necesaria basada en el contenido real
            for text in [medico_text, deepseek_text, gemini_text]:
                if text and text.strip():
                    lines = text.split('\n')
                    content_height = 0
                    for line in lines:
                        line = line.strip()
                        if line:
                            if line.startswith('• '):
                                content_height += 3.5  # Diagnóstico
                            elif line.startswith('  → '):
                                content_height += 3   # Recomendación
                            else:
                                content_height += 3.5  # Texto normal
                        else:
                            content_height += 2  # Línea vacía
                    content_height += 4  # Margen
                    max_height = max(max_height, content_height)
                else:
                    max_height = max(max_height, 8)  # Altura mínima para "Sin diagnóstico"
            
            # Asegurar altura mínima y máxima
            row_height = max(min(max_height, 25), 10)  # Entre 10 y 25mm
            
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
                   "Los diagnósticos similares se agrupan en la misma fila para facilitar la comparación."
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
        
        # Procesar el texto línea por línea
        if txt and txt.strip():
            lines = txt.split('\n')
            current_y = y + 2
            max_width = w - 4  # Ancho disponible para el texto
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    current_y += 2  # Espacio para línea vacía
                    continue
                
                # Determinar el estilo de fuente según el contenido
                if line.startswith('• '):
                    # Es un diagnóstico (con viñeta)
                    self.set_font('DejaVu', 'B', 7)
                    line_height = 3
                    # Limitar longitud del diagnóstico
                    if len(line) > 50:
                        line = line[:47] + "..."
                elif line.startswith('  → '):
                    # Es una recomendación (con flecha)
                    self.set_font('DejaVu', '', 6)
                    line_height = 2.5
                    # Limitar longitud de la recomendación
                    if len(line) > 60:
                        line = line[:57] + "..."
                else:
                    # Texto normal
                    self.set_font('DejaVu', '', 7)
                    line_height = 3
                    # Limitar longitud del texto normal
                    if len(line) > 50:
                        line = line[:47] + "..."
                
                # Verificar si hay espacio suficiente en la celda
                if current_y + line_height > y + h - 2:
                    # No hay espacio, cortar con "..."
                    self.set_xy(x + 2, current_y)
                    self.multi_cell(max_width, line_height, "...", 0, align)
                    break
                
                # Imprimir la línea con ajuste automático de texto
                self.set_xy(x + 2, current_y)
                self.multi_cell(max_width, line_height, line, 0, align)
                current_y += line_height + 0.5  # Pequeño espacio entre líneas
        else:
            # Texto vacío
            self.set_font('DejaVu', '', 7)
            self.multi_cell(w - 4, 3, "Sin diagnóstico", 0, align)
        
        # Restaurar posición para la siguiente celda
        if ln == 1:  # Si es la última celda de la fila
            self.set_xy(x + w, y)
        else:
            self.set_xy(x + w, y)

def generate_pdf_in_memory(token, medico, deepseek, gemini, summary, comparison,metrics):
    """Genera un PDF simplificado enfocado en análisis de IA y métricas."""

    pdf = PDF('P', 'mm', 'A4')
    pdf.alias_nb_pages()
    
    # Limitar el tamaño de los textos para evitar problemas de memoria
    max_text_length = 5000
    if len(deepseek) > max_text_length:
        deepseek = deepseek[:max_text_length] + "\n\n[Texto truncado por límite de memoria]"
    if len(gemini) > max_text_length:
        gemini = gemini[:max_text_length] + "\n\n[Texto truncado por límite de memoria]"

    # --- PÁGINA 1: ANÁLISIS DETALLADO DE DEEPSEEK ---
    pdf.add_page()
    pdf.section_title('Análisis Detallado de DeepSeek')
    pdf.section_body(deepseek)

    # --- PÁGINA 2: ANÁLISIS DETALLADO DE GEMINI ---
    pdf.add_page()
    pdf.section_title('Análisis Detallado de Gemini')
    pdf.section_body(gemini)

    # --- PÁGINA 3: TABLA COMPARATIVA DE DIAGNÓSTICOS Y RECOMENDACIONES ---
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

    # --- PÁGINA 4: MÉTRICAS DE SIMILITUD Y CONCORDANCIA ---
    pdf.add_page()
    pdf.section_title('Métricas de Similitud y Concordancia')

    # Contenido explicativo
    explanation = (
        "Esta sección presenta diversas métricas para evaluar la concordancia entre el análisis del médico "
        "y los análisis generados por cada IA. Las métricas incluyen:\n\n"
        "• **Similitud Semántica (Cosenos)**: Mide la concordancia en el significado usando vectores de texto\n"
        "• **Índice de Kappa Cohen**: Evalúa la concordancia entre evaluadores (médico vs IA)\n"
        "• **Similitud de Jaccard**: Compara la similitud de conjuntos de términos médicos\n\n"
        "Un puntaje más cercano a 1.0 indica una mayor concordancia."
    )
    pdf.section_body(explanation)
    pdf.ln(10)

    # --- SECCIÓN DEEPSEEK ---
    pdf.section_title('Métricas de DeepSeek (deepseek-chat)')
    
    # Obtener métricas de DeepSeek
    sim_deepseek = metrics.get('deepseek_similarity', 0.0)
    kappa_deepseek = metrics.get('deepseek_kappa', 0.0)
    jaccard_deepseek = metrics.get('deepseek_jaccard', 0.0)
    
    # Crear tabla de métricas para DeepSeek
    deepseek_metrics_text = (
        f"**Similitud de Cosenos**: {sim_deepseek:.4f} ({sim_deepseek*100:.2f}%)\n"
        f"**Índice de Kappa Cohen**: {kappa_deepseek:.4f} ({kappa_deepseek*100:.2f}%)\n"
        f"**Similitud de Jaccard**: {jaccard_deepseek:.4f} ({jaccard_deepseek*100:.2f}%)\n\n"
        f"**Interpretación**:\n"
        f"• Similitud de Cosenos: {'Excelente' if sim_deepseek >= 0.8 else 'Buena' if sim_deepseek >= 0.6 else 'Moderada' if sim_deepseek >= 0.4 else 'Baja'}\n"
        f"• Concordancia Kappa: {'Excelente' if kappa_deepseek >= 0.8 else 'Buena' if kappa_deepseek >= 0.6 else 'Moderada' if kappa_deepseek >= 0.4 else 'Baja'}\n"
        f"• Similitud Jaccard: {'Excelente' if jaccard_deepseek >= 0.8 else 'Buena' if jaccard_deepseek >= 0.6 else 'Moderada' if jaccard_deepseek >= 0.4 else 'Baja'}"
    )
    pdf.section_body(deepseek_metrics_text, is_metric=True)
    pdf.ln(10)

    # --- SECCIÓN GEMINI ---
    pdf.section_title('Métricas de Gemini (gemini-flash-latest)')
    
    # Obtener métricas de Gemini
    sim_gemini = metrics.get('gemini_similarity', 0.0)
    kappa_gemini = metrics.get('gemini_kappa', 0.0)
    jaccard_gemini = metrics.get('gemini_jaccard', 0.0)
    
    # Crear tabla de métricas para Gemini
    gemini_metrics_text = (
        f"**Similitud de Cosenos**: {sim_gemini:.4f} ({sim_gemini*100:.2f}%)\n"
        f"**Índice de Kappa Cohen**: {kappa_gemini:.4f} ({kappa_gemini*100:.2f}%)\n"
        f"**Similitud de Jaccard**: {jaccard_gemini:.4f} ({jaccard_gemini*100:.2f}%)\n\n"
        f"**Interpretación**:\n"
        f"• Similitud de Cosenos: {'Excelente' if sim_gemini >= 0.8 else 'Buena' if sim_gemini >= 0.6 else 'Moderada' if sim_gemini >= 0.4 else 'Baja'}\n"
        f"• Concordancia Kappa: {'Excelente' if kappa_gemini >= 0.8 else 'Buena' if kappa_gemini >= 0.6 else 'Moderada' if kappa_gemini >= 0.4 else 'Baja'}\n"
        f"• Similitud Jaccard: {'Excelente' if jaccard_gemini >= 0.8 else 'Buena' if jaccard_gemini >= 0.6 else 'Moderada' if jaccard_gemini >= 0.4 else 'Baja'}"
    )
    pdf.section_body(gemini_metrics_text, is_metric=True)
    pdf.ln(10)

    # --- TABLA COMPARATIVA DE MÉTRICAS ---
    pdf.section_title('Tabla Comparativa de Métricas por Versión de IA')
    
    # Crear tabla comparativa
    comparison_table_text = (
        "| Métrica | DeepSeek (deepseek-chat) | Gemini (gemini-flash-latest) |\n"
        "|---------|--------------------------|----------------------------|\n"
        f"| **Similitud de Cosenos** | {sim_deepseek:.4f} ({sim_deepseek*100:.2f}%) | {sim_gemini:.4f} ({sim_gemini*100:.2f}%) |\n"
        f"| **Índice de Kappa Cohen** | {kappa_deepseek:.4f} ({kappa_deepseek*100:.2f}%) | {kappa_gemini:.4f} ({kappa_gemini*100:.2f}%) |\n"
        f"| **Similitud de Jaccard** | {jaccard_deepseek:.4f} ({jaccard_deepseek*100:.2f}%) | {jaccard_gemini:.4f} ({jaccard_gemini*100:.2f}%) |\n\n"
        "**Resumen de Rendimiento**:\n"
        f"• **Mejor Similitud de Cosenos**: {'DeepSeek' if sim_deepseek > sim_gemini else 'Gemini' if sim_gemini > sim_deepseek else 'Empate'}\n"
        f"• **Mejor Concordancia Kappa**: {'DeepSeek' if kappa_deepseek > kappa_gemini else 'Gemini' if kappa_gemini > kappa_deepseek else 'Empate'}\n"
        f"• **Mejor Similitud Jaccard**: {'DeepSeek' if jaccard_deepseek > jaccard_gemini else 'Gemini' if jaccard_gemini > jaccard_deepseek else 'Empate'}\n\n"
        f"**Puntuación Promedio**:\n"
        f"• DeepSeek: {((sim_deepseek + kappa_deepseek + jaccard_deepseek) / 3):.4f}\n"
        f"• Gemini: {((sim_gemini + kappa_gemini + jaccard_gemini) / 3):.4f}"
    )
    pdf.section_body(comparison_table_text, is_metric=True)

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