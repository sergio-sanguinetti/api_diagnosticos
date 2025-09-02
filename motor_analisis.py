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
        return response.json()['choices'][0]['message']['content']
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
    """Calcula la similitud de coseno usando la API de Inferencia de Hugging Face."""
    if not HUGGINGFACE_API_KEY:
        print("⚠️ No se encontró la clave de API de Hugging Face en las variables de entorno.")
        return 0.0

    try:
        medico_content_match = re.search(r'SECCION_REPORTE_COMPLETO\n(.*?)\nSECCION_FIN', text_medico, re.DOTALL)
        if not medico_content_match:
            print("❌ No se encontró SECCION_REPORTE_COMPLETO en el texto del médico.")
            return 0.0
        medico_content = medico_content_match.group(1).strip()
        
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
        
        response = requests.post(HF_EMBEDDING_MODEL_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status() 
        
        similarity_scores = response.json()
        
        # La API devuelve una lista de puntajes, tomamos el primero
        if not isinstance(similarity_scores, list) or len(similarity_scores) == 0:
            print(f"❌ Respuesta de similitud inesperada de la API de Hugging Face: {similarity_scores}")
            return 0.0

        return float(similarity_scores[0])

    except requests.exceptions.RequestException as e:
        print(f"❌ Error de red con la API de Hugging Face: {e}")
        # Imprime la respuesta del servidor si está disponible, para más detalles
        if e.response:
            print(f"Server response: {e.response.text}")
        return 0.0
    except Exception as e:
        print(f"❌ Error calculando la similitud: {e}")
        return 0.0

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

def find_similar_diagnoses(medico_pairs, deepseek_pairs, gemini_pairs):
    """Encuentra diagnósticos similares entre las diferentes fuentes y asigna colores."""
    from difflib import SequenceMatcher
    
    # Función para calcular similitud entre dos textos
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    # Colores disponibles (RGB)
    colors = [
        (255, 240, 240),  # Rojo claro
        (240, 255, 240),  # Verde claro
        (240, 240, 255),  # Azul claro
        (255, 255, 240),  # Amarillo claro
        (255, 240, 255),  # Magenta claro
        (240, 255, 255),  # Cian claro
        (255, 230, 200),  # Naranja claro
        (230, 230, 255),  # Púrpura claro
    ]
    
    # Umbral de similitud (70% para considerar diagnósticos similares)
    similarity_threshold = 0.7
    
    # Diccionario para almacenar grupos de diagnósticos similares
    diagnosis_groups = {}
    color_assignments = {}
    
    # Extraer diagnósticos de cada fuente
    all_diagnoses = []
    for i, (diag, rec) in enumerate(medico_pairs):
        all_diagnoses.append(('medico', i, diag))
    for i, (diag, rec) in enumerate(deepseek_pairs):
        all_diagnoses.append(('deepseek', i, diag))
    for i, (diag, rec) in enumerate(gemini_pairs):
        all_diagnoses.append(('gemini', i, diag))
    
    # Agrupar diagnósticos similares
    group_id = 0
    for source1, idx1, diag1 in all_diagnoses:
        if (source1, idx1) in color_assignments:
            continue
            
        similar_diagnoses = [(source1, idx1, diag1)]
        
        for source2, idx2, diag2 in all_diagnoses:
            if source2 == source1 and idx2 == idx1:
                continue
            if (source2, idx2) in color_assignments:
                continue
                
            if similarity(diag1, diag2) >= similarity_threshold:
                similar_diagnoses.append((source2, idx2, diag2))
        
        # Si hay diagnósticos similares, asignar color
        if len(similar_diagnoses) > 1:
            color = colors[group_id % len(colors)]
            for source, idx, diag in similar_diagnoses:
                color_assignments[(source, idx)] = color
            group_id += 1
    
    return color_assignments

def extract_diagnosis_recommendation_pairs_with_gemini(text, source_name, api_key):
    """Extrae pares de diagnóstico-recomendación usando Gemini API con un prompt especializado."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
                    **TAREA ESPECÍFICA**: Extrae ÚNICAMENTE pares de diagnóstico-recomendación específicos mencionados en el siguiente texto.
                    
                    **INSTRUCCIONES CRÍTICAS**:
                    1. Extrae SOLO pares donde un diagnóstico médico específico tiene una recomendación clara asociada
                    2. Formato: "DIAGNÓSTICO | RECOMENDACIÓN"
                    3. NO extraigas diagnósticos sin recomendación asociada
                    4. NO extraigas recomendaciones sin diagnóstico específico
                    5. NO extraigas recomendaciones sueltas o generales
                    6. Extrae EXACTAMENTE como aparecen mencionados en el texto
                    7. Máximo 6 pares
                    8. Si no hay pares específicos, devuelve lista vacía
                    
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
        
        # Procesar la respuesta
        if "sin pares diagnóstico-recomendación" in result.lower():
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
        
        return pairs[:8]  # Limitar a 8 pares máximo
        
    except Exception as e:
        print(f"❌ Error extrayendo pares diagnóstico-recomendación con Gemini para {source_name}: {e}")
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
        
        # Encontrar diagnósticos similares y asignar colores
        color_assignments = find_similar_diagnoses(medico_pairs, deepseek_pairs, gemini_pairs)
        
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
            
            # Obtener colores de fondo para cada celda
            medico_color = color_assignments.get(('medico', i), None)
            deepseek_color = color_assignments.get(('deepseek', i), None)
            gemini_color = color_assignments.get(('gemini', i), None)
            
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
            # Considerar que diagnóstico es más alto (3.5mm) y recomendación más compacta (3mm)
            max_height = 0
            
            for text in [medico_text, deepseek_text, gemini_text]:
                if text and '\n' in text:
                    lines = text.split('\n')
                    if len(lines) >= 2:
                        # Calcular líneas necesarias para cada parte
                        diag_lines = max(1, len(lines[0]) // 25)  # Aproximadamente 25 caracteres por línea
                        rec_lines = max(1, len(lines[1]) // 25)
                        
                        # Altura total: diagnóstico + recomendación + separación
                        text_height = (diag_lines * 3.5) + (rec_lines * 3) + 2 + 4  # +2 separación, +4 márgenes
                    else:
                        # Una línea de diagnóstico
                        diag_lines = max(1, len(text) // 25)
                        text_height = (diag_lines * 3.5) + 4  # +4 márgenes
                elif text:
                    # Una línea de diagnóstico
                    diag_lines = max(1, len(text) // 25)
                    text_height = (diag_lines * 3.5) + 4  # +4 márgenes
                else:
                    text_height = 8  # Altura mínima para celda vacía
                
                max_height = max(max_height, text_height)
            
            # Asegurar altura mínima
            row_height = max(max_height, 15)  # Mínimo 15mm para diagnóstico + recomendación
            
            # Imprimir las celdas de esta fila con colores
            self._print_cell_with_wrap(col_width, row_height, medico_text, 1, 0, 'L', medico_color)
            self._print_cell_with_wrap(col_width, row_height, deepseek_text, 1, 0, 'L', deepseek_color)
            self._print_cell_with_wrap(col_width, row_height, gemini_text, 1, 0, 'L', gemini_color)
            
            self.ln(row_height)
        
        # Agregar nota explicativa
        self.ln(5)
        self.set_font('DejaVu', '', 8)
        self.set_text_color(100, 100, 100)
        note_text = "Esta tabla muestra los pares de diagnóstico-recomendación extraídos de cada fuente. " \
                   "Los diagnósticos similares entre fuentes se resaltan con el mismo color de fondo. " \
                   "Los pares se extraen usando Gemini API con prompts especializados para mayor precisión."
        self.multi_cell(0, 4, note_text)
        self.ln(5)

    def _print_cell_with_wrap(self, w, h, txt, border, ln, align, bg_color=None):
        """Imprime una celda con ajuste automático de texto usando multi_cell para saltos de línea."""
        # Guardar posición actual
        x = self.get_x()
        y = self.get_y()
        
        # Dibujar borde si es necesario
        if border:
            self.rect(x, y, w, h)
        
        # Aplicar color de fondo si se proporciona
        if bg_color:
            self.set_fill_color(*bg_color)
            self.rect(x, y, w, h, 'F')  # Rellenar con color de fondo
        
        # Configurar posición para el texto
        self.set_xy(x + 2, y + 2)  # Pequeño margen interno
        
        # Si el texto tiene diagnóstico y recomendación (separados por \n)
        if '\n' in txt and txt.strip():
            lines = txt.split('\n')
            if len(lines) >= 2:
                # Primera línea: diagnóstico en negrita
                self.set_font('DejaVu', 'B', 8)
                self.multi_cell(w - 4, 3.5, lines[0].strip(), 0, align)
                
                # Segunda línea: recomendación en normal
                self.set_font('DejaVu', '', 7)
                self.multi_cell(w - 4, 3, lines[1].strip(), 0, align)
            else:
                # Si solo hay una línea, mostrarla en negrita (es un diagnóstico)
                self.set_font('DejaVu', 'B', 8)
                self.multi_cell(w - 4, 3.5, txt, 0, align)
        else:
            # Texto simple sin separación - asumir que es un diagnóstico
            self.set_font('DejaVu', 'B', 8)
            self.multi_cell(w - 4, 3.5, txt, 0, align)
        
        # Restaurar color de fondo a blanco
        self.set_fill_color(255, 255, 255)
        
        # Restaurar posición para la siguiente celda
        if ln == 1:  # Si es la última celda de la fila
            self.set_xy(x + w, y)
        else:
            self.set_xy(x + w, y)

def generate_pdf_in_memory(token, medico, deepseek, gemini, summary, comparison,metrics):
    """Genera un PDF profesional multi-página en memoria."""

    pdf = PDF('P', 'mm', 'A4')
    pdf.alias_nb_pages()

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

     # --- PÁGINA 5: MÉTRICAS DE SIMILITUD ---
    pdf.add_page()
    pdf.section_title('Métricas de Similitud Semántica (vs. Informe Médico)')

    # Contenido explicativo
    explanation = (
        "Esta sección cuantifica la similitud en el significado (semántica) entre el análisis del médico "
        "y los análisis generados por cada IA. Se utiliza la 'Similitud de Coseno' sobre vectores de texto "
        "generados con el modelo Sentence-BERT.\n\n"
        "Un puntaje más cercano a 1.0 indica una mayor concordancia en el contenido y contexto."
    )
    pdf.section_body(explanation)
    pdf.ln(10)
   
   # Mostramos los resultados
    sim_deepseek = metrics.get('deepseek_similarity', 0.0)
    sim_gemini = metrics.get('gemini_similarity', 0.0)

    metric_text_ds = f"Similitud Semántica DeepSeek: {sim_deepseek:.4f} ({sim_deepseek*100:.2f}%)"
    metric_text_gm = f"Similitud Semántica Gemini:   {sim_gemini:.4f} ({sim_gemini*100:.2f}%)"
    
    pdf.section_body(metric_text_ds, is_metric=True)
    pdf.ln(2)
    pdf.section_body(metric_text_gm, is_metric=True)

    # --- PÁGINA 6: TABLA COMPARATIVA DE DIAGNÓSTICOS Y RECOMENDACIONES (HORIZONTAL) ---
    pdf.add_page(orientation='L')  # Página horizontal para mejor visualización
    
    # Extraer pares de diagnóstico-recomendación de cada fuente usando Gemini API para mayor precisión
    medico_pairs = extract_diagnosis_recommendation_pairs_with_gemini(medico, "Médico", GOOGLE_API_KEY)
    deepseek_pairs = extract_diagnosis_recommendation_pairs_with_gemini(deepseek, "DeepSeek", GOOGLE_API_KEY)
    gemini_pairs = extract_diagnosis_recommendation_pairs_with_gemini(gemini, "Gemini", GOOGLE_API_KEY)
    
    # Crear la tabla comparativa unificada
    pdf.print_diagnosis_recommendation_comparison_table(medico_pairs, deepseek_pairs, gemini_pairs)

    return pdf.output()