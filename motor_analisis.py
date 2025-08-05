# ==============================================================================
# SCRIPT DE ANÁLISIS MÉDICO Y GENERACIÓN DE REPORTES V2
#
# Descripción:
# Versión mejorada que genera un PDF multi-página con un diseño
# profesional, incluyendo un resumen ejecutivo sintetizado por IA.
# ==============================================================================

import mysql.connector
from mysql.connector import Error
import json
import requests
import google.generativeai as genai
from fpdf import FPDF
import sys
import textwrap
import re # Importar la librería de expresiones regulares

# ==============================================================================
# CONFIGURACIÓN DE CREDENCIALES (Sin cambios)
# ==============================================================================
DB_HOST = "193.203.175.193"
DB_USER = "u212843563_good_salud"
DB_PASS = "@9UbqRmS/oy"
DB_NAME = "u212843563_good_salud"
DEEPSEEK_API_KEY = "sk-37167855ce4243e8afe1ccb669021e64"
GOOGLE_API_KEY = "AIzaSyDqsYubkpT4Q_CofYluhK6lqmQHJui_U9A"

# ==============================================================================
# FUNCIONES DE CONEXIÓN Y ANÁLISIS DE IA (Sin cambios)
# ==============================================================================
# La función create_db_connection es la misma.
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

# La función get_patient_results es la misma.
def get_patient_results(connection, token_resultado):
    """Obtiene y formatea los resultados de un paciente específico desde la BD."""
    cursor = connection.cursor(dictionary=True)
    try:
        query = "SELECT * FROM resultados WHERE token_resultado = %s"
        cursor.execute(query, (token_resultado,))
        result = cursor.fetchone()

        if not result:
            return "No se encontraron resultados para el token proporcionado."

        try:
            diagnosticos_json = json.loads(result.get('diagnosticos', '[]'))
            diagnosticos_formateados = "\n".join([
                f"- Diagnóstico: {item.get('diagnostico', 'N/A')}\n  Recomendación: {item.get('recomendacion', 'N/A')}"
                for item in diagnosticos_json
            ])
        except json.JSONDecodeError:
            diagnosticos_formateados = result.get('diagnosticos', 'Datos de diagnóstico no válidos.')

        # Extraemos solo los resultados anormales para un resumen
        hallazgos_clave = []
        for key, value in result.items():
            if key.startswith('resultado_') and value and 'anormal' in str(value).lower():
                parametro = key.replace('resultado_', '').replace('_', ' ').title()
                valor_parametro = result.get(key.replace('resultado_', ''), 'N/A')
                hallazgos_clave.append(f"- {parametro}: {valor_parametro} (Resultado: {value})")
        
        hallazgos_formateados = "\n".join(hallazgos_clave) if hallazgos_clave else "No se encontraron hallazgos anormales en las pruebas."

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
{diagnosticos_formateados}
SECCION_FIN

SECCION_REPORTE_COMPLETO
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
SECCION_FIN
"""
        return report
    except Error as e:
        return f"❌ Error al consultar la base de datos: {e}"
    finally:
        cursor.close()

# Las funciones get_standard_prompt, analyze_with_deepseek, y analyze_with_gemini son las mismas.
def get_standard_prompt(report):
    """Crea un prompt estandarizado para asegurar respuestas consistentes."""
    report_completo = re.search(r'SECCION_REPORTE_COMPLETO\n(.*?)\nSECCION_FIN', report, re.DOTALL).group(1)
    return f"""
    **Rol:** Eres un asistente médico experto en medicina ocupacional.
    **Tarea:** Analiza el siguiente informe de resultados de un examen médico. Tu objetivo es identificar hallazgos anormales, correlacionarlos y proponer posibles diagnósticos diferenciales, junto con recomendaciones. NO inventes valores ni información; básate únicamente en los datos proporcionados.
    **Informe para analizar:**
    {report_completo}
    **Formato de Respuesta Requerido (usa Markdown):**
    ### Resumen General del Paciente
    (Describe en 1-2 frases el estado general del paciente basado en los resultados).
    ### Hallazgos Clave
    (Usa una lista con viñetas para enumerar TODOS los resultados anormales).
    ### Análisis y Correlación Diagnóstica
    (Explica qué podrían significar los hallazgos anormales en conjunto).
    ### Análisis por Examen y Posibles Diagnósticos
    (Para cada hallazgo clave, explica su significado y posibles diagnósticos asociados).
    ### Recomendaciones Sugeridas
    (Sugiere los siguientes pasos).
    """

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
# ¡NUEVAS FUNCIONES PARA RESUMEN Y COMPARACIÓN MEJORADA!
# ==============================================================================
def get_executive_summary_prompt(deepseek_analysis, gemini_analysis):
    """NUEVO PROMPT: Crea un prompt para generar un resumen ejecutivo unificado."""
    return f"""
    **Rol:** Eres un Director Médico supervisor. Tu tarea es revisar dos análisis generados por asistentes de IA (DeepSeek y Gemini) y sintetizarlos en un único "Resumen Ejecutivo" para la gerencia y el paciente.

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
    (Basado en ambos análisis, ¿cuáles son los diagnósticos o problemas de salud más importantes y acordados? Ej: Hipertrigliceridemia y Microhematuria en estudio).

    ### Acciones Prioritarias Sugeridas
    (Enumera las 3-4 recomendaciones más cruciales en las que ambos asistentes coinciden. Deben ser acciones claras. Ej: 1. Iniciar dieta baja en grasas. 2. Repetir examen de sangre en 1 mes. 3. Consultar con médico internista).

    ### Discrepancias o Puntos Únicos de Interés
    (¿Hubo algún diagnóstico o recomendación importante que un asistente mencionó y el otro no? Sé breve. Ej: El Asistente 1 sugiere una evaluación específica de resistencia a la insulina, lo cual es un punto relevante a considerar).

    ### Conclusión General
    (En una frase, resume el estado del paciente y el siguiente paso. Ej: El paciente presenta factores de riesgo metabólicos y un hallazgo urológico que requieren seguimiento especializado para determinar la causa y tratamiento).
    """

def generate_executive_summary(deepseek_analysis, gemini_analysis, api_key):
    """NUEVA FUNCIÓN: Llama a la IA para obtener el resumen ejecutivo."""
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

# La función compare_ai_analyses es la misma.
def compare_ai_analyses(deepseek_analysis, gemini_analysis, api_key):
    # ... (código sin cambios)
    return "Comparación detallada..." # Placeholder

# ==============================================================================
# ¡NUEVA CLASE PDF MEJORADA!
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
    
    def section_body(self, text):
        self.set_font('DejaVu', '', 10)
        self.set_text_color(51, 51, 51)
        # Limpieza de Markdown básico para una mejor presentación
        text = re.sub(r'###\s*(.*?)\n', r'\1\n', text)
        text = text.replace('**', '').replace('* ', '- ')
        self.multi_cell(0, 6, text)
        self.ln(5)

    def print_two_column_layout(self, title1, content1, title2, content2):
        self.set_font('DejaVu', 'B', 12)
        self.set_text_color(34, 49, 63)
        
        page_width = self.w - self.l_margin - self.r_margin
        col_width = (page_width - 10) / 2
        
        # Guardar la posición Y inicial
        y_initial = self.get_y()
        
        # Columna 1
        self.multi_cell(col_width, 8, title1, 0, 'C')
        self.set_y(self.get_y() + 2)
        self.line(self.get_x(), self.get_y(), self.get_x() + col_width, self.get_y())
        self.ln(2)
        self.set_font('DejaVu', '', 9)
        self.set_text_color(51, 51, 51)
        self.multi_cell(col_width, 5, content1, 0, 'L')
        y1 = self.get_y()
        
        # Resetear a la posición inicial para la segunda columna
        self.set_y(y_initial)
        self.set_x(self.l_margin + col_width + 10)

        # Columna 2
        self.set_font('DejaVu', 'B', 12)
        self.set_text_color(34, 49, 63)
        self.multi_cell(col_width, 8, title2, 0, 'C')
        self.set_y(self.get_y() + 2)
        self.line(self.get_x(), self.get_y(), self.get_x() + col_width, self.get_y())
        self.ln(2)
        self.set_font('DejaVu', '', 9)
        self.set_text_color(51, 51, 51)
        self.multi_cell(col_width, 5, content2, 0, 'L')
        y2 = self.get_y()
        
        # Mover a la posición Y más baja para continuar el flujo del documento
        self.set_y(max(y1, y2))
        self.ln(10)

def generate_pdf_in_memory(token, medico, deepseek, gemini, summary, comparison):
    """Genera un PDF profesional multi-página en memoria."""

    pdf = PDF('P', 'mm', 'A4')
    pdf.alias_nb_pages()

    # --- PÁGINA 1: DATOS Y DIAGNÓSTICOS DEL SISTEMA ---
    pdf.add_page()
    
    # Extraer secciones del informe del médico
    info_paciente = re.search(r'SECCION_INFO_PACIENTE\n(.*?)\nSECCION_FIN', medico, re.DOTALL).group(1)
    hallazgos_clave = re.search(r'SECCION_HALLAZGOS_CLAVE\n(.*?)\nSECCION_FIN', medico, re.DOTALL).group(1)
    diagnosticos = re.search(r'SECCION_DIAGNOSTICOS_SISTEMA\n(.*?)\nSECCION_FIN', medico, re.DOTALL).group(1)

    pdf.section_title('Datos del Paciente y Examen')
    pdf.section_body(info_paciente.strip())
    
    pdf.section_title('Resumen de Hallazgos Anormales (Sistema)')
    pdf.section_body(hallazgos_clave.strip())

    pdf.section_title('Diagnósticos y Recomendaciones Registrados')
    pdf.section_body(diagnosticos.strip())

    # --- PÁGINA 2: RESUMEN EJECUTIVO DE IA ---
    pdf.add_page()
    pdf.section_title('Resumen Ejecutivo (Análisis Sintetizado por IA)')
    pdf.section_body(summary)

    # --- PÁGINA 3: ANÁLISIS IA LADO A LADO ---
    pdf.add_page(orientation='L')
    pdf.print_two_column_layout('Análisis de DeepSeek', deepseek, 'Análisis de Gemini', gemini)
    
    # --- PÁGINA 4: COMPARACIÓN DETALLADA ---
    pdf.add_page()
    pdf.section_title('Análisis Comparativo Detallado de las IAs')
    pdf.section_body(comparison)

    return pdf.output()

# --- BLOQUE PRINCIPAL (Para pruebas locales, sin cambios) ---
if __name__ == '__main__':
    # ... (código sin cambios)
    pass