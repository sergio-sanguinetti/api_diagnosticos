# ==============================================================================
# SCRIPT DE ANÁLISIS MÉDICO COMPARATIVO CON IA Y GENERACIÓN DE PDF
#
# Descripción:
# Este script se conecta a una base de datos para obtener resultados médicos
# de un paciente, los analiza usando las APIs de DeepSeek y Gemini,
# compara los resultados de ambas IAs y genera un informe completo en PDF.
#
# Autor: Gemini (adaptado de los scripts originales)
# Fecha: 06/08/2025
# ==============================================================================

# ==============================================================================
# PASO 1: INSTALAR LIBRERÍAS NECESARIAS
# Antes de ejecutar, abre tu terminal o CMD y ejecuta:
# pip install mysql-connector-python requests google-generativeai fpdf2
# ==============================================================================
import mysql.connector
from mysql.connector import Error
import json
import requests
import google.generativeai as genai
from fpdf import FPDF
import sys
import textwrap

# ==============================================================================
# CONFIGURACIÓN DE CREDENCIALES
# ¡IMPORTANTE! Reemplaza con tus credenciales reales.
# Para mayor seguridad en producción, considera usar variables de entorno.
# ==============================================================================
DB_HOST = "193.203.175.193"
DB_USER = "u212843563_good_salud"
DB_PASS = "@9UbqRmS/oy"
DB_NAME = "u212843563_good_salud"
DEEPSEEK_API_KEY = "sk-37167855ce4243e8afe1ccb669021e64"  # Reemplazar si cambia
GOOGLE_API_KEY = "AIzaSyDqsYubkpT4Q_CofYluhK6lqmQHJui_U9A"   # Reemplazar si cambia

# ==============================================================================
# FUNCIÓN 1: CONEXIÓN A LA BASE DE DATOS
# ==============================================================================
def create_db_connection(host_name, user_name, user_password, db_name):
    """Crea y devuelve un objeto de conexión a la base de datos MySQL."""
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
    except Error as e:
        print(f"❌ Error al conectar a la base de datos: '{e}'")
    return connection

# ==============================================================================
# FUNCIÓN 2: EXTRACCIÓN Y FORMATEO DE DATOS DEL PACIENTE (INFORME MÉDICO)
# ==============================================================================
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

        report = f"""**Información del Paciente y Examen:**
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
        return report
    except Error as e:
        return f"❌ Error al consultar la base de datos: {e}"
    finally:
        cursor.close()

# ==============================================================================
# FUNCIÓN 3: PROMPT ESTANDARIZADO PARA LAS IAs
# ==============================================================================
def get_standard_prompt(report):
    """Crea un prompt estandarizado para asegurar respuestas consistentes."""
    return f"""
    **Rol:** Eres un asistente médico experto en medicina ocupacional.

    **Tarea:** Analiza el siguiente informe de resultados de un examen médico. Tu objetivo es identificar hallazgos anormales, correlacionarlos y proponer posibles diagnósticos diferenciales, junto con recomendaciones. NO inventes valores ni información; básate únicamente en los datos proporcionados.

    **Informe para analizar:**
    {report}

    **Formato de Respuesta Requerido (usa Markdown):**

    ### Resumen General del Paciente
    (Describe en 1-2 frases el estado general del paciente basado en los resultados).

    ### Hallazgos Clave
    (Usa una lista con viñetas para enumerar TODOS los resultados anormales. Ej: - Triglicéridos: 280 mg/dL (Resultado: anormal)).

    ### Análisis y Correlación Diagnóstica
    (Explica qué podrían significar los hallazgos anormales en conjunto. Correlaciona los datos entre sí).

    ### Análisis por Examen y Posibles Diagnósticos
    (Para cada hallazgo clave, explica su significado y posibles diagnósticos asociados. Ej:
    **- Perfil Lipídico (Colesterol y Triglicéridos):**
      - El colesterol total y los triglicéridos elevados sugieren un posible diagnóstico de **Dislipidemia**.
    **- Índice de Masa Corporal (IMC):**
      - Un IMC de 28.5 indica **Sobrepeso**, un factor de riesgo para otras condiciones.
    )

    ### Recomendaciones Sugeridas
    (Sugiere los siguientes pasos, como consultar a un especialista, cambios en el estilo de vida o pruebas de seguimiento).
    """

# ==============================================================================
# FUNCIÓN 4: ANÁLISIS CON DEEPSEEK
# ==============================================================================
def analyze_with_deepseek(report, api_key):
    """Envía el informe a la API de DeepSeek para su análisis."""
    prompt = get_standard_prompt(report)
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Eres un asistente médico experto en medicina ocupacional."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"❌ Error de conexión con la API de DeepSeek: {e}"
    except Exception as e:
        return f"❌ Error inesperado con DeepSeek: {e}"

# ==============================================================================
# FUNCIÓN 5: ANÁLISIS CON GEMINI
# ==============================================================================
def analyze_with_gemini(report, api_key):
    """Envía el informe a la API de Google Gemini para su análisis."""
    prompt = get_standard_prompt(report)
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error al comunicarse con la API de Gemini: {e}"

# ==============================================================================
# FUNCIÓN 6: COMPARAR ANÁLISIS DE IAS
# ==============================================================================
def compare_ai_analyses(deepseek_analysis, gemini_analysis, api_key):
    """Usa a Gemini para comparar las dos respuestas de la IA."""
    prompt = f"""
    **Rol:** Eres un médico supervisor y auditor de calidad de informes de IA.

    **Tarea:** Compara los dos análisis médicos generados por IA que te proporciono. Evalúa su similitud, coherencia y exhaustividad. NO evalúes el informe médico original, solo las dos respuestas de IA.

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
    (Describe en 2-3 frases si los análisis son mayormente similares o si presentan diferencias significativas).

    ### Puntos en Común
    (Usa una lista para enumerar los diagnósticos, hallazgos clave y recomendaciones en los que ambas IAs coinciden).

    ### Diferencias Notables
    (Usa una lista para señalar cualquier diagnóstico, recomendación o detalle que una IA mencionó y la otra omitió).

    ### Evaluación de Calidad y Conclusión
    (Indica cuál de los dos informes te parece más completo o útil y por qué. Si son de calidad similar, menciónalo).
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error al generar la comparación con la IA: {e}"

# ==============================================================================
# FUNCIÓN 7: GENERACIÓN DEL INFORME PDF
# ==============================================================================
class PDF(FPDF):
    def header(self):
        self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        self.set_font('DejaVu', '', 14)
        self.cell(0, 10, 'Reporte Comparativo de Análisis Médico', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', '', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('DejaVu', '', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, text):
        # Limpiar y preparar texto
        text = text.replace('**', '').replace('### ', '').replace('**', '')
        self.set_font('DejaVu', '', 9)
        self.multi_cell(0, 5, text)
        self.ln()

def generate_pdf_report(token, medico, deepseek, gemini, comparacion):
    """Genera el informe completo en formato PDF."""
    pdf = PDF()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)

    # --- PÁGINA 1: VISTA HORIZONTAL COMPARATIVA ---
    pdf.add_page(orientation='L')
    pdf.set_font('DejaVu', '', 10)

    col_width = (pdf.w - 30) / 3
    margin_height = pdf.h - 20

    # Función para limpiar el texto para el PDF
    def clean_text(text):
        return text.replace('**', '').replace('### ', '').encode('latin-1', 'replace').decode('latin-1')

    # Columna 1: Resultados del Médico
    pdf.x = 10
    pdf.y = 25
    pdf.set_font('DejaVu', '', 11)
    pdf.multi_cell(col_width, 6, "Resultados del Médico (BD)", border=1, align='C')
    pdf.set_font('DejaVu', '', 8)
    pdf.y += 1
    pdf.x = 10
    pdf.multi_cell(col_width, 4, clean_text(medico), border='LRB', align='L')

    # Columna 2: Análisis de DeepSeek
    pdf.x = 15 + col_width
    pdf.y = 25
    pdf.set_font('DejaVu', '', 11)
    pdf.multi_cell(col_width, 6, "Análisis de DeepSeek", border=1, align='C')
    pdf.set_font('DejaVu', '', 8)
    pdf.y += 1
    pdf.x = 15 + col_width
    pdf.multi_cell(col_width, 4, clean_text(deepseek), border='LRB', align='L')

    # Columna 3: Análisis de Gemini
    pdf.x = 20 + (col_width * 2)
    pdf.y = 25
    pdf.set_font('DejaVu', '', 11)
    pdf.multi_cell(col_width, 6, "Análisis de Gemini", border=1, align='C')
    pdf.set_font('DejaVu', '', 8)
    pdf.y += 1
    pdf.x = 20 + (col_width * 2)
    pdf.multi_cell(col_width, 4, clean_text(gemini), border='LRB', align='L')


    # --- PÁGINA 2: ANÁLISIS DE SIMILITUD ---
    pdf.add_page(orientation='P')
    pdf.chapter_title('Análisis Comparativo y de Similitud entre IAs')
    pdf.chapter_body(comparacion)

    # Guardar el PDF
    file_name = f"informe_comparativo_{token}.pdf"
    pdf.output(file_name)
    return file_name

# ==============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==============================================================================
def main():
    """Flujo principal que orquesta todo el proceso."""
    # Verificar si se pasó el token como argumento
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No se proporcionó un token de resultado."}, indent=4))
        sys.exit(1)
    
    patient_token = sys.argv[1]
    
    final_results = {}
    
    db_connection = create_db_connection(DB_HOST, DB_USER, DB_PASS, DB_NAME)
    if not db_connection:
        final_results['error'] = "No se pudo establecer conexión con la base de datos."
        print(json.dumps(final_results, indent=4, ensure_ascii=False))
        sys.exit(1)

    try:
        # 1. Obtener informe del médico desde la BD
        medico_report = get_patient_results(db_connection, patient_token)
        if "Error" in medico_report or "No se encontraron" in medico_report:
             final_results['error'] = medico_report
             print(json.dumps(final_results, indent=4, ensure_ascii=False))
             return

        final_results['medico'] = medico_report

        # 2. Analizar con ambas IAs
        deepseek_analysis = analyze_with_deepseek(medico_report, DEEPSEEK_API_KEY)
        final_results['deepseek'] = deepseek_analysis

        gemini_analysis = analyze_with_gemini(medico_report, GOOGLE_API_KEY)
        final_results['gemini'] = gemini_analysis

        # 3. Comparar los resultados de las IAs
        comparison_analysis = "No se pudo generar comparación debido a un error en los análisis previos."
        if "Error" not in deepseek_analysis and "Error" not in gemini_analysis:
            comparison_analysis = compare_ai_analyses(deepseek_analysis, gemini_analysis, GOOGLE_API_KEY)
        final_results['comparacion'] = comparison_analysis

        # 4. Generar el PDF
        pdf_filename = generate_pdf_report(
            patient_token,
            medico_report,
            deepseek_analysis,
            gemini_analysis,
            comparison_analysis
        )
        final_results['pdf_generado'] = pdf_filename

    except Exception as e:
        final_results['error'] = f"Ocurrió un error general en el script: {e}"
    
    finally:
        db_connection.close()
        # Imprimir el resultado final como JSON para que PHP lo capture
        print(json.dumps(final_results, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()