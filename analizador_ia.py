# ==============================================================================
# API FLASK PARA ANÁLISIS DE DIAGNÓSTICOS MÉDICOS
#==============================================================================
import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import google.generativeai as genai

# Importamos nuestro módulo con la lógica de reportes comparativos
import motor_analisis 

# 1. CREACIÓN DE LA APLICACIÓN FLASK
# ==================================
app = Flask(__name__)
CORS(app) # Habilita CORS para permitir peticiones desde tu frontend/PHP


# ==============================================================================
# --- SECCIÓN ORIGINAL (SIN CAMBIOS) ---
# Funcionalidad para analizar datos enviados directamente desde un formulario.
# ==============================================================================

def format_report_from_json(data):
    """Formatea los datos recibidos del formulario PHP en un informe para la IA."""
    parametros_map = {
        'presion_arterial': 'Presión Arterial', 'glucosa': 'Glucosa', 
        'colesterol_total': 'Colesterol Total', 'hdl_colesterol': 'HDL Colesterol',
        'ldl_colesterol': 'LDL Colesterol', 'trigliceridos': 'Triglicéridos',
        'ac_urico': 'Ac Úrico', 'hemoglobina': 'Hemoglobina', 'rpr': 'RPR',
        'examen_orina': 'Examen de orina', 'radiografia_torax': 'Radiografía Tórax',
        'audiometria': 'Audiometría', 'espirometria': 'Espirometría',
        'electrocardiograma': 'Electrocardiograma', 'indice_c_c': 'Índice Cintura / Cadera',
        'indice_m_c': 'Índice de Masa Corporal'
    }
    resultados_str = ""
    for campo, nombre in parametros_map.items():
        valor = data.get(f'valor_{campo}', 'N/A')
        resultado = data.get(f'resultado_{campo}', 'N/A')
        if valor or resultado != 'no_realizado':
            resultados_str += f"- {nombre}: {valor} (Resultado: {resultado})\n"

    report = f"""INFORME DE ANÁLISIS MÉDICO OCUPACIONAL
**Información del Paciente y Examen:**
- Centro Médico: {data.get('centro_medico', 'N/A')}
- Ciudad: {data.get('ciudad', 'N/A')}
- Fecha de Examen: {data.get('fecha_examen', 'N/A')}
- Puesto de Trabajo: {data.get('puesto', 'N/A')}
- Tipo de Examen: {data.get('tipo_examen', 'N/A')}
- Aptitud Declarada: {data.get('aptitud', 'N/A')}
**Resultados de Pruebas y Mediciones:**
{resultados_str}"""
    return report

def analyze_results_with_llm(report, api_key):
    """Envía el informe a la API de Google Gemini para su análisis."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return f"Error configurando la API de Google: {e}"

    prompt = f"""
    **Rol:** Eres un asistente médico experto en medicina ocupacional.
    **Tarea:** Analiza el siguiente informe de resultados de un examen médico. Tu objetivo es identificar hallazgos anormales, correlacionarlos y proponer posibles diagnósticos diferenciales, junto con recomendaciones. NO inventes valores ni información; básate únicamente en los datos proporcionados.
    **Informe para analizar:**
    {report}
    **Formato de Respuesta Requerido (usa Markdown):**
    ### Resumen General del Paciente
    (Describe en 1-2 frases el estado general del paciente basado en los resultados).
    ### Hallazgos Clave
    (Usa una lista con viñetas para enumerar todos los resultados marcados como 'anormal' o que estén claramente fuera de rangos normales).
    ### Análisis y Correlación Diagnóstica
    (Explica qué podrían significar los hallazgos anormales en conjunto).
    ### Análisis por Examen y Posibles Diagnósticos
    (Para cada examen con un hallazgo anormal, explica qué significa y qué diagnósticos sugiere).
    ### Recomendaciones Sugeridas
    (Sugiere los siguientes pasos, como consultar a un especialista o cambios en el estilo de vida).
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error al generar contenido con la IA: {e}"


@app.route('/analizar', methods=['POST'])
def analizar_endpoint():
    """ENDPOINT ORIGINAL: Analiza datos de un formulario."""
    form_data = request.get_json()
    if not form_data:
        return jsonify({"error": "No se recibieron datos en formato JSON"}), 400
    
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        return jsonify({"error": "La variable de entorno GOOGLE_API_KEY no está configurada"}), 500

    patient_report = format_report_from_json(form_data)
    ai_analysis = analyze_results_with_llm(patient_report, api_key) 
    return jsonify({"diagnostico_completo": ai_analysis})


# ==============================================================================
# --- NUEVA SECCIÓN PARA REPORTES COMPARATIVOS ---
# ==============================================================================

@app.route('/generar-reporte-comparativo', methods=['POST'])
def generar_reporte_endpoint():
    """NUEVO ENDPOINT: Genera el reporte y lo devuelve directamente como descarga."""
    
    data = request.get_json()
    if not data or 'token' not in data:
        return jsonify({"error": "Petición inválida. Se requiere un 'token'."}), 400
    
    token = data.get('token')
    final_results = {}
    db_connection = None

    try:
        # 1. Conectar a la base de datos
        db_connection = motor_analisis.create_db_connection(
            motor_analisis.DB_HOST, motor_analisis.DB_USER,
            motor_analisis.DB_PASS, motor_analisis.DB_NAME
        )
        if not db_connection:
            raise ConnectionError("No se pudo conectar a la base de datos.")

        # 2. Obtener los datos del paciente
        medico_report = motor_analisis.get_patient_results(db_connection, token)
        if "Error" in medico_report or "No se encontraron" in medico_report:
            return jsonify({"error": medico_report}), 404
        final_results['medico'] = medico_report

        # 3. Realizar análisis con las IAs
        final_results['deepseek'] = motor_analisis.analyze_with_deepseek(medico_report, motor_analisis.DEEPSEEK_API_KEY)
        final_results['gemini'] = motor_analisis.analyze_with_gemini(medico_report, motor_analisis.GOOGLE_API_KEY)
        final_results['comparacion'] = motor_analisis.compare_ai_analyses(final_results['deepseek'], final_results['gemini'], motor_analisis.GOOGLE_API_KEY)

        # 4. Generar el PDF directamente en memoria
        pdf_bytes = motor_analisis.generate_pdf_in_memory(
            token,
            final_results.get('medico', 'No disponible'),
            final_results.get('deepseek', 'No disponible'),
            final_results.get('gemini', 'No disponible'),
            final_results.get('comparacion', 'No disponible')
        )

        # 5. Crear y devolver la respuesta de Flask como un archivo para descargar
        return Response(
            bytes(pdf_bytes),
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment;filename=informe_comparativo_{token}.pdf"}
        )

    except Exception as e:
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {str(e)}"}), 500
    
    finally:
        # Asegurarse de cerrar la conexión a la base de datos
        if db_connection and db_connection.is_connected():
            db_connection.close()


# Punto de entrada para desarrollo local
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)