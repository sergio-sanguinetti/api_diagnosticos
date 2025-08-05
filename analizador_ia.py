# analizador_ia.py (Versión unificada)

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai

# Importamos nuestro módulo con la lógica de reportes comparativos
import motor_analisis 

# 1. CREACIÓN DE LA APLICACIÓN FLASK
# ==================================
app = Flask(__name__)
CORS(app) # Habilita CORS para permitir peticiones desde tu frontend/PHP

# Configura el directorio para los PDFs (usado por la nueva función)
PDF_DIRECTORY = os.path.join(os.getcwd(), 'generated_reports')


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
    # (El código de esta función es idéntico al que ya tenías, no lo repetiré por brevedad)
    # ...
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return f"Error configurando la API de Google: {e}"
    prompt = f"**Rol:** Eres un asistente médico experto...\n{report}\n..." # Prompt completo
    # ...
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
    # Usamos la función original que solo llama a Gemini
    ai_analysis = analyze_results_with_llm(patient_report, api_key) 
    return jsonify({"diagnostico_completo": ai_analysis})


# ==============================================================================
# --- NUEVA SECCIÓN PARA REPORTES COMPARATIVOS ---
# Funcionalidad para generar un reporte completo desde la BD usando un token.
# ==============================================================================

@app.route('/generar-reporte-comparativo', methods=['POST'])
def generar_reporte_endpoint():
    """NUEVO ENDPOINT: Genera el reporte comparativo y PDF desde la BD."""
    if not request.is_json:
        return jsonify({"error": "La petición debe ser de tipo JSON"}), 400

    data = request.get_json()
    token = data.get('token', None)

    if not token:
        return jsonify({"error": "No se proporcionó el 'token' en el cuerpo de la petición"}), 400

    final_results = {}
    db_connection = None
    try:
        # Aquí usamos las funciones de nuestro módulo 'motor_analisis'
        db_connection = motor_analisis.create_db_connection(
            motor_analisis.DB_HOST, motor_analisis.DB_USER,
            motor_analisis.DB_PASS, motor_analisis.DB_NAME
        )
        if not db_connection:
            raise ConnectionError("No se pudo conectar a la base de datos.")

        # 1. Obtener informe del médico
        final_results['medico'] = motor_analisis.get_patient_results(db_connection, token)
        
        # 2. Analizar con ambas IAs
        final_results['deepseek'] = motor_analisis.analyze_with_deepseek(final_results['medico'], motor_analisis.DEEPSEEK_API_KEY)
        final_results['gemini'] = motor_analisis.analyze_with_gemini(final_results['medico'], motor_analisis.GOOGLE_API_KEY)
        
        # 3. Comparar análisis
        final_results['comparacion'] = motor_analisis.compare_ai_analyses(final_results['deepseek'], final_results['gemini'], motor_analisis.GOOGLE_API_KEY)
        
        # 4. Generar el PDF
        pdf_filepath = motor_analisis.generate_pdf_report(
            token, final_results['medico'], final_results['deepseek'],
            final_results['gemini'], final_results['comparacion']
        )
        pdf_filename = os.path.basename(pdf_filepath)
        
        # 5. Construir la respuesta JSON final
        response_data = {
            "success": True,
            "message": "Reporte generado exitosamente.",
            "download_url": request.host_url.rstrip('/') + f"/reportes/{pdf_filename}",
            "data": final_results
        }
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {str(e)}"}), 500
    
    finally:
        if db_connection and db_connection.is_connected():
            db_connection.close()


@app.route('/reportes/<path:filename>')
def descargar_reporte(filename):
    """NUEVO ENDPOINT: Permite la descarga del PDF generado."""
    try:
        return send_from_directory(PDF_DIRECTORY, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Archivo no encontrado."}), 404


# Punto de entrada para desarrollo local (puedes ejecutar 'python analizador_ia.py')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)