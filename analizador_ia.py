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


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint para monitoreo del servicio."""
    return jsonify({
        "status": "healthy",
        "service": "API de Diagnósticos Médicos",
        "version": "1.0.0",
        "timestamp": __import__('datetime').datetime.now().isoformat()
    })

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
    
    # Configurar timeout para evitar que el worker se cuelgue
    import signal
    import time
    
    def timeout_handler(signum, frame):
        raise TimeoutError("El procesamiento del reporte excedió el tiempo límite")
    
    # Configurar timeout de 5 minutos (300 segundos)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)
    
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
        deepseek_analysis = motor_analisis.analyze_with_deepseek(medico_report, motor_analisis.DEEPSEEK_API_KEY)
        gemini_analysis = motor_analisis.analyze_with_gemini(medico_report, motor_analisis.GOOGLE_API_KEY)
        final_results['deepseek'] = deepseek_analysis
        final_results['gemini'] = gemini_analysis
        
        # 4. Generar el Resumen Ejecutivo y la Comparación Detallada
        summary_analysis = motor_analisis.generate_executive_summary(deepseek_analysis, gemini_analysis, motor_analisis.GOOGLE_API_KEY)
        comparison_analysis = motor_analisis.compare_ai_analyses(deepseek_analysis, gemini_analysis, motor_analisis.GOOGLE_API_KEY)

        # 5. --- CALCULAR MÉTRICAS---
        metrics = {}
        
        # Verificar si se debe calcular similitud semántica (opcional para evitar timeouts)
        enable_semantic_similarity = os.environ.get('ENABLE_SEMANTIC_SIMILARITY', 'true').lower() == 'true'
        
        if enable_semantic_similarity:
            # Similitud semántica (con manejo de errores)
            try:
                print("🔄 Calculando similitud semántica para DeepSeek...")
                metrics['deepseek_similarity'] = motor_analisis.calculate_semantic_similarity(medico_report, deepseek_analysis)
            except Exception as e:
                print(f"⚠️ Error calculando similitud semántica para DeepSeek: {e}")
                metrics['deepseek_similarity'] = 0.0
                
            try:
                print("🔄 Calculando similitud semántica para Gemini...")
                metrics['gemini_similarity'] = motor_analisis.calculate_semantic_similarity(medico_report, gemini_analysis)
            except Exception as e:
                print(f"⚠️ Error calculando similitud semántica para Gemini: {e}")
                metrics['gemini_similarity'] = 0.0
        else:
            print("⚠️ Similitud semántica deshabilitada por configuración")
            metrics['deepseek_similarity'] = 0.0
            metrics['gemini_similarity'] = 0.0
        
        # Nuevas métricas: Kappa Cohen (con manejo de errores)
        try:
            metrics['deepseek_kappa'] = motor_analisis.calculate_kappa_cohen(medico_report, deepseek_analysis)
        except Exception as e:
            print(f"⚠️ Error calculando Kappa Cohen para DeepSeek: {e}")
            metrics['deepseek_kappa'] = 0.0
            
        try:
            metrics['gemini_kappa'] = motor_analisis.calculate_kappa_cohen(medico_report, gemini_analysis)
        except Exception as e:
            print(f"⚠️ Error calculando Kappa Cohen para Gemini: {e}")
            metrics['gemini_kappa'] = 0.0
        
        # Nuevas métricas: Similitud de Jaccard (con manejo de errores)
        try:
            metrics['deepseek_jaccard'] = motor_analisis.calculate_jaccard_similarity(medico_report, deepseek_analysis)
        except Exception as e:
            print(f"⚠️ Error calculando Jaccard para DeepSeek: {e}")
            metrics['deepseek_jaccard'] = 0.0
            
        try:
            metrics['gemini_jaccard'] = motor_analisis.calculate_jaccard_similarity(medico_report, gemini_analysis)
        except Exception as e:
            print(f"⚠️ Error calculando Jaccard para Gemini: {e}")
            metrics['gemini_jaccard'] = 0.0

        # 6. Generar el PDF directamente en memoria
        print("🔄 Generando PDF en memoria...")
        pdf_bytes = motor_analisis.generate_pdf_in_memory(
            token,
            final_results.get('medico', 'No disponible'),
            final_results.get('deepseek', 'No disponible'),
            final_results.get('gemini', 'No disponible'),
            summary_analysis, # Argumento 5: el nuevo resumen
            comparison_analysis,
             metrics
        )
        
        # Limpiar variables grandes para liberar memoria
        del final_results
        del summary_analysis
        del comparison_analysis
        del metrics
        import gc
        gc.collect()
        print("✅ PDF generado exitosamente")

        # 7. Crear y devolver la respuesta de Flask como un archivo para descargar
        return Response(
            bytes(pdf_bytes),
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment;filename=informe_comparativo_{token}.pdf"}
        )

    except TimeoutError as e:
        print(f"⏰ Timeout en el procesamiento del reporte: {e}")
        return jsonify({"error": "El procesamiento del reporte excedió el tiempo límite. Por favor, intente nuevamente."}), 408
    except Exception as e:
        # Captura cualquier otro error para dar una respuesta clara
        import traceback
        traceback.print_exc() # Imprime el error detallado en los logs de Render
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {str(e)}"}), 500
    
    finally:
        # Cancelar el timeout
        signal.alarm(0)
        # Asegurarse de cerrar la conexión a la base de datos
        if db_connection and db_connection.is_connected():
            db_connection.close()


# Punto de entrada para desarrollo local
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)