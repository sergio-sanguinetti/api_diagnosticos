# analizador_ia.py

import sys
import json
import google.generativeai as genai

# ==============================================================================
# FUNCIÓN 1: FORMATEO DE DATOS (REEMPLAZA la conexión a la BD)
# ==============================================================================
def format_report_from_json(data):
    """Formatea los datos recibidos del formulario PHP en un informe para la IA."""
    
    # Mapeo de nombres de formulario a nombres legibles
    parametros_map = {
        'presion_arterial': 'Presión Arterial',
        'glucosa': 'Glucosa',
        'colesterol_total': 'Colesterol Total',
        'hdl_colesterol': 'HDL Colesterol',
        'ldl_colesterol': 'LDL Colesterol',
        'trigliceridos': 'Triglicéridos',
        'ac_urico': 'Ac Úrico',
        'hemoglobina': 'Hemoglobina',
        'rpr': 'RPR',
        'examen_orina': 'Examen de orina',
        'radiografia_torax': 'Radiografía Tórax',
        'audiometria': 'Audiometría',
        'espirometria': 'Espirometría',
        'electrocardiograma': 'Electrocardiograma',
        'indice_c_c': 'Índice Cintura / Cadera',
        'indice_m_c': 'Índice de Masa Corporal'
    }

    resultados_str = ""
    for campo, nombre in parametros_map.items():
        valor = data.get(f'valor_{campo}', 'N/A')
        resultado = data.get(f'resultado_{campo}', 'N/A')
        if valor or resultado != 'no_realizado':
             resultados_str += f"- {nombre}: {valor} (Resultado: {resultado})\n"

    report = f"""
INFORME DE ANÁLISIS MÉDICO OCUPACIONAL

**Información del Paciente y Examen:**
- Centro Médico: {data.get('centro_medico', 'N/A')}
- Ciudad: {data.get('ciudad', 'N/A')}
- Fecha de Examen: {data.get('fecha_examen', 'N/A')}
- Puesto de Trabajo: {data.get('puesto', 'N/A')}
- Tipo de Examen: {data.get('tipo_examen', 'N/A')}
- Aptitud Declarada: {data.get('aptitud', 'N/A')}

**Resultados de Pruebas y Mediciones:**
{resultados_str}
"""
    return report

# ==============================================================================
# FUNCIÓN 2: ANÁLISIS CON EL MODELO DE LENGUAJE (LLM) - Sin cambios
# ==============================================================================
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
    (Usa una lista con viñetas para enumerar todos los resultados marcados como 'anormal' o que estén claramente fuera de rangos normales. Ej: - Triglicéridos: 280 mg/dL (Resultado: anormal)).

    ### Análisis y Correlación Diagnóstica
    (Explica qué podrían significar los hallazgos anormales en conjunto. Correlaciona los datos entre sí, por ejemplo, cómo el IMC puede influir en el perfil lipídico).

    ### Análisis por Examen y Posibles Diagnósticos
    (Para cada examen con un hallazgo anormal de la sección "Hallazgos Clave", crea una subsección. Dentro de cada subsección, explica qué significa el resultado anormal y qué posibles diagnósticos sugiere. Asocia claramente cada diagnóstico al resultado del examen correspondiente. Por ejemplo:
    **- Perfil Lipídico (Colesterol y Triglicéridos):**
      - El colesterol total y los triglicéridos elevados sugieren un posible diagnóstico de **Dislipidemia** o **Hipertrigliceridemia**. Esto aumenta el riesgo cardiovascular.
    **- Índice de Masa Corporal (IMC):**
      - Un IMC de 28.5 indica **Sobrepeso**, lo que puede contribuir a la dislipidemia y la hipertensión.
    )

    ### Recomendaciones Sugeridas
    (Sugiere los siguientes pasos, como consultar a un especialista, cambios en el estilo de vida o pruebas de seguimiento, basándote en el análisis anterior).
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error al generar contenido con la IA: {e}"

# ==============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==============================================================================
def main():
    """Flujo principal que orquesta todo el proceso."""

    # Lee el JSON que se le pasa como primer argumento desde la línea de comandos
    try:
        form_data = json.loads(sys.argv[1])
    except (IndexError, json.JSONDecodeError):
        print("Error: No se proporcionaron datos válidos en formato JSON.")
        sys.exit(1)

    # --- ¡CONFIGURAR AQUÍ! ---
    # Es más seguro leer la API key desde una variable de entorno en un entorno de producción
    GOOGLE_API_KEY = "AIzaSyDqsYubkpT4Q_CofYluhK6lqmQHJui_U9A"  
    # -------------------------

    # 1. Formatear el informe a partir de los datos del formulario
    patient_report = format_report_from_json(form_data)

    # 2. Enviar a la IA para análisis
    ai_analysis = analyze_results_with_llm(patient_report, GOOGLE_API_KEY)

    # 3. Imprimir el resultado para que PHP lo capture
    print(ai_analysis)

if __name__ == '__main__':
    main()