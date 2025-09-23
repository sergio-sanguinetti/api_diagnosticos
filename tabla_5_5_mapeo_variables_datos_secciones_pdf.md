# Tabla 5.5: Mapeo de Variables de Datos a Secciones del Documento PDF

## Resumen
Esta tabla documenta la correspondencia entre las variables de datos extraídas de la base de datos MySQL y las secciones específicas del documento PDF generado por el sistema de análisis médico ocupacional asistido por IA.

## Tabla de Mapeo

| **Variable de Datos** | **Tipo de Dato** | **Sección PDF** | **Página** | **Descripción de Uso** |
|------------------------|------------------|-----------------|------------|------------------------|
| **INFORMACIÓN DEL PACIENTE Y EXAMEN** | | | | |
| `centro_medico` | VARCHAR(255) | Datos del Paciente y Examen | 1 | Nombre del centro médico donde se realizó el examen |
| `ciudad` | VARCHAR(255) | Datos del Paciente y Examen | 1 | Ciudad donde se realizó el examen |
| `fecha_examen` | DATE | Datos del Paciente y Examen | 1 | Fecha en que se realizó el examen médico |
| `puesto` | VARCHAR(255) | Datos del Paciente y Examen | 1 | Puesto de trabajo del paciente |
| `tipo_examen` | VARCHAR(255) | Datos del Paciente y Examen | 1 | Tipo de examen médico ocupacional |
| `aptitud` | VARCHAR(255) | Datos del Paciente y Examen | 1 | Aptitud médica declarada |
| **RESULTADOS DE LABORATORIO** | | | | |
| `presion_a` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Presión arterial sistólica/diastólica |
| `resultado_presion_a` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del resultado de presión arterial |
| `glucosa` | DECIMAL(5,2) | Resumen de Hallazgos Anormales | 1 | Nivel de glucosa en sangre (mg/dL) |
| `resultado_glucosa` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del resultado de glucosa |
| `colesterol_total` | DECIMAL(5,2) | Resumen de Hallazgos Anormales | 1 | Colesterol total (mg/dL) |
| `resultado_colesterol_total` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del colesterol total |
| `hdl_colesterol` | DECIMAL(5,2) | Resumen de Hallazgos Anormales | 1 | Colesterol HDL (mg/dL) |
| `resultado_hdl_colesterol` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del colesterol HDL |
| `ldl_colesterol` | DECIMAL(5,2) | Resumen de Hallazgos Anormales | 1 | Colesterol LDL (mg/dL) |
| `resultado_ldl_colesterol` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del colesterol LDL |
| `trigliceridos` | DECIMAL(5,2) | Resumen de Hallazgos Anormales | 1 | Triglicéridos (mg/dL) |
| `resultado_trigliceridos` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación de triglicéridos |
| `ac_urico` | DECIMAL(5,2) | Resumen de Hallazgos Anormales | 1 | Ácido úrico (mg/dL) |
| `resultado_ac_urico` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del ácido úrico |
| `hemoglobina` | DECIMAL(5,2) | Resumen de Hallazgos Anormales | 1 | Hemoglobina (g/dL) |
| `resultado_hemoglobina` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación de hemoglobina |
| `rpr` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Prueba RPR (sífilis) |
| `resultado_rpr` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del RPR |
| `examen_orina` | VARCHAR(255) | Resumen de Hallazgos Anormales | 1 | Examen de orina completo |
| `resultado_examen_orina` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del examen de orina |
| `radiografia_torax` | VARCHAR(255) | Resumen de Hallazgos Anormales | 1 | Radiografía de tórax |
| `resultado_radiografia_torax` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación de radiografía de tórax |
| `audiometria` | VARCHAR(255) | Resumen de Hallazgos Anormales | 1 | Prueba de audiometría |
| `resultado_audiometria` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación de audiometría |
| `espirometria` | VARCHAR(255) | Resumen de Hallazgos Anormales | 1 | Prueba de espirometría |
| `resultado_espirometria` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación de espirometría |
| `electrocardiograma` | VARCHAR(255) | Resumen de Hallazgos Anormales | 1 | Electrocardiograma |
| `resultado_electrocardiograma` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del electrocardiograma |
| `indice_c_c` | DECIMAL(4,2) | Resumen de Hallazgos Anormales | 1 | Índice cintura-cadera |
| `resultado_indice_c_c` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del índice cintura-cadera |
| `indice_m_c` | DECIMAL(4,2) | Resumen de Hallazgos Anormales | 1 | Índice de masa corporal (IMC) |
| `resultado_indice_m_c` | VARCHAR(50) | Resumen de Hallazgos Anormales | 1 | Interpretación del IMC |
| **DIAGNÓSTICOS Y RECOMENDACIONES** | | | | |
| `diagnosticos` | JSON | Diagnósticos y Recomendaciones Registrados | 1 | Array de objetos con diagnósticos y recomendaciones del sistema médico |
| **ANÁLISIS DE INTELIGENCIA ARTIFICIAL** | | | | |
| `deepseek_analysis` | TEXT | Análisis Detallado de DeepSeek | 3 | Análisis médico generado por DeepSeek AI |
| `gemini_analysis` | TEXT | Análisis Detallado de Gemini | 3 | Análisis médico generado por Google Gemini AI |
| `summary_analysis` | TEXT | Resumen Ejecutivo | 2 | Resumen sintetizado por IA combinando ambos análisis |
| `comparison_analysis` | TEXT | Análisis Comparativo Detallado | 4 | Comparación detallada entre análisis de ambas IAs |
| **MÉTRICAS DE SIMILITUD** | | | | |
| `deepseek_similarity` | DECIMAL(4,4) | Métricas de Similitud y Concordancia | 6 | Similitud semántica entre análisis médico y DeepSeek (0.0-1.0) |
| `gemini_similarity` | DECIMAL(4,4) | Métricas de Similitud y Concordancia | 6 | Similitud semántica entre análisis médico y Gemini (0.0-1.0) |
| `deepseek_kappa` | DECIMAL(4,4) | Métricas de Similitud y Concordancia | 6 | Índice de Kappa Cohen para concordancia DeepSeek (0.0-1.0) |
| `gemini_kappa` | DECIMAL(4,4) | Métricas de Similitud y Concordancia | 6 | Índice de Kappa Cohen para concordancia Gemini (0.0-1.0) |
| `deepseek_jaccard` | DECIMAL(4,4) | Métricas de Similitud y Concordancia | 6 | Similitud de Jaccard para DeepSeek (0.0-1.0) |
| `gemini_jaccard` | DECIMAL(4,4) | Métricas de Similitud y Concordancia | 6 | Similitud de Jaccard para Gemini (0.0-1.0) |
| **TABLA COMPARATIVA** | | | | |
| `medico_pairs` | ARRAY | Tabla Comparativa de Diagnósticos y Recomendaciones | 5 | Pares diagnóstico-recomendación extraídos del sistema médico |
| `deepseek_pairs` | ARRAY | Tabla Comparativa de Diagnósticos y Recomendaciones | 5 | Pares diagnóstico-recomendación extraídos de DeepSeek |
| `gemini_pairs` | ARRAY | Tabla Comparativa de Diagnósticos y Recomendaciones | 5 | Pares diagnóstico-recomendación extraídos de Gemini |

## Estructura del Documento PDF

### Página 1: Datos del Sistema Médico
- **Datos del Paciente y Examen**: Información básica del paciente y contexto del examen
- **Resumen de Hallazgos Anormales**: Resultados de laboratorio con valores anormales
- **Diagnósticos y Recomendaciones Registrados**: Diagnósticos del sistema médico en formato estructurado

### Página 2: Resumen Ejecutivo
- **Resumen Ejecutivo**: Análisis sintetizado por IA combinando ambos modelos

### Página 3: Análisis Comparativo (Orientación Horizontal)
- **Análisis Detallado de DeepSeek**: Análisis completo generado por DeepSeek AI
- **Análisis Detallado de Gemini**: Análisis completo generado por Google Gemini AI

### Página 4: Comparación Detallada
- **Análisis Comparativo Detallado**: Comparación textual entre ambos análisis de IA

### Página 5: Tabla Comparativa (Orientación Horizontal)
- **Tabla Comparativa de Diagnósticos y Recomendaciones**: Comparación lado a lado de diagnósticos y recomendaciones de las tres fuentes (Sistema Médico, DeepSeek, Gemini)

### Página 6: Métricas de Similitud
- **Métricas de Similitud y Concordancia**: Métricas cuantitativas de concordancia entre el análisis médico y cada IA
- **Tabla Comparativa de Métricas**: Resumen comparativo de todas las métricas calculadas

## Notas Técnicas

1. **Procesamiento de Datos**: Las variables de la base de datos se procesan mediante la función `get_patient_results()` que extrae y formatea los datos según las secciones del PDF.

2. **Estructura JSON**: El campo `diagnosticos` contiene un array JSON con objetos que incluyen `diagnostico` y `recomendacion`.

3. **Métricas Calculadas**: Las métricas de similitud se calculan usando:
   - Similitud semántica con DeepSeek API
   - Índice de Kappa Cohen para concordancia
   - Similitud de Jaccard para comparación de conjuntos

4. **Formato de Salida**: El PDF se genera en memoria usando la librería FPDF con fuentes personalizadas (DejaVu Sans) y diseño profesional.

5. **Límites de Memoria**: Se implementan límites de texto (5000 caracteres) para evitar problemas de memoria durante la generación del PDF.

## Dependencias del Sistema

- **Base de Datos**: MySQL con tabla `resultados`
- **APIs de IA**: DeepSeek API y Google Gemini API
- **Librerías**: FPDF, mysql-connector-python, google-generativeai, requests
- **Modelo de Embeddings**: Hugging Face sentence-transformers/all-MiniLM-L6-v2
