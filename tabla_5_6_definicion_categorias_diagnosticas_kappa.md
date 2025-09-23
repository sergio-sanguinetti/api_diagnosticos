# Tabla 5.6: Definición de las Categorías Diagnósticas para el Cálculo del Índice de Kappa

## Resumen
Esta tabla define las categorías diagnósticas utilizadas en el sistema de análisis médico ocupacional para el cálculo del Índice de Kappa Cohen, que evalúa la concordancia entre el análisis del médico y los análisis generados por las inteligencias artificiales (DeepSeek y Gemini).

## Categorías Diagnósticas del Sistema

### 1. Categorías por Tipo de Examen Médico

| **Categoría** | **Descripción** | **Términos Clave** | **Ejemplos de Diagnósticos** |
|---------------|-----------------|-------------------|------------------------------|
| **Perfil Lipídico** | Trastornos relacionados con el metabolismo de lípidos | trigliceridemia, colesterol, lipídico | Hipertrigliceridemia, Hiperlipidemia, Dislipidemia, Colesterol elevado |
| **Examen de Orina** | Anomalías detectadas en el análisis de orina | orina, hematíes, microhematuria | Microhematuria, Proteinuria, Leucocituria, Cilindros |
| **Hemograma y Bioquímica** | Alteraciones en parámetros sanguíneos y bioquímicos | policitemia, bioquímica, neutropenia, hemoglobina, hemograma | Policitemia, Anemia, Neutropenia, Trombocitopenia |
| **Oftalmología** | Trastornos visuales y oculares | ametropía, oftalmología, lentes | Miopía, Hipermetropía, Astigmatismo, Presbicia |
| **Otros Diagnósticos** | Diagnósticos que no se clasifican en las categorías anteriores | - | Gastritis, Hipertensión, Diabetes, Bradicardia |

### 2. Categorías Normalizadas para Comparación

| **Categoría Normalizada** | **Términos de Búsqueda** | **Descripción** | **Diagnósticos Incluidos** |
|---------------------------|---------------------------|-----------------|----------------------------|
| **HIPERTRIGLICERIDEMIA** | trigliceridemia, dislipidemia | Trastornos del metabolismo de triglicéridos | Hipertrigliceridemia, Trigliceridemia, Dislipidemia mixta |
| **HIPERLIPIDEMIA** | hiperlipidemia, colesterol, ldl | Trastornos del metabolismo del colesterol | Hipercolesterolemia, LDL elevado, Hiperlipidemia |
| **POLICITEMIA** | policitemia | Aumento anormal de glóbulos rojos | Policitemia vera, Policitemia secundaria |
| **SOBREPESO** | sobrepeso, obesidad, imc | Trastornos del peso corporal | Sobrepeso, Obesidad, IMC elevado |
| **BRADICARDIA** | bradicardia, cardíaco | Alteraciones del ritmo cardíaco | Bradicardia sinusal, Arritmia cardíaca |
| **DEFICIENCIA_HDL** | hdl, deficiencia | Niveles bajos de colesterol HDL | HDL bajo, Deficiencia de HDL |
| **DIABETES** | diabetes, glucosa | Trastornos del metabolismo de la glucosa | Diabetes tipo 2, Intolerancia a la glucosa |
| **HIPERTENSIÓN** | hipertensión, presión | Trastornos de la presión arterial | Hipertensión arterial, HTA |
| **ANEMIA** | anemia, hemoglobina | Trastornos de la hemoglobina | Anemia ferropénica, Anemia megaloblástica |
| **GASTRITIS** | gastritis, gástrico | Trastornos del sistema digestivo | Gastritis crónica, Gastritis erosiva |

### 3. Términos Médicos para Extracción Automática

| **Categoría de Términos** | **Términos Incluidos** | **Uso en Kappa** |
|---------------------------|------------------------|------------------|
| **Cardiovasculares** | hipertensión, hipertensivo, presión arterial, tensión, bradicardia, frecuencia cardíaca, ritmo cardíaco | Evaluación de concordancia en diagnósticos cardiovasculares |
| **Metabólicos** | diabetes, glucosa, glicemia, hemoglobina glicosilada, dislipidemia, colesterol, triglicéridos, hdl, ldl | Evaluación de concordancia en trastornos metabólicos |
| **Hematológicos** | anemia, hemoglobina, hematocrito, eritrocitos, policitemia, policitemia secundaria | Evaluación de concordancia en trastornos sanguíneos |
| **Antropométricos** | sobrepeso, obesidad, índice masa corporal, imc | Evaluación de concordancia en parámetros antropométricos |
| **Digestivos** | gastritis, úlcera, reflujo, acidez | Evaluación de concordancia en trastornos digestivos |
| **Generales** | deficiencia, insuficiencia, disfunción, evaluación, seguimiento, control, monitoreo | Evaluación de concordancia en términos generales |

## Escalas de Interpretación del Índice de Kappa

### Escala de Landis y Koch (1977)

| **Valor de Kappa** | **Interpretación** | **Nivel de Concordancia** | **Aplicación Clínica** |
|--------------------|--------------------|---------------------------|------------------------|
| **< 0.00** | Sin acuerdo | Pobre | Los diagnósticos son contradictorios |
| **0.00 - 0.20** | Acuerdo insignificante | Ligero | Concordancia mínima, no confiable |
| **0.21 - 0.40** | Acuerdo bajo | Regular | Concordancia débil, requiere revisión |
| **0.41 - 0.60** | Acuerdo moderado | Moderado | Concordancia aceptable para uso clínico |
| **0.61 - 0.80** | Acuerdo bueno | Sustancial | Concordancia buena, confiable |
| **0.81 - 1.00** | Acuerdo muy bueno | Casi perfecto | Concordancia excelente, muy confiable |

### Escala Adaptada para IA Médica

| **Valor de Kappa** | **Interpretación IA** | **Recomendación** |
|--------------------|----------------------|-------------------|
| **0.80 - 1.00** | Excelente concordancia | IA altamente confiable para diagnóstico |
| **0.60 - 0.79** | Buena concordancia | IA confiable con supervisión médica |
| **0.40 - 0.59** | Concordancia moderada | IA útil como apoyo, requiere validación |
| **0.20 - 0.39** | Concordancia baja | IA limitada, uso con precaución |
| **0.00 - 0.19** | Concordancia muy baja | IA no recomendada para diagnóstico |

## Metodología de Cálculo

### 1. Extracción de Términos Médicos
- **Función**: `extract_medical_terms(text)`
- **Proceso**: Búsqueda de términos médicos predefinidos en texto en minúsculas
- **Normalización**: Conversión a minúsculas y eliminación de duplicados

### 2. Cálculo del Índice de Kappa
- **Fórmula**: κ = (Po - Pe) / (1 - Pe)
- **Po**: Probabilidad de acuerdo observado
- **Pe**: Probabilidad de acuerdo esperado (0.5 - valor conservador)
- **Rango**: [-1.0, 1.0]

### 3. Validación de Resultados
- **Verificación de rango**: Valores entre -1.0 y 1.0
- **Manejo de errores**: Retorno de 0.0 en caso de error
- **Logging**: Registro de errores para debugging

## Aplicación en el Sistema

### Comparaciones Realizadas
1. **Médico vs DeepSeek**: Concordancia entre diagnóstico médico y análisis de DeepSeek AI
2. **Médico vs Gemini**: Concordancia entre diagnóstico médico y análisis de Google Gemini AI
3. **DeepSeek vs Gemini**: Concordancia entre ambos modelos de IA

### Criterios de Evaluación
- **Coincidencia exacta**: Términos médicos idénticos
- **Coincidencia semántica**: Términos relacionados conceptualmente
- **Coincidencia categórica**: Términos de la misma categoría diagnóstica

## Limitaciones y Consideraciones

### Limitaciones del Sistema
1. **Lista finita de términos**: Solo evalúa términos predefinidos
2. **Simplificación de Pe**: Uso de valor conservador (0.5) para probabilidad esperada
3. **Contexto limitado**: No considera el contexto clínico completo
4. **Idioma**: Optimizado para términos en español

### Mejoras Futuras
1. **Expansión de vocabulario**: Inclusión de más términos médicos
2. **Normalización avanzada**: Uso de sinónimos y variantes
3. **Ponderación por importancia**: Asignación de pesos según relevancia clínica
4. **Análisis contextual**: Consideración del contexto semántico

## Referencias Técnicas

- **Landis, J. R., & Koch, G. G. (1977)**: "The measurement of observer agreement for categorical data"
- **Cohen, J. (1960)**: "A coefficient of agreement for nominal scales"
- **Fleiss, J. L. (1971)**: "Measuring nominal scale agreement among many raters"

## Implementación en Código

```python
def calculate_kappa_cohen(text_medico, text_ia):
    """Calcula el Índice de Kappa Cohen entre análisis médico e IA."""
    medico_terms = extract_medical_terms(text_medico)
    ia_terms = extract_medical_terms(text_ia)
    
    all_terms = set(medico_terms + ia_terms)
    agreed_terms = set(medico_terms) & set(ia_terms)
    
    po = len(agreed_terms) / len(all_terms) if len(all_terms) > 0 else 0
    pe = 0.5  # Valor conservador
    
    kappa = (po - pe) / (1 - pe) if pe != 1 else (1.0 if po == 1 else 0.0)
    return max(-1.0, min(1.0, kappa))
```
