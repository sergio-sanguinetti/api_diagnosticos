# Generador de Métricas de Similitud Médica

Este módulo implementa un sistema completo para calcular métricas de similitud entre diagnósticos y recomendaciones médicas, utilizando únicamente los datos extraídos de la tabla comparativa.

## 🎯 Métricas Implementadas

### 1. Similitud de Jaccard
- **Fórmula**: `J(A,B) = |A ∩ B| / |A ∪ B|`
- **Rango**: 0.0 - 1.0
- **Uso**: Compara conjuntos de términos médicos entre diagnósticos
- **Interpretación**:
  - 0.0: Sin similitud (ningún término común)
  - 0.3-0.7: Similitud moderada
  - 0.7-1.0: Alta similitud
  - 1.0: Similitud perfecta

### 2. Similitud de Cosenos
- **Método**: TF-IDF + Coseno del ángulo entre vectores
- **Rango**: 0.0 - 1.0
- **Uso**: Compara la similitud semántica entre textos médicos
- **Ventaja**: Considera la frecuencia de términos y contexto

### 3. Índice de Kappa Cohen
- **Fórmula**: `κ = (Po - Pe) / (1 - Pe)`
- **Rango**: -1.0 a 1.0
- **Uso**: Evalúa concordancia entre evaluadores (médico vs IA)
- **Interpretación**:
  - < 0.2: Concordancia pobre
  - 0.2-0.4: Concordancia justa
  - 0.4-0.6: Concordancia moderada
  - 0.6-0.8: Concordancia sustancial
  - > 0.8: Concordancia casi perfecta

## 📊 Datos de Entrada

El sistema utiliza los datos de la tabla comparativa con las siguientes fuentes:

- **Médico/Sistema**: Diagnósticos y recomendaciones del sistema médico tradicional
- **DeepSeek**: Diagnósticos y recomendaciones del modelo DeepSeek
- **Gemini**: Diagnósticos y recomendaciones del modelo Gemini

### Casos Analizados:
1. Obesidad Mórbida
2. Ametropía Corregida
3. Linfopenia
4. Hipotiroidismo No Especificado
5. Prediabetes/Glucosa Elevada
6. Hipotiroidismo No Especificado (segunda instancia)

## 🚀 Instalación y Uso

### Requisitos
```bash
pip install -r requirements.txt
```

### Uso Básico
```python
from generador_metricas import GeneradorMetricas

# Crear instancia
generador = GeneradorMetricas()

# Calcular métricas individuales
jaccard = generador.calcular_similitud_jaccard(texto1, texto2)
cosenos = generador.calcular_similitud_cosenos(texto1, texto2)

# Generar todas las métricas
resultados = generador.generar_metricas_completas()
generador.imprimir_resultados(resultados)
```

### Ejecutar Ejemplos
```bash
python ejemplo_uso_metricas.py
```

## 📈 Estructura de Resultados

### Similitud Jaccard
```python
{
    'Obesidad Mórbida': {
        'medico_vs_deepseek_diag': 0.750,
        'medico_vs_gemini_diag': 0.750,
        'deepseek_vs_gemini_diag': 1.000,
        'medico_vs_deepseek_rec': 0.333,
        'medico_vs_gemini_rec': 0.200,
        'deepseek_vs_gemini_rec': 0.250
    },
    # ... más casos
}
```

### Similitud de Cosenos
```python
{
    'Obesidad Mórbida': {
        'medico_vs_deepseek_diag': 0.856,
        'medico_vs_gemini_diag': 0.789,
        'deepseek_vs_gemini_diag': 0.923,
        # ... más comparaciones
    },
    # ... más casos
}
```

### Índice Kappa Cohen
```python
{
    'medico_vs_deepseek': 0.456,
    'medico_vs_gemini': 0.234,
    'deepseek_vs_gemini': 0.678
}
```

## 🔧 Funcionalidades Principales

### Extracción de Términos Médicos
- Identifica automáticamente términos médicos relevantes
- Normaliza texto para mejor comparación
- Maneja casos especiales ("Sin diagnóstico")

### Comparaciones Automáticas
- Médico vs DeepSeek
- Médico vs Gemini  
- DeepSeek vs Gemini
- Para diagnósticos y recomendaciones por separado

### Manejo de Casos Especiales
- Textos vacíos o "Sin diagnóstico"
- División por cero en cálculos
- Errores de procesamiento

## 📋 Términos Médicos Reconocidos

El sistema incluye un vocabulario médico extenso que incluye:

### Condiciones Médicas
- Obesidad, obesidad mórbida, IMC
- Ametropía, lentes correctores
- Linfopenia, leucocitos, linfocitos
- Hipotiroidismo, TSH, T4, T3
- Prediabetes, glucosa, diabetes
- Hipertensión, presión arterial
- Dislipidemia, colesterol, triglicéridos

### Especialidades
- Endocrinología, medicina interna
- Cardiología, oftalmología
- Nutrición, dietista

### Tratamientos
- Dieta, alimentación, ejercicio
- Medicamentos, seguimiento, control
- Derivación, consulta, evaluación

## 🎯 Interpretación de Resultados

### Similitud Alta (>0.7)
- Los diagnósticos/recomendaciones son muy similares
- Concordancia excelente entre fuentes
- Terminología médica consistente

### Similitud Moderada (0.3-0.7)
- Algunos elementos en común
- Diferencias en terminología o enfoque
- Requiere revisión manual

### Similitud Baja (<0.3)
- Diagnósticos/recomendaciones muy diferentes
- Posible discrepancia clínica
- Requiere evaluación detallada

## 🔍 Casos de Uso

1. **Evaluación de Sistemas de IA**: Comparar rendimiento de diferentes modelos
2. **Validación de Diagnósticos**: Verificar concordancia entre evaluadores
3. **Análisis de Calidad**: Identificar áreas de mejora en diagnósticos
4. **Investigación Médica**: Estudiar patrones en diagnósticos médicos

## 📊 Estadísticas Generadas

- Promedios de similitud por métrica
- Desviaciones estándar
- Comparaciones entre pares de sistemas
- Resumen estadístico completo

## 🛠️ Personalización

### Agregar Nuevos Términos Médicos
```python
generador.terminos_medicos.add('nuevo_termino_medico')
```

### Modificar Configuración TF-IDF
```python
generador.vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='spanish',
    ngram_range=(1, 3),  # Cambiar rango de n-gramas
    max_features=2000    # Cambiar número máximo de características
)
```

## 📝 Notas Técnicas

- **Dependencias**: numpy, scikit-learn
- **Codificación**: UTF-8 para caracteres especiales
- **Manejo de Errores**: Robusto con valores por defecto
- **Rendimiento**: Optimizado para conjuntos de datos médicos

## 🤝 Contribuciones

Para mejorar el sistema:
1. Expandir vocabulario médico
2. Agregar sinónimos médicos
3. Implementar normalización avanzada
4. Mejorar manejo de abreviaciones médicas

## 📄 Licencia

Este proyecto está bajo la misma licencia que el sistema principal de análisis médico ocupacional.
