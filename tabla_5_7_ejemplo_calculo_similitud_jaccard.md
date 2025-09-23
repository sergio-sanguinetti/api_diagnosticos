# Tabla 5.7: Ejemplo Paso a Paso del Cálculo de Similitud Jaccard para dos Diagnósticos

## Resumen
Esta tabla presenta un ejemplo detallado del cálculo de similitud Jaccard entre dos diagnósticos médicos, mostrando cada paso del proceso implementado en el sistema de análisis médico ocupacional asistido por IA.

## Fórmula de Similitud Jaccard

La similitud de Jaccard se calcula usando la siguiente fórmula:

```
J(A,B) = |A ∩ B| / |A ∪ B|
```

Donde:
- **A** = Conjunto de términos médicos del primer diagnóstico
- **B** = Conjunto de términos médicos del segundo diagnóstico
- **A ∩ B** = Intersección (términos comunes)
- **A ∪ B** = Unión (todos los términos únicos)
- **J(A,B)** = Similitud Jaccard (rango: 0.0 - 1.0)

## Ejemplo Práctico: Comparación de Diagnósticos

### Caso de Estudio
**Diagnóstico 1 (Sistema Médico)**: "Hipertrigliceridemia con colesterol elevado, requiere dieta baja en grasas y control de lípidos"

**Diagnóstico 2 (IA DeepSeek)**: "Dislipidemia mixta con triglicéridos altos, recomendación de alimentación saludable y seguimiento cardiológico"

## Tabla Paso a Paso del Cálculo

| **Paso** | **Descripción** | **Operación** | **Resultado** |
|----------|-----------------|---------------|---------------|
| **1** | **Extracción de términos médicos del Diagnóstico 1** | `extract_medical_terms(diagnostico1)` | `['hipertrigliceridemia', 'colesterol', 'dieta', 'lípidos']` |
| **2** | **Extracción de términos médicos del Diagnóstico 2** | `extract_medical_terms(diagnostico2)` | `['dislipidemia', 'triglicéridos', 'alimentación', 'cardiológico']` |
| **3** | **Creación de conjuntos** | `A = set(terminos1)`, `B = set(terminos2)` | `A = {'hipertrigliceridemia', 'colesterol', 'dieta', 'lípidos'}`<br>`B = {'dislipidemia', 'triglicéridos', 'alimentación', 'cardiológico'}` |
| **4** | **Cálculo de intersección** | `A ∩ B` | `{}` (conjunto vacío) |
| **5** | **Cálculo de unión** | `A ∪ B` | `{'hipertrigliceridemia', 'colesterol', 'dieta', 'lípidos', 'dislipidemia', 'triglicéridos', 'alimentación', 'cardiológico'}` |
| **6** | **Conteo de intersección** | `|A ∩ B|` | `0` |
| **7** | **Conteo de unión** | `|A ∪ B|` | `8` |
| **8** | **Cálculo de Jaccard** | `0 / 8` | `0.0` |

## Análisis Detallado del Proceso

### Paso 1: Extracción de Términos del Diagnóstico 1
```python
texto1 = "Hipertrigliceridemia con colesterol elevado, requiere dieta baja en grasas y control de lípidos"
terminos1 = extract_medical_terms(texto1)
# Resultado: ['hipertrigliceridemia', 'colesterol', 'dieta', 'lípidos']
```

**Términos encontrados:**
- ✅ `hipertrigliceridemia` - Encontrado en lista de términos médicos
- ✅ `colesterol` - Encontrado en lista de términos médicos  
- ✅ `dieta` - Encontrado en lista de términos médicos
- ❌ `elevado` - No está en la lista de términos médicos
- ❌ `grasas` - No está en la lista de términos médicos
- ❌ `control` - No está en la lista de términos médicos
- ✅ `lípidos` - Encontrado en lista de términos médicos

### Paso 2: Extracción de Términos del Diagnóstico 2
```python
texto2 = "Dislipidemia mixta con triglicéridos altos, recomendación de alimentación saludable y seguimiento cardiológico"
terminos2 = extract_medical_terms(texto2)
# Resultado: ['dislipidemia', 'triglicéridos', 'alimentación', 'cardiológico']
```

**Términos encontrados:**
- ✅ `dislipidemia` - Encontrado en lista de términos médicos
- ❌ `mixta` - No está en la lista de términos médicos
- ✅ `triglicéridos` - Encontrado en lista de términos médicos
- ❌ `altos` - No está en la lista de términos médicos
- ❌ `recomendación` - No está en la lista de términos médicos
- ✅ `alimentación` - Encontrado en lista de términos médicos
- ❌ `saludable` - No está en la lista de términos médicos
- ❌ `seguimiento` - No está en la lista de términos médicos
- ✅ `cardiológico` - Encontrado en lista de términos médicos

### Paso 3: Creación de Conjuntos
```python
A = set(['hipertrigliceridemia', 'colesterol', 'dieta', 'lípidos'])
B = set(['dislipidemia', 'triglicéridos', 'alimentación', 'cardiológico'])
```

### Paso 4: Cálculo de Intersección
```python
interseccion = A & B
# Resultado: set() (conjunto vacío)
```

**Análisis de intersección:**
- `hipertrigliceridemia` ∉ B
- `colesterol` ∉ B
- `dieta` ∉ B
- `lípidos` ∉ B

**No hay términos comunes entre ambos diagnósticos.**

### Paso 5: Cálculo de Unión
```python
union = A | B
# Resultado: {'hipertrigliceridemia', 'colesterol', 'dieta', 'lípidos', 'dislipidemia', 'triglicéridos', 'alimentación', 'cardiológico'}
```

**Todos los términos únicos:**
- Términos de A: `hipertrigliceridemia`, `colesterol`, `dieta`, `lípidos`
- Términos de B: `dislipidemia`, `triglicéridos`, `alimentación`, `cardiológico`
- Total: 8 términos únicos

### Paso 6: Aplicación de la Fórmula
```python
jaccard = len(interseccion) / len(union)
jaccard = 0 / 8
jaccard = 0.0
```

## Interpretación del Resultado

### Similitud Jaccard = 0.0

| **Rango** | **Interpretación** | **Significado** |
|-----------|-------------------|-----------------|
| **0.0** | Sin similitud | Los diagnósticos no comparten términos médicos comunes |
| **0.0 - 0.3** | Baja similitud | Pocos términos en común |
| **0.3 - 0.7** | Similitud moderada | Algunos términos compartidos |
| **0.7 - 1.0** | Alta similitud | Muchos términos en común |
| **1.0** | Similitud perfecta | Todos los términos son idénticos |

### Análisis del Caso
Aunque ambos diagnósticos se refieren a **trastornos del metabolismo de lípidos**, utilizan terminología diferente:

- **Diagnóstico 1** usa: `hipertrigliceridemia`, `colesterol`, `dieta`, `lípidos`
- **Diagnóstico 2** usa: `dislipidemia`, `triglicéridos`, `alimentación`, `cardiológico`

**Problema identificado**: La función `extract_medical_terms()` no reconoce sinónimos médicos como:
- `hipertrigliceridemia` ≈ `triglicéridos`
- `colesterol` ≈ `lípidos`
- `dieta` ≈ `alimentación`

## Ejemplo con Mayor Similitud

### Caso Mejorado
**Diagnóstico 1**: "Hipertensión arterial con presión elevada, requiere dieta baja en sodio"

**Diagnóstico 2**: "Presión arterial alta, recomendación de alimentación sin sal"

| **Paso** | **Operación** | **Resultado** |
|----------|---------------|---------------|
| **1** | Términos A | `['hipertensión', 'presión arterial', 'dieta']` |
| **2** | Términos B | `['presión arterial', 'alimentación']` |
| **3** | Intersección | `{'presión arterial'}` |
| **4** | Unión | `{'hipertensión', 'presión arterial', 'dieta', 'alimentación'}` |
| **5** | Jaccard | `1 / 4 = 0.25` |

## Implementación en el Sistema

### Código de la Función
```python
def calculate_jaccard_similarity(text_medico, text_ia):
    """Calcula la Similitud de Jaccard entre conjuntos de términos médicos."""
    try:
        # Extraer términos médicos de ambos textos
        medico_terms = set(extract_medical_terms(text_medico))
        ia_terms = set(extract_medical_terms(text_ia))
        
        # Casos especiales
        if len(medico_terms) == 0 and len(ia_terms) == 0:
            return 1.0  # Ambos vacíos = perfecta similitud
        
        if len(medico_terms) == 0 or len(ia_terms) == 0:
            return 0.0  # Uno vacío, otro no = sin similitud
        
        # Calcular intersección y unión
        intersection = medico_terms & ia_terms
        union = medico_terms | ia_terms
        
        # Calcular Jaccard
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        
        return jaccard
        
    except Exception as e:
        print(f"❌ Error calculando Jaccard: {e}")
        return 0.0
```

### Casos Especiales Manejados

| **Caso** | **Condición** | **Resultado** | **Justificación** |
|----------|---------------|---------------|-------------------|
| **Ambos vacíos** | `len(A) == 0 and len(B) == 0` | `1.0` | Sin términos = perfecta similitud |
| **Uno vacío** | `len(A) == 0 or len(B) == 0` | `0.0` | No hay términos comunes |
| **División por cero** | `len(union) == 0` | `0.0` | Evita error matemático |
| **Error de extracción** | `Exception` | `0.0` | Manejo robusto de errores |

## Limitaciones del Método Actual

### 1. **Falta de Reconocimiento de Sinónimos**
- `hipertrigliceridemia` ≠ `triglicéridos` (aunque son relacionados)
- `dieta` ≠ `alimentación` (aunque son sinónimos)
- `colesterol` ≠ `lípidos` (aunque son relacionados)

### 2. **Lista Limitada de Términos**
- Solo 50+ términos médicos predefinidos
- No incluye variaciones morfológicas
- No considera contexto médico

### 3. **Búsqueda Literal**
- No reconoce abreviaciones (IMC vs Índice de Masa Corporal)
- No maneja plurales (triglicérido vs triglicéridos)
- No considera acentos o variaciones ortográficas

## Mejoras Propuestas

### 1. **Expansión del Vocabulario Médico**
```python
# Agregar sinónimos y variaciones
medical_terms = [
    # Términos actuales
    'hipertensión', 'presión arterial', 'tensión',
    'diabetes', 'glucosa', 'glicemia',
    'dislipidemia', 'colesterol', 'triglicéridos',
    
    # Sinónimos y variaciones
    'hipertrigliceridemia', 'triglicéridos altos',
    'dieta', 'alimentación', 'nutrición',
    'lípidos', 'grasas', 'colesterol total'
]
```

### 2. **Normalización de Términos**
```python
def normalize_medical_term(term):
    """Normaliza términos médicos para mejor comparación."""
    # Convertir a minúsculas
    term = term.lower()
    
    # Mapear sinónimos
    synonyms = {
        'dieta': 'alimentación',
        'triglicéridos': 'hipertrigliceridemia',
        'presión arterial': 'hipertensión'
    }
    
    return synonyms.get(term, term)
```

### 3. **Uso de Embeddings Semánticos**
```python
def calculate_semantic_jaccard(text1, text2):
    """Calcula Jaccard usando embeddings semánticos."""
    # Generar embeddings de términos
    terms1_embeddings = get_term_embeddings(extract_medical_terms(text1))
    terms2_embeddings = get_term_embeddings(extract_medical_terms(text2))
    
    # Calcular similitud semántica entre términos
    # Usar umbral de similitud para determinar "términos similares"
    # Aplicar fórmula de Jaccard modificada
```

## Aplicación en el Sistema Médico

### Uso en Métricas de Concordancia
La similitud Jaccard se utiliza junto con otras métricas:

1. **Similitud Semántica** (DeepSeek API): Análisis contextual profundo
2. **Índice de Kappa Cohen**: Concordancia entre evaluadores
3. **Similitud Jaccard**: Comparación de conjuntos de términos

### Interpretación en Contexto Clínico
- **Jaccard > 0.7**: Diagnósticos muy similares en terminología
- **Jaccard 0.3-0.7**: Diagnósticos moderadamente similares
- **Jaccard < 0.3**: Diagnósticos diferentes en terminología
- **Jaccard = 0.0**: Sin términos médicos comunes

## Referencias Técnicas

- **Jaccard, P. (1912)**: "The distribution of the flora in the alpine zone"
- **Similitud de Conjuntos**: Medida de similitud entre conjuntos finitos
- **Aplicación en Medicina**: Comparación de diagnósticos y terminología médica
- **Implementación**: Algoritmo eficiente para grandes conjuntos de datos
