# API de Diagnósticos Médicos Ocupacionales

## 📋 Descripción General

Esta API proporciona análisis automatizado de diagnósticos médicos ocupacionales utilizando inteligencia artificial. Combina el análisis de múltiples modelos de IA (Google Gemini y DeepSeek) para generar informes médicos comparativos y recomendaciones clínicas.

## 🏗️ Arquitectura del Sistema

### Componentes Principales

- **API Flask**: Servidor web que maneja las peticiones HTTP
- **Motor de Análisis**: Módulo que procesa datos médicos y genera reportes
- **Integración con IAs**: Conexión con Google Gemini y DeepSeek APIs
- **Base de Datos MySQL**: Almacenamiento de resultados médicos
- **Generador de PDFs**: Creación de informes en formato PDF

### Tecnologías Utilizadas

- **Backend**: Python 3.x, Flask
- **Base de Datos**: MySQL
- **APIs de IA**: Google Gemini, DeepSeek
- **Procesamiento**: NumPy, FPDF2
- **Despliegue**: Render (PaaS)

## 🚀 Implementación en Render

### 1. Preparación del Proyecto

#### Estructura de Archivos Requerida
```
api_diagnosticos/
├── analizador_ia.py          # Archivo principal de la API
├── motor_analisis.py         # Lógica de análisis médico
├── requirements.txt           # Dependencias de Python
├── DejaVuSans.ttf           # Fuente para PDFs
├── DejaVuSans-Bold.ttf      # Fuente en negrita para PDFs
└── diagramas/                # Diagramas de arquitectura
```

#### Dependencias (requirements.txt)
```
Flask
Flask-Cors
fpdf2
google-generativeai
gunicorn
mysql-connector-python
requests
numpy
```

### 2. Configuración en Render

#### Variables de Entorno Requeridas
```bash
GOOGLE_API_KEY=tu_clave_api_google
DEEPSEEK_API_KEY=tu_clave_api_deepseek
HUGGINGFACE_API_KEY=tu_clave_api_huggingface
DB_HOST=tu_host_mysql
DB_USER=tu_usuario_mysql
DB_PASS=tu_password_mysql
DB_NAME=tu_base_datos_mysql
```

#### Configuración del Servicio
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn analizador_ia:app`
- **Plan**: Free o superior según necesidades

### 3. Pasos de Despliegue

1. **Conectar Repositorio Git**
   - Conectar tu repositorio de GitHub a Render
   - Configurar auto-deploy en cambios

2. **Configurar Variables de Entorno**
   - Ir a Environment Variables en tu servicio
   - Agregar todas las variables requeridas

3. **Desplegar**
   - Render detectará automáticamente el código Python
   - Construirá las dependencias
   - Iniciará el servicio

4. **Verificar Despliegue**
   - Revisar logs de construcción
   - Probar endpoints de la API

## 📡 Endpoints de la API

### 1. Análisis de Formulario (`POST /analizar`)

Analiza datos médicos enviados directamente desde un formulario.

**Request Body:**
```json
{
  "centro_medico": "Centro Médico Ejemplo",
  "ciudad": "Ciudad Ejemplo",
  "fecha_examen": "2024-01-15",
  "puesto": "Operador",
  "tipo_examen": "Pre-ocupacional",
  "aptitud": "Apto",
  "valor_glucosa": "95",
  "resultado_glucosa": "normal",
  "valor_colesterol_total": "200",
  "resultado_colesterol_total": "normal"
}
```

**Response:**
```json
{
  "diagnostico_completo": "Análisis médico generado por IA..."
}
```

### 2. Reporte Comparativo (`POST /generar-reporte-comparativo`)

Genera un reporte PDF comparativo usando múltiples modelos de IA.

**Request Body:**
```json
{
  "token": "token_del_paciente"
}
```

**Response:**
- Archivo PDF para descargar
- Contiene análisis comparativo de DeepSeek y Gemini

## 🔧 Configuración Local

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Variables de Entorno Locales
Crear archivo `.env`:
```env
GOOGLE_API_KEY=tu_clave_api_google
DEEPSEEK_API_KEY=tu_clave_api_deepseek
HUGGINGFACE_API_KEY=tu_clave_api_huggingface
DB_HOST=localhost
DB_USER=usuario
DB_PASS=password
DB_NAME=base_datos
```

### Ejecución Local
```bash
python analizador_ia.py
```

## 📊 Funcionalidades del Sistema

### Análisis Médico
- **Evaluación Automática**: Análisis de resultados de laboratorio
- **Diagnósticos Diferenciales**: Sugerencias de diagnósticos basadas en hallazgos
- **Recomendaciones Clínicas**: Sugerencias de tratamiento y seguimiento

### Comparación de IAs
- **Análisis Dual**: Comparación entre DeepSeek y Gemini
- **Resumen Ejecutivo**: Síntesis unificada de ambos análisis
- **Métricas de Similitud**: Evaluación cuantitativa de concordancia

### Generación de Reportes
- **PDFs Profesionales**: Informes estructurados y formateados
- **Múltiples Secciones**: Datos del paciente, hallazgos, análisis y comparaciones
- **Tabla Comparativa de Diagnósticos**: Comparación horizontal de diagnósticos encontrados por médico, DeepSeek y Gemini
- **Fuentes Personalizadas**: Soporte para caracteres especiales

## 🔒 Seguridad y Consideraciones

### Variables de Entorno
- **Nunca** incluir claves API en el código
- Usar variables de entorno en producción
- Rotar claves API regularmente

### Base de Datos
- Usar conexiones seguras (SSL/TLS)
- Implementar autenticación robusta
- Validar todas las entradas de usuario

### APIs Externas
- Implementar rate limiting
- Manejar errores de API gracefully
- Logging de todas las operaciones

## 📈 Monitoreo y Logs

### Logs de Render
- Revisar logs de construcción para errores
- Monitorear logs de aplicación en tiempo real
- Configurar alertas para errores críticos

### Métricas de Rendimiento
- Tiempo de respuesta de endpoints
- Uso de memoria y CPU
- Errores de API externas

## 🚨 Troubleshooting Común

### Error de Conexión a Base de Datos
- Verificar credenciales en variables de entorno
- Confirmar que la base de datos esté accesible
- Revisar configuración de firewall

### Error de API de IA
- Verificar claves API válidas
- Confirmar límites de uso de API
- Revisar logs de error específicos

### Error de Generación de PDF
- Verificar que las fuentes estén presentes
- Confirmar permisos de escritura
- Revisar formato de datos de entrada

## 🔄 Mantenimiento

### Actualizaciones Regulares
- Mantener dependencias actualizadas
- Revisar cambios en APIs externas
- Actualizar documentación según cambios

### Backup y Recuperación
- Backup regular de base de datos
- Versionado de código en Git
- Documentación de procedimientos de rollback

## 📞 Soporte

Para soporte técnico o preguntas sobre la implementación:
- Revisar logs de Render
- Verificar configuración de variables de entorno
- Consultar documentación de APIs externas

## 📝 Notas de Versión

### v3.1 (Actual)
- Mejoras en diseño de PDF
- Agrupación de diagnósticos por tipo de examen
- Métricas de similitud semántica
- Comparación robusta de análisis de IA

### v3.0
- Integración de múltiples modelos de IA
- Generación de reportes comparativos
- Sistema de métricas de calidad

### v2.0
- Análisis básico con Google Gemini
- Generación de PDFs
- Integración con base de datos MySQL

---

**Desarrollado para análisis médico ocupacional asistido por inteligencia artificial**

