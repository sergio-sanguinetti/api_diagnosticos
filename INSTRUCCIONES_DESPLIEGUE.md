# Instrucciones para el Despliegue en Render

## 🚨 Problema Identificado

El error `ModuleNotFoundError: No module named 'app'` se debe a que Render no puede encontrar el módulo de la aplicación Flask.

## ✅ Solución Implementada

### 1. **Archivos Creados para Solucionar el Problema**

- **`app.py`**: Archivo de entrada que importa la aplicación desde `analizador_ia.py`
- **`app_minimal.py`**: Versión mínima de la aplicación para pruebas de despliegue
- **`Procfile`**: Archivo de configuración estándar para despliegues
- **`.python-version`**: Especifica la versión de Python

### 2. **Configuración Actual**

El archivo `render.yaml` está configurado para usar `app_minimal:app` temporalmente.

## 🔧 Pasos para Restaurar la Funcionalidad Completa

### Paso 1: Verificar que el Despliegue Básico Funcione

1. Hacer commit y push de los cambios actuales
2. Verificar que Render despliegue correctamente
3. Probar el endpoint `/` para confirmar que la aplicación responde

### Paso 2: Instalar Dependencias Faltantes

Una vez que el despliegue básico funcione, necesitamos instalar `google-generativeai`:

```bash
# En el entorno de Render, agregar a requirements.txt:
google-generativeai
```

### Paso 3: Restaurar la Funcionalidad Completa

1. **Cambiar el comando de inicio en `render.yaml`**:
   ```yaml
   startCommand: gunicorn --workers 2 --bind 0.0.0.0:$PORT --timeout 60 --keep-alive 2 --max-requests 100 --max-requests-jitter 10 app:app
   ```

2. **Verificar que todas las dependencias estén en `requirements.txt`**:
   ```
   Flask
   Flask-Cors
   fpdf2
   google-generativeai
   gunicorn
   mysql-connector-python
   requests
   numpy
   scikit-learn
   ```

### Paso 4: Configurar Variables de Entorno

En el dashboard de Render, configurar las siguientes variables de entorno:

- `GOOGLE_API_KEY`: Clave de API de Google Gemini
- `DEEPSEEK_API_KEY`: Clave de API de DeepSeek
- `HUGGINGFACE_API_KEY`: Clave de API de Hugging Face
- `DB_HOST`: Host de la base de datos MySQL
- `DB_USER`: Usuario de la base de datos
- `DB_PASS`: Contraseña de la base de datos
- `DB_NAME`: Nombre de la base de datos
- `ENABLE_SEMANTIC_SIMILARITY`: "true"

## 📁 Archivos del Proyecto

### Archivos Principales
- `analizador_ia.py`: Aplicación Flask principal con toda la funcionalidad
- `motor_analisis.py`: Lógica de análisis médico y generación de reportes
- `generador_metricas.py`: Sistema de métricas de similitud médica

### Archivos de Configuración
- `render.yaml`: Configuración de Render
- `requirements.txt`: Dependencias de Python
- `Procfile`: Comando de inicio para Gunicorn
- `.python-version`: Versión de Python

### Archivos de Entrada
- `app.py`: Archivo de entrada principal (importa desde analizador_ia.py)
- `app_minimal.py`: Versión mínima para pruebas

## 🚀 Estado Actual

- ✅ **Despliegue básico**: Configurado con `app_minimal.py`
- ⏳ **Funcionalidad completa**: Pendiente de restaurar después del despliegue básico
- ✅ **Dependencias básicas**: Instaladas y funcionando
- ⏳ **Google Generative AI**: Pendiente de instalación

## 🔍 Próximos Pasos

1. **Hacer commit y push** de los cambios actuales
2. **Verificar el despliegue** en Render
3. **Probar el endpoint básico** `/`
4. **Restaurar la funcionalidad completa** siguiendo los pasos anteriores
5. **Configurar variables de entorno** en Render
6. **Probar todos los endpoints** de la aplicación completa

## 📞 Soporte

Si hay problemas durante el proceso de restauración:

1. Verificar los logs de Render para errores específicos
2. Confirmar que todas las dependencias estén instaladas
3. Verificar que las variables de entorno estén configuradas
4. Probar localmente antes de hacer push a producción
