# Diagramas PlantUML - API de Diagnósticos Médicos

Esta carpeta contiene todos los diagramas PlantUML que documentan la arquitectura, flujos y componentes del sistema de análisis médico ocupacional.

## 📊 Diagramas Disponibles

### 1. **01_arquitectura_general.puml**
- **Descripción**: Vista general de la arquitectura del sistema completo
- **Contenido**: Componentes principales, APIs externas, base de datos y flujo de datos
- **Uso**: Entender la estructura general del sistema

### 2. **02_flujo_datos.puml**
- **Descripción**: Flujo de datos desde la entrada hasta la generación del PDF
- **Contenido**: Proceso completo de análisis médico con ambos endpoints
- **Uso**: Comprender el flujo de trabajo del sistema

### 3. **03_secuencia_analisis.puml**
- **Descripción**: Diagrama de secuencia del endpoint principal
- **Contenido**: Interacción detallada entre componentes durante el análisis
- **Uso**: Entender la secuencia de operaciones del sistema

### 4. **04_diagrama_clases.puml**
- **Descripción**: Estructura de clases y métodos del sistema
- **Contenido**: Clases principales, atributos, métodos y relaciones
- **Uso**: Entender la estructura del código y las dependencias

### 5. **05_despliegue_render.puml**
- **Descripción**: Proceso de despliegue en Render PaaS
- **Contenido**: Flujo desde GitHub hasta el servicio activo
- **Uso**: Guía para implementar el sistema en Render

### 6. **06_estructura_bd.puml**
- **Descripción**: Estructura de la base de datos MySQL
- **Contenido**: Tabla principal, campos, tipos de datos y relaciones
- **Uso**: Entender el esquema de datos del sistema

### 7. **07_componentes_sistema.puml**
- **Descripción**: Componentes del sistema y sus interacciones
- **Contenido**: Módulos, interfaces y dependencias entre componentes
- **Uso**: Entender la arquitectura modular del sistema

### 8. **08_casos_uso.puml**
- **Descripción**: Casos de uso del sistema
- **Contenido**: Actores, funcionalidades y relaciones de uso
- **Uso**: Entender las funcionalidades disponibles y usuarios del sistema

## 🛠️ Cómo Visualizar los Diagramas

### Opción 1: PlantUML Online
1. Ve a [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)
2. Copia y pega el contenido de cualquier archivo `.puml`
3. El diagrama se generará automáticamente

### Opción 2: Extensión de VS Code
1. Instala la extensión "PlantUML" en VS Code
2. Abre cualquier archivo `.puml`
3. Presiona `Alt+Shift+D` para previsualizar

### Opción 3: Plugin de IntelliJ/WebStorm
1. Instala el plugin "PlantUML integration"
2. Abre cualquier archivo `.puml`
3. El diagrama se mostrará automáticamente

### Opción 4: Herramienta de Línea de Comandos
```bash
# Instalar PlantUML
npm install -g @plantuml/plantuml

# Generar PNG desde archivo .puml
plantuml diagrama.puml

# Generar SVG
plantuml -tsvg diagrama.puml
```

## 🎨 Personalización de Diagramas

Los diagramas utilizan un tema personalizado con:
- **Colores**: Azul profesional (#2E86AB, #E8F4FD)
- **Fondo**: Blanco (#FFFFFF)
- **Estilo**: Limpio y profesional
- **Fuentes**: Arial/Helvetica

Para modificar el estilo, edita los parámetros `skinparam` en cada archivo.

## 📝 Notas de Mantenimiento

- **Actualización**: Los diagramas deben actualizarse cuando cambie la arquitectura del sistema
- **Consistencia**: Mantener el mismo estilo visual en todos los diagramas
- **Documentación**: Cada diagrama debe tener notas explicativas claras
- **Versionado**: Los diagramas están versionados junto con el código

## 🔗 Relación con el Código

Los diagramas están basados en:
- `analizador_ia.py` - API principal y endpoints
- `motor_analisis.py` - Lógica de análisis y generación de reportes
- `requirements.txt` - Dependencias del sistema
- Estructura de base de datos MySQL

## 📞 Soporte

Para preguntas sobre los diagramas o sugerencias de mejora:
- Revisar la documentación principal del proyecto
- Verificar que los diagramas coincidan con el código actual
- Actualizar diagramas cuando se modifique la arquitectura

---

**Nota**: Estos diagramas son herramientas de documentación y deben mantenerse sincronizados con el código del sistema.

