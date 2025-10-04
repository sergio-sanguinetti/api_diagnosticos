#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación Flask mínima para pruebas de despliegue
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Crear aplicación Flask
app = Flask(__name__)
CORS(app)

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
    """Endpoint básico para análisis."""
    form_data = request.get_json()
    if not form_data:
        return jsonify({"error": "No se recibieron datos en formato JSON"}), 400
    
    return jsonify({"diagnostico_completo": "Análisis básico funcionando"})

@app.route('/generar-reporte-comparativo', methods=['POST'])
def generar_reporte_endpoint():
    """Endpoint básico para reportes."""
    data = request.get_json()
    if not data or 'token' not in data:
        return jsonify({"error": "Petición inválida. Se requiere un 'token'."}), 400
    
    return jsonify({"error": "Funcionalidad completa no disponible en modo de prueba"}), 501

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
