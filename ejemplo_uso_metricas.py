#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de Uso del Generador de Métricas
=========================================

Este script demuestra cómo usar el GeneradorMetricas para calcular
las métricas de similitud entre diagnósticos y recomendaciones médicas.

Autor: Sistema de Análisis Médico Ocupacional
Fecha: 2024
"""

from generador_metricas import GeneradorMetricas
import json


def ejemplo_basico():
    """Ejemplo básico de uso del generador de métricas."""
    print("🔬 EJEMPLO BÁSICO DE MÉTRICAS MÉDICAS")
    print("=" * 50)
    
    # Crear instancia del generador
    generador = GeneradorMetricas()
    
    # Ejemplo 1: Comparar dos diagnósticos similares
    diagnostico1 = "Obesidad mórbida con IMC elevado"
    diagnostico2 = "Obesidad mórbida, índice de masa corporal alto"
    
    print(f"\n📋 Diagnóstico 1: {diagnostico1}")
    print(f"📋 Diagnóstico 2: {diagnostico2}")
    
    # Calcular métricas
    jaccard = generador.calcular_similitud_jaccard(diagnostico1, diagnostico2)
    cosenos = generador.calcular_similitud_cosenos(diagnostico1, diagnostico2)
    
    print(f"\n📊 Resultados:")
    print(f"   Similitud Jaccard: {jaccard:.3f}")
    print(f"   Similitud Cosenos: {cosenos:.3f}")
    
    # Ejemplo 2: Comparar diagnósticos diferentes
    diagnostico3 = "Hipotiroidismo no especificado"
    diagnostico4 = "Diabetes mellitus tipo 2"
    
    print(f"\n📋 Diagnóstico 3: {diagnostico3}")
    print(f"📋 Diagnóstico 4: {diagnostico4}")
    
    jaccard2 = generador.calcular_similitud_jaccard(diagnostico3, diagnostico4)
    cosenos2 = generador.calcular_similitud_cosenos(diagnostico3, diagnostico4)
    
    print(f"\n📊 Resultados:")
    print(f"   Similitud Jaccard: {jaccard2:.3f}")
    print(f"   Similitud Cosenos: {cosenos2:.3f}")


def ejemplo_completo():
    """Ejemplo completo con todos los datos de la tabla."""
    print("\n🔬 EJEMPLO COMPLETO CON TABLA COMPARATIVA")
    print("=" * 50)
    
    # Crear instancia del generador
    generador = GeneradorMetricas()
    
    # Generar todas las métricas
    resultados = generador.generar_metricas_completas()
    
    # Imprimir resultados
    generador.imprimir_resultados(resultados)
    
    return resultados


def ejemplo_personalizado():
    """Ejemplo con datos personalizados."""
    print("\n🔬 EJEMPLO CON DATOS PERSONALIZADOS")
    print("=" * 50)
    
    generador = GeneradorMetricas()
    
    # Datos personalizados
    casos_personalizados = [
        {
            'medico': "Hipertensión arterial esencial",
            'deepseek': "Presión arterial elevada",
            'gemini': "HTA, tensión arterial alta"
        },
        {
            'medico': "Diabetes mellitus tipo 2",
            'deepseek': "Diabetes tipo 2",
            'gemini': "DM2, glucosa elevada"
        },
        {
            'medico': "Sin diagnóstico",
            'deepseek': "Sin diagnóstico",
            'gemini': "Sin diagnóstico"
        }
    ]
    
    print("\n📊 Comparaciones personalizadas:")
    for i, caso in enumerate(casos_personalizados, 1):
        print(f"\n--- Caso {i} ---")
        print(f"Médico: {caso['medico']}")
        print(f"DeepSeek: {caso['deepseek']}")
        print(f"Gemini: {caso['gemini']}")
        
        # Calcular métricas entre pares
        jaccard_md = generador.calcular_similitud_jaccard(caso['medico'], caso['deepseek'])
        jaccard_mg = generador.calcular_similitud_jaccard(caso['medico'], caso['gemini'])
        jaccard_dg = generador.calcular_similitud_jaccard(caso['deepseek'], caso['gemini'])
        
        cosenos_md = generador.calcular_similitud_cosenos(caso['medico'], caso['deepseek'])
        cosenos_mg = generador.calcular_similitud_cosenos(caso['medico'], caso['gemini'])
        cosenos_dg = generador.calcular_similitud_cosenos(caso['deepseek'], caso['gemini'])
        
        print(f"Jaccard Médico-DeepSeek: {jaccard_md:.3f}")
        print(f"Jaccard Médico-Gemini: {jaccard_mg:.3f}")
        print(f"Jaccard DeepSeek-Gemini: {jaccard_dg:.3f}")
        print(f"Cosenos Médico-DeepSeek: {cosenos_md:.3f}")
        print(f"Cosenos Médico-Gemini: {cosenos_mg:.3f}")
        print(f"Cosenos DeepSeek-Gemini: {cosenos_dg:.3f}")


def guardar_resultados_json(resultados, archivo="resultados_metricas.json"):
    """Guarda los resultados en un archivo JSON."""
    print(f"\n💾 Guardando resultados en {archivo}...")
    
    # Convertir numpy arrays a listas para JSON
    def convertir_para_json(obj):
        if isinstance(obj, dict):
            return {k: convertir_para_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convertir_para_json(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj
    
    resultados_json = convertir_para_json(resultados)
    
    with open(archivo, 'w', encoding='utf-8') as f:
        json.dump(resultados_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Resultados guardados exitosamente en {archivo}")


def main():
    """Función principal que ejecuta todos los ejemplos."""
    print("🚀 EJEMPLOS DE USO DEL GENERADOR DE MÉTRICAS")
    print("=" * 60)
    
    try:
        # Ejecutar ejemplos
        ejemplo_basico()
        resultados = ejemplo_completo()
        ejemplo_personalizado()
        
        # Guardar resultados
        guardar_resultados_json(resultados)
        
        print("\n🎉 Todos los ejemplos ejecutados exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
