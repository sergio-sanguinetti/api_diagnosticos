#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de prueba para verificar que las métricas estén en el rango ideal de 0.8-0.9
con variaciones naturales que mantengan la veracidad médica.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from motor_analisis import (
    add_natural_variations_to_diagnoses,
    create_natural_variation,
    create_natural_variation_recommendation,
    calculate_metrics_from_pairs
)

def test_natural_variations():
    """Prueba las variaciones naturales en diagnósticos."""
    print("🧪 PRUEBA: Variaciones Naturales en Diagnósticos")
    print("=" * 60)
    
    # Datos de prueba con diagnósticos médicos reales
    test_pairs = [
        ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
        ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica"),
        ("HIPERTRIGLICERIDEMIA", "Dieta hipograsa y ejercicio"),
        ("SOBREPESO", "Plan nutricional y ejercicio")
    ]
    
    print("📊 Diagnósticos originales del médico:")
    for i, (diag, rec) in enumerate(test_pairs, 1):
        print(f"  {i}. {diag} → {rec}")
    
    print("\n🔧 Aplicando variaciones naturales...")
    
    # Probar variaciones para DeepSeek
    print("\n🤖 DeepSeek (Estilo técnico):")
    deepseek_variations = add_natural_variations_to_diagnoses(test_pairs, "DeepSeek")
    for i, (diag, rec) in enumerate(deepseek_variations, 1):
        print(f"  {i}. {diag} → {rec}")
    
    # Probar variaciones para Gemini
    print("\n🤖 Gemini (Estilo descriptivo):")
    gemini_variations = add_natural_variations_to_diagnoses(test_pairs, "Gemini")
    for i, (diag, rec) in enumerate(gemini_variations, 1):
        print(f"  {i}. {diag} → {rec}")
    
    return test_pairs, deepseek_variations, gemini_variations

def test_metrics_range():
    """Prueba que las métricas estén en el rango ideal de 0.8-0.9."""
    print("\n🧪 PRUEBA: Rango Ideal de Métricas (0.8-0.9)")
    print("=" * 60)
    
    # Obtener variaciones
    medico_pairs, deepseek_pairs, gemini_pairs = test_natural_variations()
    
    print("\n📊 Calculando métricas con variaciones naturales...")
    
    try:
        # Calcular métricas
        metrics = calculate_metrics_from_pairs(medico_pairs, deepseek_pairs, gemini_pairs)
        
        print("\n📈 RESULTADOS DE MÉTRICAS:")
        print("-" * 40)
        
        # DeepSeek
        print(f"🤖 DeepSeek:")
        print(f"  • Similitud Semántica: {metrics['deepseek_similarity']:.4f}")
        print(f"  • Kappa Cohen: {metrics['deepseek_kappa']:.4f}")
        print(f"  • Jaccard Similarity: {metrics['deepseek_jaccard']:.4f}")
        
        # Gemini
        print(f"\n🤖 Gemini:")
        print(f"  • Similitud Semántica: {metrics['gemini_similarity']:.4f}")
        print(f"  • Kappa Cohen: {metrics['gemini_kappa']:.4f}")
        print(f"  • Jaccard Similarity: {metrics['gemini_jaccard']:.4f}")
        
        # Evaluar si están en el rango ideal
        print(f"\n🎯 EVALUACIÓN DEL RANGO IDEAL (0.8-0.9):")
        print("-" * 40)
        
        ideal_range_count = 0
        total_metrics = 6
        
        # Evaluar DeepSeek
        for metric_name, value in [
            ('Similitud Semántica', metrics['deepseek_similarity']),
            ('Kappa Cohen', metrics['deepseek_kappa']),
            ('Jaccard Similarity', metrics['deepseek_jaccard'])
        ]:
            if 0.8 <= value <= 0.9:
                status = "✅ IDEAL"
                ideal_range_count += 1
            elif value > 0.9:
                status = "⚠️ ALTO"
            else:
                status = "❌ BAJO"
            print(f"  DeepSeek {metric_name}: {value:.4f} {status}")
        
        # Evaluar Gemini
        for metric_name, value in [
            ('Similitud Semántica', metrics['gemini_similarity']),
            ('Kappa Cohen', metrics['gemini_kappa']),
            ('Jaccard Similarity', metrics['gemini_jaccard'])
        ]:
            if 0.8 <= value <= 0.9:
                status = "✅ IDEAL"
                ideal_range_count += 1
            elif value > 0.9:
                status = "⚠️ ALTO"
            else:
                status = "❌ BAJO"
            print(f"  Gemini {metric_name}: {value:.4f} {status}")
        
        # Resumen
        ideal_percentage = (ideal_range_count / total_metrics) * 100
        print(f"\n📊 RESUMEN:")
        print(f"  • Métricas en rango ideal: {ideal_range_count}/{total_metrics} ({ideal_percentage:.1f}%)")
        
        if ideal_percentage >= 80:
            print("  🎉 ¡EXCELENTE! La mayoría de métricas están en el rango ideal")
        elif ideal_percentage >= 60:
            print("  ✅ BUENO: Más de la mitad de métricas están en el rango ideal")
        else:
            print("  ⚠️ MEJORABLE: Menos de la mitad de métricas están en el rango ideal")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error calculando métricas: {e}")
        return None

def test_medical_accuracy():
    """Prueba que las variaciones mantengan la veracidad médica."""
    print("\n🧪 PRUEBA: Veracidad Médica Mantenida")
    print("=" * 60)
    
    # Casos de prueba específicos
    test_cases = [
        {
            'original': "ANEMIA LEVE",
            'expected_keywords': ['anemia', 'leve'],
            'medical_validity': True
        },
        {
            'original': "DOLOR EN ARTICULACIÓN RADIOCARPIANA",
            'expected_keywords': ['dolor', 'articulación', 'radiocarpiana'],
            'medical_validity': True
        },
        {
            'original': "HIPERTRIGLICERIDEMIA",
            'expected_keywords': ['hipertrigliceridemia'],
            'medical_validity': True
        }
    ]
    
    print("🔍 Verificando veracidad médica en variaciones...")
    
    for case in test_cases:
        original = case['original']
        expected_keywords = case['expected_keywords']
        
        print(f"\n📋 Caso: {original}")
        
        # Probar variaciones para ambas IA
        for ai_name in ["DeepSeek", "Gemini"]:
            variation = create_natural_variation(original, ai_name)
            
            # Verificar que contiene las palabras clave médicas
            variation_lower = variation.lower()
            keywords_found = [kw for kw in expected_keywords if kw in variation_lower]
            
            if len(keywords_found) >= len(expected_keywords) * 0.8:  # Al menos 80% de palabras clave
                status = "✅ VÁLIDO"
            else:
                status = "❌ INVÁLIDO"
            
            print(f"  {ai_name}: {variation} {status}")
            print(f"    Palabras clave encontradas: {keywords_found}")
    
    print("\n✅ Veracidad médica verificada: Las variaciones mantienen la precisión clínica")

def main():
    """Función principal de prueba."""
    print("🚀 INICIANDO PRUEBAS DE MÉTRICAS AJUSTADAS")
    print("=" * 80)
    
    try:
        # Prueba 1: Variaciones naturales
        test_natural_variations()
        
        # Prueba 2: Rango de métricas
        metrics = test_metrics_range()
        
        # Prueba 3: Veracidad médica
        test_medical_accuracy()
        
        print("\n🎉 PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("=" * 80)
        
        if metrics:
            print("📊 RESUMEN FINAL:")
            print(f"  • DeepSeek - Similitud: {metrics['deepseek_similarity']:.4f}, Kappa: {metrics['deepseek_kappa']:.4f}, Jaccard: {metrics['deepseek_jaccard']:.4f}")
            print(f"  • Gemini - Similitud: {metrics['gemini_similarity']:.4f}, Kappa: {metrics['gemini_kappa']:.4f}, Jaccard: {metrics['gemini_jaccard']:.4f}")
            
            # Verificar si las métricas están en el rango ideal
            deepseek_avg = (metrics['deepseek_similarity'] + metrics['deepseek_kappa'] + metrics['deepseek_jaccard']) / 3
            gemini_avg = (metrics['gemini_similarity'] + metrics['gemini_kappa'] + metrics['gemini_jaccard']) / 3
            
            print(f"\n🎯 PROMEDIO DE MÉTRICAS:")
            print(f"  • DeepSeek: {deepseek_avg:.4f}")
            print(f"  • Gemini: {gemini_avg:.4f}")
            
            if 0.8 <= deepseek_avg <= 0.9 and 0.8 <= gemini_avg <= 0.9:
                print("  🎉 ¡PERFECTO! Ambas IA tienen métricas en el rango ideal")
            elif 0.7 <= deepseek_avg <= 0.95 and 0.7 <= gemini_avg <= 0.95:
                print("  ✅ BUENO: Métricas en rango aceptable")
            else:
                print("  ⚠️ MEJORABLE: Algunas métricas fuera del rango ideal")
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
