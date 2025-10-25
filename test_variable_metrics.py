#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de prueba para verificar que las métricas sean variables y reflejen diferencias reales
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from motor_analisis import (
    ensure_complete_diagnosis_generation,
    calculate_metrics_from_pairs,
    adjust_metrics_display
)

def test_variable_metrics():
    """Prueba que las métricas sean variables y reflejen diferencias reales."""
    print("🧪 PRUEBA: Métricas Variables que Reflejan Diferencias Reales")
    print("=" * 80)
    
    # Casos de prueba con diferentes niveles de similitud
    test_cases = [
        {
            'name': 'Caso 1: Diagnósticos Idénticos',
            'medico_pairs': [
                ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
                ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica")
            ],
            'deepseek_pairs': [
                ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
                ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica")
            ],
            'gemini_pairs': [
                ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
                ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica")
            ]
        },
        {
            'name': 'Caso 2: Diagnósticos Similares',
            'medico_pairs': [
                ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
                ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica")
            ],
            'deepseek_pairs': [
                ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
                ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica")
            ],
            'gemini_pairs': [
                ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
                ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica")
            ]
        },
        {
            'name': 'Caso 3: Diagnósticos Diferentes',
            'medico_pairs': [
                ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
                ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica")
            ],
            'deepseek_pairs': [
                ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
                ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica")
            ],
            'gemini_pairs': [
                ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
                ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica")
            ]
        }
    ]
    
    all_results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"📋 {case['name']}")
        print(f"{'='*60}")
        
        print("📊 Diagnósticos del médico:")
        for j, (diag, rec) in enumerate(case['medico_pairs'], 1):
            print(f"  {j}. {diag} → {rec}")
        
        print("\n🤖 DeepSeek:")
        for j, (diag, rec) in enumerate(case['deepseek_pairs'], 1):
            print(f"  {j}. {diag} → {rec}")
        
        print("\n🤖 Gemini:")
        for j, (diag, rec) in enumerate(case['gemini_pairs'], 1):
            print(f"  {j}. {diag} → {rec}")
        
        print("\n📊 Calculando métricas...")
        
        try:
            # Calcular métricas
            metrics = calculate_metrics_from_pairs(
                case['medico_pairs'], 
                case['deepseek_pairs'], 
                case['gemini_pairs']
            )
            
            print(f"\n📈 RESULTADOS PARA {case['name']}:")
            print("-" * 50)
            
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
            
            # Guardar resultados para comparación
            all_results.append({
                'case': case['name'],
                'metrics': metrics
            })
            
        except Exception as e:
            print(f"❌ Error calculando métricas para {case['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Análisis de variabilidad
    print(f"\n{'='*80}")
    print("📊 ANÁLISIS DE VARIABILIDAD DE MÉTRICAS")
    print(f"{'='*80}")
    
    if len(all_results) >= 2:
        # Comparar métricas entre casos
        print("\n🔍 Comparación entre casos:")
        
        for metric_name in ['deepseek_similarity', 'deepseek_kappa', 'deepseek_jaccard', 
                           'gemini_similarity', 'gemini_kappa', 'gemini_jaccard']:
            values = [result['metrics'][metric_name] for result in all_results]
            min_val = min(values)
            max_val = max(values)
            variation = max_val - min_val
            
            print(f"  {metric_name}:")
            print(f"    Rango: {min_val:.4f} - {max_val:.4f}")
            print(f"    Variación: {variation:.4f}")
            
            if variation < 0.01:
                print(f"    ⚠️ VARIACIÓN MUY BAJA (todas las métricas son similares)")
            elif variation < 0.05:
                print(f"    ⚠️ VARIACIÓN BAJA")
            elif variation < 0.1:
                print(f"    ✅ VARIACIÓN MODERADA")
            else:
                print(f"    ✅ VARIACIÓN ALTA (buena diferenciación)")
        
        # Verificar si hay diferencias significativas
        total_variation = sum(max([result['metrics'][metric] for result in all_results]) - 
                             min([result['metrics'][metric] for result in all_results]) 
                             for metric in ['deepseek_similarity', 'deepseek_kappa', 'deepseek_jaccard', 
                                           'gemini_similarity', 'gemini_kappa', 'gemini_jaccard'])
        
        print(f"\n📊 RESUMEN DE VARIABILIDAD:")
        print(f"  • Variación total: {total_variation:.4f}")
        
        if total_variation < 0.1:
            print("  ⚠️ PROBLEMA: Las métricas son muy similares entre casos")
            print("  💡 Recomendación: Ajustar el algoritmo para mayor variabilidad")
        elif total_variation < 0.3:
            print("  ⚠️ MEJORABLE: Las métricas tienen variación moderada")
            print("  💡 Recomendación: Considerar ajustes adicionales")
        else:
            print("  ✅ BUENO: Las métricas tienen buena variabilidad")
            print("  💡 El sistema refleja correctamente las diferencias")
    
    return all_results

def main():
    """Función principal de prueba."""
    print("🚀 INICIANDO PRUEBAS DE VARIABILIDAD DE MÉTRICAS")
    print("=" * 80)
    
    try:
        # Prueba de variabilidad
        results = test_variable_metrics()
        
        print("\n🎉 PRUEBAS COMPLETADAS")
        print("=" * 80)
        
        if results:
            print("📊 RESUMEN FINAL:")
            for i, result in enumerate(results, 1):
                metrics = result['metrics']
                print(f"  Caso {i}: {result['case']}")
                print(f"    DeepSeek - Similitud: {metrics['deepseek_similarity']:.4f}, Kappa: {metrics['deepseek_kappa']:.4f}, Jaccard: {metrics['deepseek_jaccard']:.4f}")
                print(f"    Gemini - Similitud: {metrics['gemini_similarity']:.4f}, Kappa: {metrics['gemini_kappa']:.4f}, Jaccard: {metrics['gemini_jaccard']:.4f}")
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
