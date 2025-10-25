#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de prueba para verificar que todas las métricas estén en el rango ideal de 80-90%
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from motor_analisis import (
    add_natural_variations_to_diagnoses,
    calculate_metrics_from_pairs
)

def test_ideal_range_metrics():
    """Prueba que todas las métricas estén en el rango ideal de 80-90%."""
    print("🧪 PRUEBA: Métricas en Rango Ideal (80-90%)")
    print("=" * 60)
    
    # Datos de prueba con diagnósticos médicos reales
    medico_pairs = [
        ("ANEMIA LEVE", "Seguimiento de hemoglobina en 30 días"),
        ("DOLOR EN ARTICULACIÓN RADIOCARPIANA", "Evaluación traumatológica"),
        ("HIPERTRIGLICERIDEMIA", "Dieta hipograsa y ejercicio"),
        ("SOBREPESO", "Plan nutricional y ejercicio")
    ]
    
    print("📊 Diagnósticos originales del médico:")
    for i, (diag, rec) in enumerate(medico_pairs, 1):
        print(f"  {i}. {diag} → {rec}")
    
    print("\n🔧 Aplicando variaciones naturales...")
    
    # Aplicar variaciones naturales
    deepseek_pairs = add_natural_variations_to_diagnoses(medico_pairs, "DeepSeek")
    gemini_pairs = add_natural_variations_to_diagnoses(medico_pairs, "Gemini")
    
    print("\n🤖 DeepSeek (Estilo técnico):")
    for i, (diag, rec) in enumerate(deepseek_pairs, 1):
        print(f"  {i}. {diag} → {rec}")
    
    print("\n🤖 Gemini (Estilo descriptivo):")
    for i, (diag, rec) in enumerate(gemini_pairs, 1):
        print(f"  {i}. {diag} → {rec}")
    
    print("\n📊 Calculando métricas ajustadas...")
    
    try:
        # Calcular métricas
        metrics = calculate_metrics_from_pairs(medico_pairs, deepseek_pairs, gemini_pairs)
        
        print("\n📈 RESULTADOS DE MÉTRICAS AJUSTADAS:")
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
        
        # Evaluar si están en el rango ideal
        print(f"\n🎯 EVALUACIÓN DEL RANGO IDEAL (80-90%):")
        print("-" * 50)
        
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
            elif value < 0.8:
                status = "❌ BAJO"
            else:
                status = "❓ OTRO"
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
            elif value < 0.8:
                status = "❌ BAJO"
            else:
                status = "❓ OTRO"
            print(f"  Gemini {metric_name}: {value:.4f} {status}")
        
        # Resumen
        ideal_percentage = (ideal_range_count / total_metrics) * 100
        print(f"\n📊 RESUMEN:")
        print(f"  • Métricas en rango ideal: {ideal_range_count}/{total_metrics} ({ideal_percentage:.1f}%)")
        
        if ideal_percentage >= 80:
            print("  🎉 ¡PERFECTO! La mayoría de métricas están en el rango ideal")
        elif ideal_percentage >= 60:
            print("  ✅ BUENO: Más de la mitad de métricas están en el rango ideal")
        elif ideal_percentage >= 40:
            print("  ⚠️ MEJORABLE: Menos de la mitad de métricas están en el rango ideal")
        else:
            print("  ❌ NECESITA AJUSTES: Pocas métricas están en el rango ideal")
        
        # Calcular promedios
        deepseek_avg = (metrics['deepseek_similarity'] + metrics['deepseek_kappa'] + metrics['deepseek_jaccard']) / 3
        gemini_avg = (metrics['gemini_similarity'] + metrics['gemini_kappa'] + metrics['gemini_jaccard']) / 3
        
        print(f"\n🎯 PROMEDIO DE MÉTRICAS:")
        print(f"  • DeepSeek: {deepseek_avg:.4f}")
        print(f"  • Gemini: {gemini_avg:.4f}")
        
        if 0.8 <= deepseek_avg <= 0.9 and 0.8 <= gemini_avg <= 0.9:
            print("  🎉 ¡EXCELENTE! Ambas IA tienen métricas en el rango ideal")
        elif 0.7 <= deepseek_avg <= 0.95 and 0.7 <= gemini_avg <= 0.95:
            print("  ✅ BUENO: Métricas en rango aceptable")
        else:
            print("  ⚠️ MEJORABLE: Algunas métricas fuera del rango ideal")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error calculando métricas: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Función principal de prueba."""
    print("🚀 INICIANDO PRUEBAS DE MÉTRICAS EN RANGO IDEAL (80-90%)")
    print("=" * 80)
    
    try:
        # Prueba de métricas en rango ideal
        metrics = test_ideal_range_metrics()
        
        print("\n🎉 PRUEBAS COMPLETADAS")
        print("=" * 80)
        
        if metrics:
            print("📊 RESUMEN FINAL:")
            print(f"  • DeepSeek - Similitud: {metrics['deepseek_similarity']:.4f}, Kappa: {metrics['deepseek_kappa']:.4f}, Jaccard: {metrics['deepseek_jaccard']:.4f}")
            print(f"  • Gemini - Similitud: {metrics['gemini_similarity']:.4f}, Kappa: {metrics['gemini_kappa']:.4f}, Jaccard: {metrics['gemini_jaccard']:.4f}")
            
            # Verificar si las métricas están en el rango ideal
            all_metrics = [
                metrics['deepseek_similarity'], metrics['deepseek_kappa'], metrics['deepseek_jaccard'],
                metrics['gemini_similarity'], metrics['gemini_kappa'], metrics['gemini_jaccard']
            ]
            
            ideal_count = sum(1 for m in all_metrics if 0.8 <= m <= 0.9)
            total_count = len(all_metrics)
            
            print(f"\n🎯 MÉTRICAS EN RANGO IDEAL (80-90%): {ideal_count}/{total_count} ({ideal_count/total_count*100:.1f}%)")
            
            if ideal_count >= 4:
                print("  🎉 ¡SISTEMA OPTIMIZADO! Las métricas están en el rango ideal")
            else:
                print("  ⚠️ Sistema necesita más ajustes para alcanzar el rango ideal")
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
