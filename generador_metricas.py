#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Métricas de Similitud para Diagnósticos Médicos
============================================================

Este módulo implementa tres métricas principales para evaluar la concordancia
entre diagnósticos y recomendaciones médicas:

1. Similitud de Jaccard
2. Similitud de Cosenos  
3. Índice de Kappa Cohen

Autor: Sistema de Análisis Médico Ocupacional
Fecha: 2024
"""

import re
import math
from typing import List, Dict, Tuple, Set
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GeneradorMetricas:
    """Clase principal para generar métricas de similitud médica."""
    
    def __init__(self):
        """Inicializa el generador con términos médicos y configuraciones."""
        self.terminos_medicos = self._cargar_terminos_medicos()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,  # No usar stop words predefinidas
            ngram_range=(1, 2),
            max_features=1000
        )
    
    def _cargar_terminos_medicos(self) -> Set[str]:
        """Carga el conjunto de términos médicos reconocidos."""
        return {
            # Condiciones médicas principales
            'obesidad', 'obesidad mórbida', 'obesidad morbida', 'imc', 'índice masa corporal',
            'ametropia', 'ametropía', 'corregida', 'lentes', 'correctores',
            'linfopenia', 'linopenia', 'leucocitos', 'linfocitos',
            'hipotiroidismo', 'tiroides', 'tsh', 't4', 't3',
            'prediabetes', 'glucosa', 'glicemia', 'diabetes', 'hemoglobina glicosilada',
            'hipertensión', 'presión arterial', 'tensión arterial', 'hta',
            'dislipidemia', 'colesterol', 'triglicéridos', 'hdl', 'ldl',
            'anemia', 'hemoglobina', 'hematocrito', 'ferritina',
            
            # Especialidades médicas
            'endocrinología', 'endocrino', 'medicina interna', 'internista',
            'cardiología', 'cardiólogo', 'oftalmología', 'oftalmólogo',
            'nutrición', 'nutricionista', 'dietista',
            
            # Tratamientos y recomendaciones
            'dieta', 'alimentación', 'nutrición', 'ejercicio', 'actividad física',
            'medicamento', 'tratamiento', 'terapia', 'seguimiento', 'control',
            'derivación', 'consulta', 'evaluación', 'estudio', 'prueba',
            'lentes', 'gafas', 'anteojos', 'corrección visual',
            
            # Términos de seguimiento
            'continuar', 'mantener', 'repetir', 'monitorear', 'vigilar',
            'sugerir', 'recomendar', 'indicar', 'prescribir'
        }
    
    def extraer_datos_tabla(self) -> Dict[str, List[Dict]]:
        """
        Extrae los datos de diagnósticos y recomendaciones de la tabla comparativa.
        
        Returns:
            Dict con los datos estructurados por fuente (médico, deepseek, gemini)
        """
        # Datos extraídos de la tabla comparativa de la imagen
        datos_tabla = {
            'medico_sistema': [
                {
                    'caso': 'Obesidad Mórbida',
                    'diagnostico': 'OBESIDAD MORBIDA',
                    'recomendacion': 'SE SUGIERE SEGUIMIENTO POR ENDOCRINOLOGIA'
                },
                {
                    'caso': 'Ametropía Corregida', 
                    'diagnostico': 'AMETROPIA CORREGIDA',
                    'recomendacion': 'CONTINUAR CON SUS LENTES CORRECTORES Y CONTRO..'
                },
                {
                    'caso': 'Linfopenia',
                    'diagnostico': 'LINFOPENIA', 
                    'recomendacion': 'SE SUGIERE EVALUACIÓN POR MEDICINA INTERNA'
                },
                {
                    'caso': 'Hipotiroidismo No Especificado',
                    'diagnostico': 'HIPOTIROIDISMO, NO ESPECIFICADO',
                    'recomendacion': 'SE SUGIERE SEGUIMIENTO POR ENDOCRINOLOGIA'
                },
                {
                    'caso': 'Prediabetes/Glucosa Elevada',
                    'diagnostico': 'Sin diagnóstico',
                    'recomendacion': ''
                },
                {
                    'caso': 'Hipotiroidismo No Especificado (2)',
                    'diagnostico': 'Sin diagnóstico',
                    'recomendacion': ''
                }
            ],
            'deepseek': [
                {
                    'caso': 'Obesidad Mórbida',
                    'diagnostico': 'Obesidad mórbida',
                    'recomendacion': 'Derivación a endocrinología para manejo integ...'
                },
                {
                    'caso': 'Ametropía Corregida',
                    'diagnostico': 'Sin diagnóstico',
                    'recomendacion': ''
                },
                {
                    'caso': 'Linfopenia', 
                    'diagnostico': 'Linopenia',
                    'recomendacion': 'Medicina interna para estudio de linfopenia'
                },
                {
                    'caso': 'Hipotiroidismo No Especificado',
                    'diagnostico': 'Sin diagnóstico',
                    'recomendacion': ''
                },
                {
                    'caso': 'Prediabetes/Glucosa Elevada',
                    'diagnostico': 'Prediabetes',
                    'recomendacion': 'Derivación a endocrinología para manejo integ...'
                },
                {
                    'caso': 'Hipotiroidismo No Especificado (2)',
                    'diagnostico': 'Hipotiroidismo no especificado',
                    'recomendacion': ''
                }
            ],
            'gemini': [
                {
                    'caso': 'Obesidad Mórbida',
                    'diagnostico': 'Obesidad mórbida',
                    'recomendacion': 'Manejo multidisciplinario incluyendo dieta, e..'
                },
                {
                    'caso': 'Ametropía Corregida',
                    'diagnostico': 'Sin diagnóstico', 
                    'recomendacion': ''
                },
                {
                    'caso': 'Linfopenia',
                    'diagnostico': 'Linopenia',
                    'recomendacion': 'Investigar causa subyacente mediante hemogram...'
                },
                {
                    'caso': 'Hipotiroidismo No Especificado',
                    'diagnostico': 'Sin diagnóstico',
                    'recomendacion': ''
                },
                {
                    'caso': 'Prediabetes/Glucosa Elevada',
                    'diagnostico': 'Glucosa: nivel ligeramente elevado, s...',
                    'recomendacion': 'Prueba de tolerancia oral a la glucosa (ogtt).'
                },
                {
                    'caso': 'Hipotiroidismo No Especificado (2)',
                    'diagnostico': 'Hipotiroidismo no especificado',
                    'recomendacion': ''
                }
            ]
        }
        
        return datos_tabla
    
    def extraer_terminos_medicos(self, texto: str) -> List[str]:
        """
        Extrae términos médicos de un texto dado.
        
        Args:
            texto: Texto del cual extraer términos médicos
            
        Returns:
            Lista de términos médicos encontrados
        """
        if not texto or texto.strip() == '' or texto.lower() == 'sin diagnóstico':
            return []
        
        # Normalizar texto
        texto_normalizado = texto.lower().strip()
        
        # Buscar términos médicos
        terminos_encontrados = []
        for termino in self.terminos_medicos:
            if termino.lower() in texto_normalizado:
                terminos_encontrados.append(termino.lower())
        
        return terminos_encontrados
    
    def calcular_similitud_jaccard(self, texto1: str, texto2: str) -> float:
        """
        Calcula la similitud de Jaccard entre dos textos médicos.
        
        Args:
            texto1: Primer texto médico
            texto2: Segundo texto médico
            
        Returns:
            Valor de similitud Jaccard (0.0 - 1.0)
        """
        try:
            # Extraer términos médicos
            terminos1 = set(self.extraer_terminos_medicos(texto1))
            terminos2 = set(self.extraer_terminos_medicos(texto2))
            
            # Casos especiales
            if len(terminos1) == 0 and len(terminos2) == 0:
                return 1.0  # Ambos vacíos = perfecta similitud
            
            if len(terminos1) == 0 or len(terminos2) == 0:
                return 0.0  # Uno vacío, otro no = sin similitud
            
            # Calcular intersección y unión
            interseccion = terminos1 & terminos2
            union = terminos1 | terminos2
            
            # Calcular Jaccard
            jaccard = len(interseccion) / len(union) if len(union) > 0 else 0.0
            
            return jaccard
            
        except Exception as e:
            print(f"❌ Error calculando Jaccard: {e}")
            return 0.0
    
    def calcular_similitud_cosenos(self, texto1: str, texto2: str) -> float:
        """
        Calcula la similitud de cosenos entre dos textos médicos usando TF-IDF.
        
        Args:
            texto1: Primer texto médico
            texto2: Segundo texto médico
            
        Returns:
            Valor de similitud de cosenos (0.0 - 1.0)
        """
        try:
            # Preparar textos
            textos = [texto1, texto2]
            
            # Filtrar textos vacíos
            textos_validos = [t for t in textos if t and t.strip() and t.lower() != 'sin diagnóstico']
            
            if len(textos_validos) < 2:
                return 1.0 if len(textos_validos) == 0 else 0.0
            
            # Calcular TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(textos_validos)
            
            # Calcular similitud de cosenos
            similitud = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similitud
            
        except Exception as e:
            print(f"❌ Error calculando similitud de cosenos: {e}")
            return 0.0
    
    def calcular_kappa_cohen(self, datos_medico: List[Dict], datos_ia: List[Dict]) -> float:
        """
        Calcula el índice de Kappa Cohen para evaluar concordancia entre evaluadores.
        
        Args:
            datos_medico: Lista de diagnósticos del médico/sistema
            datos_ia: Lista de diagnósticos del sistema de IA
            
        Returns:
            Valor del índice Kappa Cohen (-1.0 a 1.0)
        """
        try:
            # Crear matriz de confusión
            categorias = set()
            
            # Recopilar todas las categorías únicas
            for caso in datos_medico + datos_ia:
                diagnostico = caso.get('diagnostico', '').strip()
                if diagnostico and diagnostico.lower() != 'sin diagnóstico':
                    categorias.add(diagnostico.lower())
            
            # Convertir a lista ordenada
            categorias = sorted(list(categorias))
            
            if len(categorias) == 0:
                return 1.0  # Sin categorías = perfecta concordancia
            
            # Crear matriz de confusión
            n_categorias = len(categorias)
            matriz_confusion = np.zeros((n_categorias, n_categorias))
            
            # Mapear diagnósticos a índices
            categoria_to_idx = {cat: idx for idx, cat in enumerate(categorias)}
            
            # Llenar matriz de confusión
            for i in range(len(datos_medico)):
                medico_diag = datos_medico[i].get('diagnostico', '').strip().lower()
                ia_diag = datos_ia[i].get('diagnostico', '').strip().lower()
                
                # Manejar casos sin diagnóstico
                if medico_diag == 'sin diagnóstico' or medico_diag == '':
                    medico_idx = -1
                else:
                    medico_idx = categoria_to_idx.get(medico_diag, -1)
                
                if ia_diag == 'sin diagnóstico' or ia_diag == '':
                    ia_idx = -1
                else:
                    ia_idx = categoria_to_idx.get(ia_diag, -1)
                
                # Solo contar si ambos tienen diagnóstico válido
                if medico_idx >= 0 and ia_idx >= 0:
                    matriz_confusion[medico_idx, ia_idx] += 1
            
            # Calcular métricas de Kappa
            n_total = np.sum(matriz_confusion)
            
            if n_total == 0:
                return 1.0  # Sin datos = perfecta concordancia
            
            # Concordancia observada (Po)
            concordancia_observada = np.trace(matriz_confusion) / n_total
            
            # Concordancia esperada (Pe)
            suma_filas = np.sum(matriz_confusion, axis=1)
            suma_columnas = np.sum(matriz_confusion, axis=0)
            concordancia_esperada = np.sum(suma_filas * suma_columnas) / (n_total ** 2)
            
            # Calcular Kappa
            if concordancia_esperada == 1.0:
                kappa = 1.0
            else:
                kappa = (concordancia_observada - concordancia_esperada) / (1.0 - concordancia_esperada)
            
            return kappa
            
        except Exception as e:
            print(f"❌ Error calculando Kappa Cohen: {e}")
            return 0.0
    
    def generar_metricas_completas(self) -> Dict[str, Dict]:
        """
        Genera todas las métricas de similitud para los datos de la tabla.
        
        Returns:
            Diccionario con todas las métricas calculadas
        """
        print("🔍 Generando métricas de similitud...")
        
        # Extraer datos de la tabla
        datos = self.extraer_datos_tabla()
        
        resultados = {
            'jaccard': {},
            'cosenos': {},
            'kappa_cohen': {}
        }
        
        # Calcular métricas Jaccard y Cosenos por caso
        casos = datos['medico_sistema']
        
        for i, caso in enumerate(casos):
            caso_nombre = caso['caso']
            
            # Datos del médico
            medico_diag = caso['diagnostico']
            medico_rec = caso['recomendacion']
            
            # Datos DeepSeek
            deepseek_diag = datos['deepseek'][i]['diagnostico']
            deepseek_rec = datos['deepseek'][i]['recomendacion']
            
            # Datos Gemini
            gemini_diag = datos['gemini'][i]['diagnostico']
            gemini_rec = datos['gemini'][i]['recomendacion']
            
            # Calcular Jaccard
            resultados['jaccard'][caso_nombre] = {
                'medico_vs_deepseek_diag': self.calcular_similitud_jaccard(medico_diag, deepseek_diag),
                'medico_vs_gemini_diag': self.calcular_similitud_jaccard(medico_diag, gemini_diag),
                'deepseek_vs_gemini_diag': self.calcular_similitud_jaccard(deepseek_diag, gemini_diag),
                'medico_vs_deepseek_rec': self.calcular_similitud_jaccard(medico_rec, deepseek_rec),
                'medico_vs_gemini_rec': self.calcular_similitud_jaccard(medico_rec, gemini_rec),
                'deepseek_vs_gemini_rec': self.calcular_similitud_jaccard(deepseek_rec, gemini_rec)
            }
            
            # Calcular Cosenos
            resultados['cosenos'][caso_nombre] = {
                'medico_vs_deepseek_diag': self.calcular_similitud_cosenos(medico_diag, deepseek_diag),
                'medico_vs_gemini_diag': self.calcular_similitud_cosenos(medico_diag, gemini_diag),
                'deepseek_vs_gemini_diag': self.calcular_similitud_cosenos(deepseek_diag, gemini_diag),
                'medico_vs_deepseek_rec': self.calcular_similitud_cosenos(medico_rec, deepseek_rec),
                'medico_vs_gemini_rec': self.calcular_similitud_cosenos(medico_rec, gemini_rec),
                'deepseek_vs_gemini_rec': self.calcular_similitud_cosenos(deepseek_rec, gemini_rec)
            }
        
        # Calcular Kappa Cohen para diagnósticos
        resultados['kappa_cohen'] = {
            'medico_vs_deepseek': self.calcular_kappa_cohen(datos['medico_sistema'], datos['deepseek']),
            'medico_vs_gemini': self.calcular_kappa_cohen(datos['medico_sistema'], datos['gemini']),
            'deepseek_vs_gemini': self.calcular_kappa_cohen(datos['deepseek'], datos['gemini'])
        }
        
        return resultados
    
    def imprimir_resultados(self, resultados: Dict[str, Dict]) -> None:
        """
        Imprime los resultados de las métricas de forma organizada.
        
        Args:
            resultados: Diccionario con los resultados calculados
        """
        print("\n" + "="*80)
        print("📊 RESULTADOS DE MÉTRICAS DE SIMILITUD")
        print("="*80)
        
        # Imprimir Jaccard
        print("\n🔸 SIMILITUD DE JACCARD")
        print("-" * 50)
        for caso, metricas in resultados['jaccard'].items():
            print(f"\n📋 {caso}:")
            for metrica, valor in metricas.items():
                print(f"   {metrica}: {valor:.3f}")
        
        # Imprimir Cosenos
        print("\n🔸 SIMILITUD DE COSENOS")
        print("-" * 50)
        for caso, metricas in resultados['cosenos'].items():
            print(f"\n📋 {caso}:")
            for metrica, valor in metricas.items():
                print(f"   {metrica}: {valor:.3f}")
        
        # Imprimir Kappa Cohen
        print("\n🔸 ÍNDICE DE KAPPA COHEN")
        print("-" * 50)
        for comparacion, valor in resultados['kappa_cohen'].items():
            print(f"   {comparacion}: {valor:.3f}")
        
        # Resumen estadístico
        print("\n📈 RESUMEN ESTADÍSTICO")
        print("-" * 50)
        
        # Promedios Jaccard
        todos_jaccard = []
        for caso_metricas in resultados['jaccard'].values():
            todos_jaccard.extend(caso_metricas.values())
        
        if todos_jaccard:
            print(f"   Jaccard promedio: {np.mean(todos_jaccard):.3f}")
            print(f"   Jaccard desviación: {np.std(todos_jaccard):.3f}")
        
        # Promedios Cosenos
        todos_cosenos = []
        for caso_metricas in resultados['cosenos'].values():
            todos_cosenos.extend(caso_metricas.values())
        
        if todos_cosenos:
            print(f"   Cosenos promedio: {np.mean(todos_cosenos):.3f}")
            print(f"   Cosenos desviación: {np.std(todos_cosenos):.3f}")
        
        # Promedio Kappa
        kappa_values = list(resultados['kappa_cohen'].values())
        if kappa_values:
            print(f"   Kappa promedio: {np.mean(kappa_values):.3f}")
            print(f"   Kappa desviación: {np.std(kappa_values):.3f}")


def main():
    """Función principal para ejecutar el generador de métricas."""
    print("🚀 Iniciando Generador de Métricas de Similitud Médica")
    print("=" * 60)
    
    # Crear instancia del generador
    generador = GeneradorMetricas()
    
    # Generar métricas
    resultados = generador.generar_metricas_completas()
    
    # Imprimir resultados
    generador.imprimir_resultados(resultados)
    
    print("\n✅ Proceso completado exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()
