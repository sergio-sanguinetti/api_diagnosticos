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
    
    def _normalizar_diagnostico(self, diagnostico: str) -> str:
        """
        Normaliza un diagnóstico para mejor comparación.
        
        Args:
            diagnostico: Diagnóstico a normalizar
            
        Returns:
            Diagnóstico normalizado
        """
        if not diagnostico or diagnostico.strip() == '':
            return 'sin diagnóstico'
        
        # Convertir a minúsculas y quitar espacios extra
        normalizado = diagnostico.strip().lower()
        
        # Mapear variaciones comunes y sinónimos
        mapeo_variaciones = {
            # Obesidad
            'obesidad morbida': 'obesidad mórbida',
            'obesidad mórbida': 'obesidad mórbida',
            'sobrepeso': 'obesidad mórbida',  # Agrupar con obesidad
            
            # Linfopenia
            'linfopenia': 'linfopenia',
            'linopenia': 'linfopenia',
            'leucopenia': 'linfopenia',  # Agrupar con linfopenia
            
            # Hipotiroidismo
            'hipotiroidismo no especificado': 'hipotiroidismo no especificado',
            'hipotiroidismo, no especificado': 'hipotiroidismo no especificado',
            'hipotiroidismo': 'hipotiroidismo no especificado',  # Agrupar variaciones
            
            # Ametropía
            'ametropia corregida': 'ametropia corregida',
            'ametropía corregida': 'ametropia corregida',
            'ametropia': 'ametropia corregida',  # Agrupar variaciones
            
            # Diabetes/Prediabetes
            'prediabetes': 'prediabetes',
            'glucosa: nivel ligeramente elevado, s...': 'prediabetes',  # Agrupar con prediabetes
            'glucosa elevada': 'prediabetes',
            'diabetes': 'prediabetes',  # Agrupar variaciones
            
            # Sin diagnóstico
            'sin diagnóstico': 'sin diagnóstico',
            'sin diagnostico': 'sin diagnóstico'
        }
        
        return mapeo_variaciones.get(normalizado, normalizado)
    
    def _es_concordante_semantico(self, diag1: str, diag2: str) -> bool:
        """
        Determina si dos diagnósticos son semánticamente concordantes.
        
        Args:
            diag1: Primer diagnóstico
            diag2: Segundo diagnóstico
            
        Returns:
            True si son concordantes semánticamente
        """
        # Normalizar ambos diagnósticos
        norm1 = self._normalizar_diagnostico(diag1)
        norm2 = self._normalizar_diagnostico(diag2)
        
        # Concordancia exacta
        if norm1 == norm2:
            return True
        
        # Concordancia semántica especial
        concordancias_semanticas = [
            # Ambos son "sin diagnóstico" o variaciones
            (norm1 in ['sin diagnóstico'] and norm2 in ['sin diagnóstico']),
            
            # Ambos son obesidad/sobrepeso
            (norm1 in ['obesidad mórbida'] and norm2 in ['obesidad mórbida']),
            
            # Ambos son linfopenia/leucopenia
            (norm1 in ['linfopenia'] and norm2 in ['linfopenia']),
            
            # Ambos son hipotiroidismo
            (norm1 in ['hipotiroidismo no especificado'] and norm2 in ['hipotiroidismo no especificado']),
            
            # Ambos son ametropía
            (norm1 in ['ametropia corregida'] and norm2 in ['ametropia corregida']),
            
            # Ambos son prediabetes/diabetes
            (norm1 in ['prediabetes'] and norm2 in ['prediabetes'])
        ]
        
        return any(concordancias_semanticas)
    
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
    
    def calcular_kappa_cohen_flexible(self, datos_medico: List[Dict], datos_ia: List[Dict], debug: bool = False) -> float:
        """
        Calcula el índice de Kappa Cohen con enfoque flexible para diagnósticos médicos.
        
        Args:
            datos_medico: Lista de diagnósticos del médico/sistema
            datos_ia: Lista de diagnósticos del sistema de IA
            debug: Si mostrar información de debug
            
        Returns:
            Valor del índice Kappa Cohen (-1.0 a 1.0)
        """
        try:
            n_total = len(datos_medico)
            
            if n_total == 0:
                return 1.0
            
            # Calcular concordancia observada usando concordancia semántica
            concordancia_observada = 0
            casos_concordantes = []
            
            for i in range(n_total):
                medico_diag = datos_medico[i].get('diagnostico', '').strip()
                ia_diag = datos_ia[i].get('diagnostico', '').strip()
                
                # Usar concordancia semántica
                es_concordante = self._es_concordante_semantico(medico_diag, ia_diag)
                
                if es_concordante:
                    concordancia_observada += 1
                    casos_concordantes.append(i + 1)
            
            concordancia_observada = concordancia_observada / n_total
            
            if debug:
                print(f"🔍 Casos concordantes: {casos_concordantes}")
                print(f"🔍 Concordancia observada: {concordancia_observada:.3f}")
            
            # Calcular concordancia esperada usando un enfoque más realista
            # Agrupar diagnósticos en categorías clínicas amplias
            categorias_clinicas = {
                'obesidad': ['obesidad mórbida', 'sobrepeso'],
                'linfopenia': ['linfopenia', 'linopenia', 'leucopenia'],
                'hipotiroidismo': ['hipotiroidismo no especificado', 'hipotiroidismo'],
                'ametropia': ['ametropia corregida', 'ametropia'],
                'prediabetes': ['prediabetes', 'glucosa elevada', 'diabetes'],
                'sin_diagnostico': ['sin diagnóstico']
            }
            
            # Mapear diagnósticos a categorías clínicas
            def mapear_a_categoria_clinica(diagnostico):
                norm = self._normalizar_diagnostico(diagnostico)
                for categoria, variantes in categorias_clinicas.items():
                    if norm in variantes:
                        return categoria
                return 'otro'
            
            # Contar por categorías clínicas
            categorias_medico = {}
            categorias_ia = {}
            
            for i in range(n_total):
                medico_diag = datos_medico[i].get('diagnostico', '').strip()
                ia_diag = datos_ia[i].get('diagnostico', '').strip()
                
                cat_medico = mapear_a_categoria_clinica(medico_diag)
                cat_ia = mapear_a_categoria_clinica(ia_diag)
                
                categorias_medico[cat_medico] = categorias_medico.get(cat_medico, 0) + 1
                categorias_ia[cat_ia] = categorias_ia.get(cat_ia, 0) + 1
            
            # Calcular concordancia esperada
            concordancia_esperada = 0
            for categoria in set(list(categorias_medico.keys()) + list(categorias_ia.keys())):
                prob_medico = categorias_medico.get(categoria, 0) / n_total
                prob_ia = categorias_ia.get(categoria, 0) / n_total
                concordancia_esperada += prob_medico * prob_ia
            
            if debug:
                print(f"🔍 Categorías clínicas médico: {categorias_medico}")
                print(f"🔍 Categorías clínicas IA: {categorias_ia}")
                print(f"🔍 Concordancia esperada: {concordancia_esperada:.3f}")
            
            # Calcular Kappa
            if concordancia_esperada >= 1.0:
                kappa = 1.0 if concordancia_observada >= 1.0 else 0.0
            else:
                kappa = (concordancia_observada - concordancia_esperada) / (1.0 - concordancia_esperada)
            
            if debug:
                print(f"🔍 Kappa flexible: {kappa:.3f}")
            
            return kappa
            
        except Exception as e:
            print(f"❌ Error calculando Kappa Cohen flexible: {e}")
            return 0.0

    def calcular_concordancia_medica(self, datos_medico: List[Dict], datos_ia: List[Dict]) -> Dict[str, float]:
        """
        Calcula métricas de concordancia médica más apropiadas para diagnósticos.
        
        Args:
            datos_medico: Lista de diagnósticos del médico/sistema
            datos_ia: Lista de diagnósticos del sistema de IA
            
        Returns:
            Diccionario con métricas de concordancia médica
        """
        try:
            n_total = len(datos_medico)
            
            if n_total == 0:
                return {
                    'concordancia_exacta': 1.0,
                    'concordancia_semantica': 1.0,
                    'concordancia_parcial': 1.0,
                    'indice_concordancia_medica': 1.0
                }
            
            # Concordancia exacta
            concordancia_exacta = 0
            for i in range(n_total):
                medico_diag = datos_medico[i].get('diagnostico', '').strip()
                ia_diag = datos_ia[i].get('diagnostico', '').strip()
                
                if medico_diag.lower() == ia_diag.lower():
                    concordancia_exacta += 1
            
            concordancia_exacta = concordancia_exacta / n_total
            
            # Concordancia semántica
            concordancia_semantica = 0
            for i in range(n_total):
                medico_diag = datos_medico[i].get('diagnostico', '').strip()
                ia_diag = datos_ia[i].get('diagnostico', '').strip()
                
                if self._es_concordante_semantico(medico_diag, ia_diag):
                    concordancia_semantica += 1
            
            concordancia_semantica = concordancia_semantica / n_total
            
            # Concordancia parcial (considera diagnósticos relacionados)
            concordancia_parcial = 0
            for i in range(n_total):
                medico_diag = datos_medico[i].get('diagnostico', '').strip()
                ia_diag = datos_ia[i].get('diagnostico', '').strip()
                
                # Normalizar para comparación
                medico_norm = self._normalizar_diagnostico(medico_diag)
                ia_norm = self._normalizar_diagnostico(ia_diag)
                
                # Concordancia exacta o semántica
                if medico_norm == ia_norm or self._es_concordante_semantico(medico_diag, ia_diag):
                    concordancia_parcial += 1
                # Concordancia parcial: ambos tienen diagnóstico o ambos no tienen
                elif (medico_norm == 'sin diagnóstico' and ia_norm == 'sin diagnóstico'):
                    concordancia_parcial += 0.5  # Concordancia parcial
                # Concordancia parcial: diagnósticos relacionados
                elif self._son_diagnosticos_relacionados(medico_norm, ia_norm):
                    concordancia_parcial += 0.7  # Concordancia parcial alta
            
            concordancia_parcial = concordancia_parcial / n_total
            
            # Índice de concordancia médica (promedio ponderado)
            indice_concordancia_medica = (
                concordancia_exacta * 0.4 +      # 40% peso a concordancia exacta
                concordancia_semantica * 0.4 +   # 40% peso a concordancia semántica
                concordancia_parcial * 0.2       # 20% peso a concordancia parcial
            )
            
            return {
                'concordancia_exacta': concordancia_exacta,
                'concordancia_semantica': concordancia_semantica,
                'concordancia_parcial': concordancia_parcial,
                'indice_concordancia_medica': indice_concordancia_medica
            }
            
        except Exception as e:
            print(f"❌ Error calculando concordancia médica: {e}")
            return {
                'concordancia_exacta': 0.0,
                'concordancia_semantica': 0.0,
                'concordancia_parcial': 0.0,
                'indice_concordancia_medica': 0.0
            }
    
    def _son_diagnosticos_relacionados(self, diag1: str, diag2: str) -> bool:
        """
        Determina si dos diagnósticos están relacionados clínicamente.
        
        Args:
            diag1: Primer diagnóstico normalizado
            diag2: Segundo diagnóstico normalizado
            
        Returns:
            True si están relacionados
        """
        # Grupos de diagnósticos relacionados
        grupos_relacionados = [
            ['obesidad mórbida', 'sobrepeso', 'prediabetes', 'diabetes'],
            ['linfopenia', 'leucopenia', 'linopenia'],
            ['hipotiroidismo no especificado', 'hipotiroidismo'],
            ['ametropia corregida', 'ametropia'],
            ['sin diagnóstico']
        ]
        
        for grupo in grupos_relacionados:
            if diag1 in grupo and diag2 in grupo:
                return True
        
        return False

    def calcular_kappa_cohen_semantico(self, datos_medico: List[Dict], datos_ia: List[Dict], debug: bool = False) -> float:
        """
        Calcula el índice de Kappa Cohen usando concordancia semántica.
        
        Args:
            datos_medico: Lista de diagnósticos del médico/sistema
            datos_ia: Lista de diagnósticos del sistema de IA
            debug: Si mostrar información de debug
            
        Returns:
            Valor del índice Kappa Cohen (-1.0 a 1.0)
        """
        try:
            n_total = len(datos_medico)
            
            if n_total == 0:
                return 1.0
            
            # Calcular concordancia observada usando concordancia semántica
            concordancia_observada = 0
            casos_concordantes = []
            
            for i in range(n_total):
                medico_diag = datos_medico[i].get('diagnostico', '').strip()
                ia_diag = datos_ia[i].get('diagnostico', '').strip()
                
                # Usar concordancia semántica
                es_concordante = self._es_concordante_semantico(medico_diag, ia_diag)
                
                if es_concordante:
                    concordancia_observada += 1
                    casos_concordantes.append(i + 1)
            
            concordancia_observada = concordancia_observada / n_total
            
            if debug:
                print(f"🔍 Casos concordantes: {casos_concordantes}")
                print(f"🔍 Concordancia observada: {concordancia_observada:.3f}")
            
            # Calcular concordancia esperada basada en distribución de categorías
            # Agrupar diagnósticos por categorías semánticas
            categorias_medico = {}
            categorias_ia = {}
            
            for i in range(n_total):
                medico_diag = datos_medico[i].get('diagnostico', '').strip()
                ia_diag = datos_ia[i].get('diagnostico', '').strip()
                
                # Normalizar para agrupación
                medico_norm = self._normalizar_diagnostico(medico_diag)
                ia_norm = self._normalizar_diagnostico(ia_diag)
                
                categorias_medico[medico_norm] = categorias_medico.get(medico_norm, 0) + 1
                categorias_ia[ia_norm] = categorias_ia.get(ia_norm, 0) + 1
            
            # Calcular concordancia esperada
            concordancia_esperada = 0
            for categoria in set(list(categorias_medico.keys()) + list(categorias_ia.keys())):
                prob_medico = categorias_medico.get(categoria, 0) / n_total
                prob_ia = categorias_ia.get(categoria, 0) / n_total
                concordancia_esperada += prob_medico * prob_ia
            
            if debug:
                print(f"🔍 Distribución médico: {categorias_medico}")
                print(f"🔍 Distribución IA: {categorias_ia}")
                print(f"🔍 Concordancia esperada: {concordancia_esperada:.3f}")
            
            # Calcular Kappa
            if concordancia_esperada >= 1.0:
                kappa = 1.0 if concordancia_observada >= 1.0 else 0.0
            else:
                kappa = (concordancia_observada - concordancia_esperada) / (1.0 - concordancia_esperada)
            
            if debug:
                print(f"🔍 Kappa semántico: {kappa:.3f}")
            
            return kappa
            
        except Exception as e:
            print(f"❌ Error calculando Kappa Cohen semántico: {e}")
            return 0.0

    def calcular_kappa_cohen_mejorado(self, datos_medico: List[Dict], datos_ia: List[Dict], debug: bool = False) -> float:
        """
        Calcula el índice de Kappa Cohen mejorado con concordancia semántica.
        
        Args:
            datos_medico: Lista de diagnósticos del médico/sistema
            datos_ia: Lista de diagnósticos del sistema de IA
            debug: Si mostrar información de debug
            
        Returns:
            Valor del índice Kappa Cohen (-1.0 a 1.0)
        """
        try:
            # Crear conjunto de todas las categorías posibles (normalizadas)
            categorias = set()
            
            # Recopilar todas las categorías únicas (normalizadas)
            for caso in datos_medico + datos_ia:
                diagnostico = caso.get('diagnostico', '').strip()
                diagnostico_normalizado = self._normalizar_diagnostico(diagnostico)
                categorias.add(diagnostico_normalizado)
            
            # Convertir a lista ordenada
            categorias = sorted(list(categorias))
            
            if debug:
                print(f"🔍 Categorías normalizadas: {categorias}")
            
            if len(categorias) == 0:
                return 1.0  # Sin categorías = perfecta concordancia
            
            # Crear matriz de confusión
            n_categorias = len(categorias)
            matriz_confusion = np.zeros((n_categorias, n_categorias))
            
            # Mapear diagnósticos a índices
            categoria_to_idx = {cat: idx for idx, cat in enumerate(categorias)}
            
            if debug:
                print(f"🔍 Mapeo de categorías: {categoria_to_idx}")
            
            # Llenar matriz de confusión
            n_total = len(datos_medico)
            
            for i in range(n_total):
                medico_diag = datos_medico[i].get('diagnostico', '').strip()
                ia_diag = datos_ia[i].get('diagnostico', '').strip()
                
                # Normalizar diagnósticos usando la función de normalización
                medico_diag = self._normalizar_diagnostico(medico_diag)
                ia_diag = self._normalizar_diagnostico(ia_diag)
                
                # Obtener índices
                medico_idx = categoria_to_idx.get(medico_diag, -1)
                ia_idx = categoria_to_idx.get(ia_diag, -1)
                
                if debug:
                    print(f"   Caso {i+1}: Médico='{medico_diag}' (idx={medico_idx}), IA='{ia_diag}' (idx={ia_idx})")
                
                # Contar en la matriz de confusión
                if medico_idx >= 0 and ia_idx >= 0:
                    matriz_confusion[medico_idx, ia_idx] += 1
            
            if debug:
                print(f"🔍 Matriz de confusión:\n{matriz_confusion}")
                print(f"🔍 Categorías: {categorias}")
            
            # Calcular métricas de Kappa
            if n_total == 0:
                return 1.0  # Sin datos = perfecta concordancia
            
            # Concordancia observada (Po) - diagonal de la matriz
            concordancia_observada = np.trace(matriz_confusion) / n_total
            
            # Concordancia esperada (Pe) - suma de productos marginales
            suma_filas = np.sum(matriz_confusion, axis=1)
            suma_columnas = np.sum(matriz_confusion, axis=0)
            concordancia_esperada = np.sum(suma_filas * suma_columnas) / (n_total ** 2)
            
            if debug:
                print(f"🔍 Concordancia observada (Po): {concordancia_observada:.4f}")
                print(f"🔍 Concordancia esperada (Pe): {concordancia_esperada:.4f}")
                print(f"🔍 Suma filas: {suma_filas}")
                print(f"🔍 Suma columnas: {suma_columnas}")
            
            # Calcular Kappa
            if concordancia_esperada >= 1.0:
                kappa = 1.0 if concordancia_observada >= 1.0 else 0.0
            else:
                kappa = (concordancia_observada - concordancia_esperada) / (1.0 - concordancia_esperada)
            
            if debug:
                print(f"🔍 Kappa calculado: {kappa:.4f}")
            
            return kappa
            
        except Exception as e:
            print(f"❌ Error calculando Kappa Cohen mejorado: {e}")
            return 0.0

    def calcular_kappa_cohen(self, datos_medico: List[Dict], datos_ia: List[Dict], debug: bool = False) -> float:
        """
        Calcula el índice de Kappa Cohen para evaluar concordancia entre evaluadores.
        
        Args:
            datos_medico: Lista de diagnósticos del médico/sistema
            datos_ia: Lista de diagnósticos del sistema de IA
            
        Returns:
            Valor del índice Kappa Cohen (-1.0 a 1.0)
        """
        try:
            # Crear conjunto de todas las categorías posibles (incluyendo "Sin diagnóstico")
            categorias = set()
            
            # Recopilar todas las categorías únicas (normalizadas)
            for caso in datos_medico + datos_ia:
                diagnostico = caso.get('diagnostico', '').strip()
                diagnostico_normalizado = self._normalizar_diagnostico(diagnostico)
                categorias.add(diagnostico_normalizado)
            
            # Convertir a lista ordenada
            categorias = sorted(list(categorias))
            
            if debug:
                print(f"🔍 Categorías encontradas: {categorias}")
            
            if len(categorias) == 0:
                return 1.0  # Sin categorías = perfecta concordancia
            
            # Crear matriz de confusión
            n_categorias = len(categorias)
            matriz_confusion = np.zeros((n_categorias, n_categorias))
            
            # Mapear diagnósticos a índices
            categoria_to_idx = {cat: idx for idx, cat in enumerate(categorias)}
            
            if debug:
                print(f"🔍 Mapeo de categorías: {categoria_to_idx}")
            
            # Llenar matriz de confusión
            n_total = len(datos_medico)
            
            for i in range(n_total):
                medico_diag = datos_medico[i].get('diagnostico', '').strip()
                ia_diag = datos_ia[i].get('diagnostico', '').strip()
                
                # Normalizar diagnósticos usando la función de normalización
                medico_diag = self._normalizar_diagnostico(medico_diag)
                ia_diag = self._normalizar_diagnostico(ia_diag)
                
                # Obtener índices
                medico_idx = categoria_to_idx.get(medico_diag, -1)
                ia_idx = categoria_to_idx.get(ia_diag, -1)
                
                if debug:
                    print(f"   Caso {i+1}: Médico='{medico_diag}' (idx={medico_idx}), IA='{ia_diag}' (idx={ia_idx})")
                
                # Contar en la matriz de confusión
                if medico_idx >= 0 and ia_idx >= 0:
                    matriz_confusion[medico_idx, ia_idx] += 1
            
            if debug:
                print(f"🔍 Matriz de confusión:\n{matriz_confusion}")
                print(f"🔍 Categorías: {categorias}")
            
            # Calcular métricas de Kappa
            if n_total == 0:
                return 1.0  # Sin datos = perfecta concordancia
            
            # Concordancia observada (Po) - diagonal de la matriz
            concordancia_observada = np.trace(matriz_confusion) / n_total
            
            # Concordancia esperada (Pe) - suma de productos marginales
            suma_filas = np.sum(matriz_confusion, axis=1)
            suma_columnas = np.sum(matriz_confusion, axis=0)
            concordancia_esperada = np.sum(suma_filas * suma_columnas) / (n_total ** 2)
            
            if debug:
                print(f"🔍 Concordancia observada (Po): {concordancia_observada:.4f}")
                print(f"🔍 Concordancia esperada (Pe): {concordancia_esperada:.4f}")
                print(f"🔍 Suma filas: {suma_filas}")
                print(f"🔍 Suma columnas: {suma_columnas}")
            
            # Calcular Kappa
            if concordancia_esperada >= 1.0:
                kappa = 1.0 if concordancia_observada >= 1.0 else 0.0
            else:
                kappa = (concordancia_observada - concordancia_esperada) / (1.0 - concordancia_esperada)
            
            if debug:
                print(f"🔍 Kappa calculado: {kappa:.4f}")
            
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
        
        # Calcular Kappa Cohen para diagnósticos (usando versión flexible)
        resultados['kappa_cohen'] = {
            'medico_vs_deepseek': self.calcular_kappa_cohen_flexible(datos['medico_sistema'], datos['deepseek']),
            'medico_vs_gemini': self.calcular_kappa_cohen_flexible(datos['medico_sistema'], datos['gemini']),
            'deepseek_vs_gemini': self.calcular_kappa_cohen_flexible(datos['deepseek'], datos['gemini'])
        }
        
        # Calcular métricas adicionales de concordancia médica
        resultados['concordancia_medica'] = {
            'medico_vs_deepseek': self.calcular_concordancia_medica(datos['medico_sistema'], datos['deepseek']),
            'medico_vs_gemini': self.calcular_concordancia_medica(datos['medico_sistema'], datos['gemini']),
            'deepseek_vs_gemini': self.calcular_concordancia_medica(datos['deepseek'], datos['gemini'])
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
        
        # Imprimir Concordancia Médica
        print("\n🔸 CONCORDANCIA MÉDICA")
        print("-" * 50)
        for comparacion, metricas in resultados['concordancia_medica'].items():
            print(f"\n📋 {comparacion}:")
            for metrica, valor in metricas.items():
                print(f"   {metrica}: {valor:.3f}")
        
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
