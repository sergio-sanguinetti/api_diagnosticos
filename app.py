#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Archivo de entrada para Render
Importa la aplicación Flask desde analizador_ia.py
"""

import os
from analizador_ia import app

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
