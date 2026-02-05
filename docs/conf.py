import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'PyPhysicist'
author = 'Samyar Noruzi'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

exclude_patterns = ['_build']

html_theme = 'alabaster'
