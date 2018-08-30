from __future__ import division, print_function, unicode_literals

import os
import sys

import sphinx_rtd_theme
from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath('..'))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath('_ext'))

extensions = [
#     'sphinxcontrib.spelling',
#     'sphinxcontrib.fulltoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
#     'sphinxcontrib.plantuml',
]
templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'
project =  u'Stacker'
copyright = '2017, Shahar Azulay, Ariel Hanemann'

version = '1.0'
release = '1.0'
exclude_patterns = ['_build']
default_role = 'obj'
# intersphinx_mapping = {
#     'python': ('http://python.readthedocs.io/en/latest/', None),
#     'django': ('http://django.readthedocs.io/en/1.9.x/', None),
#     'sphinx': ('http://sphinx.readthedocs.io/en/latest/', None),
# }

htmlhelp_basename = 'Stacker'
latex_documents = [
    (master_doc, 'Stacker.tex', u'Stacker Documentation',
     u'Shahar Azulay, Ariel Hanemann', 'manual'),
]
man_pages = [
    (master_doc, 'stacker', u'Stacker Documentation',
     [u'Shahar Azulay, Ariel Hanemann'], 1)
]

exclude_patterns = [
    # 'api' # needed for ``make gettext`` to not die.
]


html_logo = '_static/logo.jpeg'
html_favicon = '_static/logo.ico'
html_static_path = ['_static']

# Activate autosectionlabel plugin
autosectionlabel_prefix_document = True
