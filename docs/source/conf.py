# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

import shutil
import sys
from collections import defaultdict
from typing import Any, Dict, Optional

import sphinx.ext.autodoc

sys.path.insert(0, os.path.abspath("../../aitlas"))
print(sys.executable)

project = "AiTLAS : Artificial Intelligence Toolbox for Earth Observation"
copyright = "2023, Bias Variance Labs"
author = "Bias Variance Labs"
#release = "1.0.0"
doc_title="AiTLAS documentation"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The release is read from __init__ file and version is shortened release string.

with open(os.path.join(os.path.dirname(__file__), "../../setup.py")) as setup_file:
    for line in setup_file:
        if "version=" in line:
            release = line.split("=")[1].strip('", \n').strip("'")
            version = release.rsplit(".", 1)[0]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_mdinclude",
    'sphinx.ext.imgconverter',
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

#

# Include typehints in descriptions
autodoc_typehints = "description"
autodoc_mock_imports = ['gdal','tensorflow','osr','SpaceNet6Dataset']
#nitpicky = True

# Both the class’ and the __init__ method’s docstring are concatenated and inserted.
autoclass_content = "both"

# Content is in the same order as in module
autodoc_member_order = "bysource"


source_suffix = [".rst", ".md"]
master_doc = "index"
autoclass_content = "both"
add_module_names = True

napoleon_google_docstring = True
napoleon_use_param = True
# napoleon_use_ivar = True

nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]



# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "_media/AiTALS_vertical_gradient.png"
html_theme = "sphinx_book_theme"
html_static_path = ["_static", "_media"]

htmlhelp_basename = 'mainDoc'






# -- Options for LaTeX output ------------------------------------------------
latex_engine = 'pdflatex'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    'papersize': 'a4paper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    'fncychap': '\\usepackage[Sonny]{fncychap}',
    'fontpkg': '\\usepackage{amsmath, amsfonts, amssymb, amsthm, plex-sans, plex-serif, plex-mono, xcolor}',
    'printindex': r'\def\twocolumn[#1]{#1}\footnotesize\raggedright\printindex',
     # Additional stuff for the LaTeX preamble.
    #

   
     'preamble': r'''


    \usepackage{datetime}
    \newdateformat{MonthYearFormat}{%
    \monthname[\THEMONTH], \THEYEAR}
    ''',

    'maketitle': r'''
   
     
        \begin{titlepage}
            \centering

           \begin{figure}[!h]
            \centering
                \includegraphics[width=0.3\linewidth]{AiTALS_vertical_gradient.png}
            \end{figure}

            \vspace*{40mm} %%% * is used to give space from top
            {\sffamily \Huge \textbf{AiTLAS}}\\ 
            \vspace*{5mm}
            {\sffamily \Large Artificial Intelligence Toolbox for Earth Observation}\\
         

            \vspace{0mm}
 

            \vspace{40mm}
            {\sffamily \Large \textbf{Documentation}}\\
           
            \vspace{30mm}
            {\sffamily Bias Variance Labs\\}
             \url{www.bvlabs.ai}\\

            \vspace*{10mm}
            {\sffamily \small  \MonthYearFormat\today}

         \end{titlepage}
         
         {\sffamily \small \tableofcontents
         \clearpage}

     ''',
    # # Latex figure (float) alignment
    
    # # 'figure_align': 'htbp',
    'sphinxsetup': \
        'TitleColor={rgb}{0,0,0}, \
         HeaderFamily=\\sffamily\\bfseries, \
         InnerLinkColor={rgb}{0.208,0.374,0.486},',
  }
#latex_engine = 'xelatex'
latex_show_urls = 'footnote'
    # latex_elements = {
    # 'fontpkg': r'''
    #     \setmainfont{DejaVu Serif}
    #     \setsansfont{DejaVu Sans}
    #     \setmonofont{DejaVu Sans Mono}
    # ''',
latex_logo = '_media/AiTALS_vertical_gradient.png'


#latex_documents = [(master_doc, 'aitlas.tex', doc_title, 'Bias Variance Labs', 'manual','toctree_only=False')]

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
# latex_documents = [
#     ("working", "aitlas_working.tex", doc_title, author, "doc")
#     ("examples", "aitlas_examples.tex", doc_title, author, "api_doc")
#     ("modules", "aitlas_modules.tex", doc_title, author, "api_doc"),
# ]