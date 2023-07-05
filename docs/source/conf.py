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

project = "AiTLAS"
copyright = "2023, Bias Variance Labs"
author = "Bias Variance Labs"
release = "1.0.0"

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
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

#

# Include typehints in descriptions
autodoc_typehints = "description"

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

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "_media/AiTALS_vertical_gradient.png"
html_theme = "sphinx_book_theme"
html_static_path = ["_static", "_media"]
