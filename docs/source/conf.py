import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust path to your project root
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pysces'
copyright = '2026, Owen K Hughes'
author = 'Owen K Hughes'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["numpydoc",
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon']


# Autosummary
autosummary_generate = True  # Generate autosummary pages
autosummary_generate_overwrite = False

# Napoleon settings (for NumPy/Google docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


def setup(app):
    """Generate API docs automatically."""
    import subprocess
    subprocess.run(['python', 'generate_docs.py'], cwd='.')
