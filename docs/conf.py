"""Sphinx configuration for xmerge documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "xmerge"
copyright = "2026, Griffith-7"
author = "Griffith-7"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_member_order = "bysource"
