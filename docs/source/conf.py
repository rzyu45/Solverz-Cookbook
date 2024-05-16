# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from datetime import datetime

project = 'Solverz Cookbook'
copyright = f'{datetime.now().year}, rzyu'
author = 'rzyu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx_math_dollar', 'sphinx.ext.mathjax', 'numpydoc',
              'sphinx_reredirects', 'sphinx_copybutton',
              'sphinx.ext.graphviz', 'sphinxcontrib.jquery',
              'matplotlib.sphinxext.plot_directive', 'myst_parser',
              'sphinx.ext.intersphinx', ]  # 'sphinx.ext.linkcode'

# To stop docstrings inheritance.
autodoc_inherit_docstrings = False

# Sphinx是一个文档生成器，可以将Markdown或reStructuredText等文本格式转化为HTML、PDF等格式的文档。而MathJax是一个用于渲染数学公式的JavaScript库，它可以帮助将LaTeX或MathML格式的数学公式渲染为高质量的矢量图形。
# 虽然Sphinx本身提供了一些对数学公式的支持，但其渲染效果不如MathJax优秀。因此，为了获得更好的数学公式渲染效果，使用MathJax插件可以帮助Sphinx在生成文档时自动渲染数学公式，从而提高文档的质量和可读性。

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

mathjax3_config = {
    "tex": {
        "inlineMath": [['\\(', '\\)']],
        "displayMath": [["\\[", "\\]"]],
        'packages': {'[+]': ['physics']}
    },
    'loader': {'load': ['[tex]/physics']},
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    "description": "A Collection of Solverz' Recipes",
    "github_user": "rzyu45",
    "github_repo": "Solverz-Cookbook",
    "fixed_sidebar": True
}
