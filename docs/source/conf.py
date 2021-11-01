# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys

SYNCTOOLBOX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
assert os.path.exists(os.path.join(SYNCTOOLBOX_DIR, 'synctoolbox'))
sys.path.insert(0, SYNCTOOLBOX_DIR)

# -- Project information -----------------------------------------------------

project = 'SyncToolbox'
copyright = '2021, Meinard Müller, Yigitcan Özer, Michael Krause, Thomas Prätzlich and Jonathan Driedger'
author = 'Meinard Müller, Yigitcan Özer, Michael Krause, Thomas Prätzlich and Jonathan Driedger'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',   # documentation based on docstrings
    'sphinx.ext.napoleon',  # for having google/numpy style docstrings
    'sphinx.ext.viewcode',  # link source code
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks'
]
autodoc_preserve_defaults = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
import sphinx_rtd_theme  # noqa

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_use_index = True
html_use_modindex = True

html_logo = os.path.join(html_static_path[0], 'logo_synctoolbox.png')

html_theme_options = {'logo_only': True}

napoleon_custom_sections = [('Returns', 'params_style'), ('Parameters', 'params_style')]


extlinks = {'fmpbook': ('https://www.audiolabs-erlangen.de/fau/professor/mueller/bookFMP', 'FMP'),
            'fmpnotebook': ('https://www.audiolabs-erlangen.de/resources/MIR/FMP/%s.html', '%s.ipynb')}


def link_notebook(app, what, name, obj, options, lines):
    for i, line in enumerate(lines):
        if 'Notebook:' in line:
            match = re.search('Notebook: (.*?)\.ipynb', line)
            if match:
                link = match.group(1)
                lines[i] = lines[i].replace(f'{link}.ipynb', f':fmpnotebook:`{link}`')


def remove_module_docstring(app, what, name, obj, options, lines):
    if what == 'module':
        del lines[:]


def setup(app):
    app.connect('autodoc-process-docstring', link_notebook)
    app.connect('autodoc-process-docstring', remove_module_docstring)
