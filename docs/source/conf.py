#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# torchtext documentation build configuration file, created by
# sphinx-quickstart on Thu Nov 16 01:05:05 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import torchtext
import pytorch_sphinx_theme
import os
import re

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
]

#  Implementation from https://github.com/pytorch/audio/blob/main/docs/source/conf.py


def _get_var(var, default=False):
    if var not in os.environ:
        return default

    val = os.environ.get(var, "0")
    trues = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]
    falses = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]
    if val in trues:
        return True
    if val not in falses:
        print(
            f" --- WARNING: Unexpected environment variable value `{var}={val}`. " f"Expected one of {trues + falses}"
        )
    return False

# Implementation from https://github.com/pytorch/audio/blob/main/docs/source/conf.py


def _get_pattern():
    pattern = os.getenv("GALLERY_PATTERN")
    # If BUILD_GALLERY is falsy -> no build
    # If BUILD_GALLERY is truey -> build
    # If BUILD_GALLERY is undefined
    #    If GALLERY_PATTERN is defined     -> build
    #    If GALLERY_PATTERN is not defined -> not build
    if not _get_var("BUILD_GALLERY", default=False if pattern is None else True):
        if pattern is not None:
            print(
                ' --- WARNING: "GALLERY_PATTERN" is provided, but "BUILD_GALLERY" value is falsy. '
                "Sphinx galleries are not built. To build galleries, set `BUILD_GALLERY=1`."
            )
        return {
            "ignore_pattern": r"\.py",
        }

    ret = {"filename_pattern": "tutorial.py"}
    if os.getenv("GALLERY_PATTERN"):
        # See https://github.com/pytorch/tutorials/blob/cbf2238df0e78d84c15bd94288966d2f4b2e83ae/conf.py#L75-L83
        ret["ignore_pattern"] = r"/(?!" + re.escape(os.getenv("GALLERY_PATTERN")) + r")[^/]+$"
    return ret


sphinx_gallery_conf = {
    "examples_dirs": [
        "../../examples/tutorials",
    ],
    "gallery_dirs": [
        "tutorials",
    ],
    **_get_pattern(),
    "backreferences_dir": "gen_modules/backreferences",
    "first_notebook_cell": None,
    "doc_module": ("torchtext",),
}


napoleon_use_ivar = True
napoleon_numpy_docstring = False
napoleon_google_docstring = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'torchtext'
copyright = '2017, Torch Contributors'
author = 'Torch Contributors'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = torchtext.__version__
# The full version, including alpha/beta/rc tags.
release = torchtext.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["experimental_*"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'pytorch_project': 'docs',
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
    'analytics_id': 'UA-117752657-2',
}

html_logo = '_static/img/pytorch-logo-dark.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'torch': ('http://pytorch.org/docs/0.3.0/', None)
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'PyTorchdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'pytorch.tex', 'torchtext Documentation',
     'Torch Contributors', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'torchtext', 'torchtext Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'torchtext', 'torchtext Documentation',
     author, 'torchtext', 'One line description of project.',
     'Miscellaneous'),
]


# -- A patch that prevents Sphinx from cross-referencing ivar tags -------
# See http://stackoverflow.com/a/41184353/3343043

from docutils import nodes
from sphinx.util.docfields import TypedField
from sphinx import addnodes


def patched_make_field(self, types, domain, items, **kw):
    # `kw` catches `env=None` needed for newer sphinx while maintaining
    #  backwards compatibility when passed along further down!

    # type: (List, unicode, Tuple) -> nodes.field
    def handle_item(fieldarg, content):
        par = nodes.paragraph()
        par += addnodes.literal_strong('', fieldarg)  # Patch: this line added
        # par.extend(self.make_xrefs(self.rolename, domain, fieldarg,
        #                           addnodes.literal_strong))
        if fieldarg in types:
            par += nodes.Text(' (')
            # NOTE: using .pop() here to prevent a single type node to be
            # inserted twice into the doctree, which leads to
            # inconsistencies later when references are resolved
            fieldtype = types.pop(fieldarg)
            if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
                typename = u''.join(n.astext() for n in fieldtype)
                typename = typename.replace('int', 'python:int')
                typename = typename.replace('long', 'python:long')
                typename = typename.replace('float', 'python:float')
                typename = typename.replace('type', 'python:type')
                par.extend(self.make_xrefs(self.typerolename, domain, typename,
                                           addnodes.literal_emphasis, **kw))
            else:
                par += fieldtype
            par += nodes.Text(')')
        par += nodes.Text(' -- ')
        par += content
        return par

    fieldname = nodes.field_name('', self.label)
    if len(items) == 1 and self.can_collapse:
        fieldarg, content = items[0]
        bodynode = handle_item(fieldarg, content)
    else:
        bodynode = self.list_type()
        for fieldarg, content in items:
            bodynode += nodes.list_item('', handle_item(fieldarg, content))
    fieldbody = nodes.field_body('', bodynode)
    return nodes.field('', fieldname, fieldbody)


TypedField.make_field = patched_make_field
