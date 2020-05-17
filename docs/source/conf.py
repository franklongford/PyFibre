# -*- coding: utf-8 -*-

import sphinx.environment
from docutils.utils import get_source_line
import sys
import os

from pyfibre.version import __version__ as RELEASE


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )


MOCK_MODULES = []


def _warn_node(self, msg, node, **kwargs):
    if not msg.startswith('nonlocal image URI found:'):
        self._warnfunc(msg, '%s:%s' % get_source_line(node), **kwargs)


sphinx.environment.BuildEnvironment.warn_node = _warn_node


def mock_modules():
    import sys

    from unittest.mock import MagicMock

    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
            return Mock()

        def __call__(self, *args, **kwards):
            return Mock()

    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
    print('mocking {}'.format(MOCK_MODULES))


mock_modules()

extensions = [
    'sphinxcontrib.apidoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'PyFibre'
copyright = u'2018, PyFibre Project'
version = ".".join(RELEASE.split(".")[0:3])
release = RELEASE
pygments_style = 'sphinx'
html_theme = 'classic'
html_static_path = ['_static']
html_logo = '_static/icon.ico'
htmlhelp_basename = 'PyFibredoc'
intersphinx_mapping = {'http://docs.python.org/': None}
apidoc_module_dir = '../../pyfibre'
apidoc_output_dir = 'api'
apidoc_excluded_paths = ['*tests*', '*cli', '*gui*']
apidoc_separate_modules = True
