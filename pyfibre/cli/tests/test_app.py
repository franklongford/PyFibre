import os
import subprocess
from unittest import mock, TestCase
from contextlib import contextmanager

from pyfibre.tests import fixtures

@contextmanager
def cd(dir):
    cwd = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(cwd)


class TestCLIApp(TestCase):

    def setUp(self):
        self.name = 'file-shg.tif,/a/path/to/some/file-pl-shg.tif'
        self.directory = ''
        self.key = 'shg'

    def test_plain_invocation_mco(self):
        with cd(fixtures.directory):
            try:
                subprocess.check_output(["PyFibre", '--help'],
                                        stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                self.fail("PyFibre returned error at plain invocation.")
