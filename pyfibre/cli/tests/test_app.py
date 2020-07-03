import os
import subprocess
from unittest import TestCase
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

    def test_plain_invocation_mco(self):
        with cd(fixtures.directory):
            try:
                subprocess.check_output(["PyFibre", '--help'],
                                        stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                self.fail("PyFibre returned error at plain invocation.")
