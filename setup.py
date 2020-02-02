import os
from setuptools import setup, find_packages


VERSION = '2.0.0'

#: Read description
with open('README.md', 'r') as readme:
    README_TEXT = readme.read()


def write_version_py():

    filename = os.path.join(
        os.path.abspath('.'),
        'pyfibre',
        'version.py')

    ver = f"__version__ = '{VERSION}'\n"
    with open(filename, 'w') as outfile:
        outfile.write(ver)


write_version_py()

with open('requirements.txt', 'r') as infile:
    REQUIREMENTS = infile.readlines()

setup(
    name='PyFibre',
    version=VERSION,
    author='Frank Longford',
    description='Open source image analysis toolkit for fibrous tissue',
    long_description=README_TEXT,
    packages=find_packages(),
    entry_points={
        'gui_scripts': ['PyFibre = pyfibre.cli.app:pyfibre',
                        'PyFibre_GUI = pyfibre.gui.app:pyfibre']
    },
    install_requires=REQUIREMENTS
)
