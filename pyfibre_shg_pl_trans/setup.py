import os
from setuptools import setup, find_packages


VERSION = '0.0.1'


def write_version_py():

    filename = os.path.join(
        os.path.abspath('.'),
        'pyfibre_shg_pl_trans',
        'version.py')

    ver = f"__version__ = '{VERSION}'\n"
    with open(filename, 'w') as outfile:
        outfile.write(ver)


write_version_py()

setup(
    name='PyFibre SHG-PL-Trans plugin',
    version=VERSION,
    author='Frank Longford',
    description='Open source image analysis toolkit for fibrous tissue',
    packages=find_packages(),
    entry_points={
        "pyfibre.plugins": [
            "pyfibre_shg_pl_trans = "
            "pyfibre_shg_pl_trans.shg_pl_trans_plugin"
            ":SHGPLTransPlugin"
        ]
    }
)
