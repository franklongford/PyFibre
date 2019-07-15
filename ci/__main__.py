import click
import os
from subprocess import check_call

DEFAULT_PYTHON_VERSION = "3.6"
PYTHON_VERSIONS = ["3.6"]
PYFIBRE_REPO = os.path.abspath('.')

CORE_DEPS = ['Click==7.0-1',
             "envisage==4.7.2-1",
             'pytables==3.5.1-1',
             'traits==5.1.1-1',
             'traitsui==6.1.1-1',
             "pyface==6.1.0-2",
             "pygments==2.2.0-1",
             "pyqt==4.11.4-7",
             "qt==4.8.7-10",
             "sip==4.17-4",
             "chaco==4.7.2-3",
             "pyzmq==16.0.0-7"
             ]
DEV_DEPS = ["flake8==3.7.7-1",
            "mock==2.0.0-3"]
DOCS_DEPS = []
PIP_DEPS = []


def get_env_name():
    return "PyFibre"


def edm_run(env_name, command, cwd=None):
    check_call(
        ['edm', 'run', '-e', env_name, '--'] + command,
        cwd=cwd
    )


@click.group()
def cli():
    pass

python_version_option = click.option(
    '--python-version',
    default=DEFAULT_PYTHON_VERSION,
    type=click.Choice(PYTHON_VERSIONS),
    show_default=True,
    help="Python version for environment"
)

@cli.command(name="build-env",
             help="Creates the edm execution environment")
@python_version_option
def build_env(python_version):
    env_name = get_env_name()
    check_call([
        "edm", "env", "remove", "--purge", "--force",
        "--yes", env_name]
    )
    check_call(
        ["edm", "env", "create", "--version",
         python_version, env_name]
    )

    check_call([
        "edm", "install", "-e", env_name,
        "--yes"] + CORE_DEPS + DEV_DEPS + DOCS_DEPS
    )
    if len(PIP_DEPS):
        check_call([
            "edm", "run", "-e", env_name, "--",
            "pip", "install"] + PIP_DEPS
        )


@cli.command(name="install",
    help='Creates the execution binary inside the PyFibre environment'
)
def install():

    env_name = get_env_name()
    edm_run(env_name, ['pip', 'install', '-e', '.'])


@cli.command(help="Run flake")
def flake8():

    env_name = get_env_name()
    edm_run(env_name, ["flake8", "."])


@cli.command(help="Run the tests")
def test():

    env_name = get_env_name()
    edm_run(env_name,
            ["python", "-m", "unittest", "discover", "-v"])


if __name__ == "__main__":
    cli()