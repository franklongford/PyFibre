import click
import os
from subprocess import check_call

DEFAULT_PYTHON_VERSION = "3.6"
PYTHON_VERSIONS = ["3.6"]
PYFIBRE_REPO = os.path.abspath('.')

EDM_CORE_DEPS = [
    'Click==7.0-1',
    "chaco==4.8.0-2",
    "enable==4.8.1-1",
    "envisage==4.9.0-1",
    'pytables==3.5.1-1',
    'traits==5.2.0-1',
    'traitsui==6.1.3-4',
    "pyface==6.1.2-4",
    "pygments==2.2.0-1",
    "pyqt>=4.11.4-7",
    "qt>=4.8.7-10",
    "sip>=4.18.1-1",
    "pyzmq==16.0.0-7",
    "swig==3.0.11-2",
    "traits_futures==0.1.0-16"]

EDM_DEV_DEPS = ["flake8==3.7.7-1",
                "mock==2.0.0-3"]

DOCS_DEPS = []


def remove_dot(python_version):
    return "".join(python_version.split("."))


def get_env_name(python_version):
    return f"PyFibre-py{remove_dot(python_version)}"


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


@cli.command(
    name="build-env",
    help="Creates the edm execution environment"
)
@python_version_option
def build_env(python_version):
    env_name = get_env_name(python_version)

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
        "--yes"] + EDM_CORE_DEPS + EDM_DEV_DEPS + DOCS_DEPS
    )


@cli.command(
    name="install",
    help='Creates the execution binary inside the PyFibre environment'
)
@python_version_option
def install(python_version):

    env_name = get_env_name(python_version)

    print('Installing PyFibre to edm environment')
    edm_run(env_name, ['pip', 'install', '-e', '.'])


@cli.command(help="Run flake (dev)")
@python_version_option
def flake8(python_version):

    env_name = get_env_name(python_version)
    edm_run(env_name, ["flake8", "."])


@cli.command(help="Run the unit tests")
@python_version_option
def test(python_version):

    env_name = get_env_name(python_version)
    edm_run(env_name,
            ["python", "-m", "unittest", "discover", "-v"])


if __name__ == "__main__":
    cli()
