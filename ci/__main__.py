import click
import os
from subprocess import check_call

DEFAULT_PYTHON_VERSION = "3.6"
PYTHON_VERSIONS = ["3.6"]
PYFIBRE_REPO = os.path.abspath('.')
PIP_DEVS = []


with open(f"{PYFIBRE_REPO}/requirements.txt", 'r') as infile:
    ADDITIONAL_CORE_DEPS = infile.readlines()


def get_env_name():
    return "PyFibre"


def edm_run(env_name, command, cwd=None):
    check_call(['edm', 'run', '-e', env_name, '--']+command, cwd=cwd)


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

@cli.command(name="build-env", help="Creates the execution environment")
@python_version_option
def build_env(python_version):
    env_name = get_env_name()
    check_call([
        "edm", "envs", "remove", "--purge", "--force",
        "--yes", env_name]
    )
    check_call(
        ["edm", "envs", "create", "--version",
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

    command = ["pip", "install"] + ADDITIONAL_CORE_DEPS
    check_call(command)

    env_name = get_env_name()
    edm_run(env_name, ['pip', 'install', '-e', '.'])


@cli.command(name="uninstall",
    help='Removes the execution environment and binaries'
)
def uninstall():

    command = ["pip", "uninstall"] + ADDITIONAL_CORE_DEPS
    check_call(command)

    env_name = get_env_name()
    edm_run(env_name, ['pip', 'uninstall', '-e', '.'])


@cli.command(help="Run flake")
def flake8():

    env_name = get_env_name()
    edm_run(env_name, ["flake8", "."])


@cli.command(help="Run the tests")
def test():

    env_name = get_env_name()

    edm_run(env_name, ["python", "-m", "unittest", "discover", "-v"])


if __name__ == "__main__":
    cli()