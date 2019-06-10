import click
import os
from subprocess import check_call

DEFAULT_PYTHON_VERSION = "3.6"
PYTHON_VERSIONS = ["3.6"]
PYFIBRE_REPO = os.path.abspath('.')


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

@cli.command(name="install",
			 help='Creates the execution environment and binaries')
@python_version_option
def install(python_version):

	command = ["pip", "install"] + ADDITIONAL_CORE_DEPS
	check_call(command)

	env_name = get_env_name()
	edm_run(env_name, ['pip', 'install', '-e', '.'])


@cli.command(name="uninstall",
			 help='Removes the execution environment and binaries')
@python_version_option
def uninstall(python_version):

	command = ["pip", "uninstall"] + ADDITIONAL_CORE_DEPS
	check_call(command)

	env_name = get_env_name()
	edm_run(env_name, ['pip', 'uninstall', '-e', '.'])


@cli.command(name="test",
			 help='Runs unit tests for repository')
@python_version_option
def test(python_version):

	env_name = get_env_name()
	edm_run(env_name, ['python', '-m', 'unittest',
					   'discover', '*/test/*'])


if __name__ == "__main__":
	cli()