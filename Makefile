PYTHON=python
PIP=pip

python_version_full := $(wordlist 2,4,$(subst ., ,$(shell $(PYTHON) --version 2>&1)))
python_version_major := $(word 1,${python_version_full})
python_version_minor := $(word 2,${python_version_full})
python_version_patch := $(word 3,${python_version_full})

BIN=$(firstword $(subst :, " ", $(PATH)))
NAME=PyFibre
DESKTOP=$(HOME)/Desktop/
CURRENT_DIR = $(shell pwd)


init: check install #test


check:	
	@echo "Checking default binary folder path:"
	@echo ${BIN}
	@echo
	@echo "Checking default python version:"
	@$(PYTHON) --version || (echo "No python distribution detected"; exit 1)
	@echo
	@( [ $(python_version_major) -ge 3 ] && [ $(python_version_minor) -ge 6 ] ) || (echo "Python distribution >= 3.6 required"; exit 1)
	@echo
	@echo "Checking default pip version:"
	@$(PIP) --version || (echo "No pip distribution detected"; exit 1)


install:
	@echo
	@echo "Installing ${NAME}"
	@echo
	@$(PIP) install -r requirements.txt
	@$(PYTHON) make.py install $(NAME) $(BIN) $(DESKTOP) || (echo "Installation failed"; exit 1)


test:
	@echo
	@echo "Running unit tests"
	@echo
	@python -m unittest tests/test.py


uninstall:
	@$(PYTHON) make.py uninstall $(NAME) $(BIN) $(DESKTOP) || (echo "Uninstallation failed"; exit 1)
		

clean:
	@rm -f -r bin
	@rm -f -r pyfibre/tests/stubs/data/
	@rm -f -r pyfibre/tests/stubs/fig/
	@rm -f -r .cache/
	@rm -f  .DS_Store
	@rm -f -r .pytest_cache/
	@rm -f -r *.h5
	@rm -f -r *.xls
	@rm -f -r *.log
