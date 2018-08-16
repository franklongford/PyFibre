PYTHON=python
PIP=pip

BIN=$(firstword $(subst :, " ", $(PATH)))
NAME=ImageCol
NAME_MPI=ImageCol_mpi


init: check install test


check:	
	@echo "Checking default binary folder path:"
	@echo ${BIN}
	@echo
	@echo "Checking default python version:"
	@$(PYTHON) --version || (echo "No python distribution detected"; exit 1)
	@echo
	@echo "Checking default pip version:"
	@$(PIP) --version || (echo "No pip distribution detected"; exit 1)


install:
	@echo
	@echo "Installing ${NAME}"
	@echo
	@$(PIP) install -r requirements.txt
	@$(PYTHON) make.py install $(NAME) $(BIN) || (echo "Installation failed"; exit 1)

install_mpi:
	@echo
	@echo "Installing ${NAME_MPI}"
	@echo
	@$(PIP) install -r requirements_mpi.txt
	@$(PYTHON) make.py install_mpi $(NAME_MPI) $(BIN) || (echo "Installation failed"; exit 1)

test:
	@echo
	@echo "Running unit tests"
	@echo
	@pytest tests/ -v -l


uninstall:
	@$(PYTHON) make.py uninstall $(NAME) $(BIN) || (echo "Uninstallation failed"; exit 1)
		

uninstall_mpi:
	@$(PYTHON) make.py uninstall_mpi $(NAME) $(BIN) || (echo "Uninstallation failed"; exit 1)

clean:
	@rm -f -r bin
	@rm -f -r tests/__pycache__
	@rm -f -r src/__pycache__
	@rm -f src/*.pyc
	@rm -f -r .cache/
	@rm -f  .DS_Store
	@rm -f -r .pytest_cache/
