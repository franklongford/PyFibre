import sys, os, subprocess, multiprocessing

task = sys.argv[1]
program_name = sys.argv[2]
speed_test_name = "speed_test"
bin_dir = sys.argv[3]
if '--python' in sys.argv: python_command = sys.argv[sys.argv.index('--python') + 1]
else: python_command = 'python'
python_version = sys.version_info
current_dir = os.getcwd()

if task in ['install', 'install_mpi']:

	print("Checking python executable version\n")

	if python_version[0] < 3:
		print(f"Error: current python version = {python_version[0]}.{python_version[1]}.{python_version[2]}\n       version required >= 3.0\n")

		print("Checking python3 executable version\n")
		bashCommand = "python3 --version"
		try: process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		except:
			print("No python3 executable found, exiting installation\n\nYou need a Python 3 distribution to continue\n")
			sys.exit(1)

		output, error = process.communicate()
		python_version = output[:-1]

		if output[1] >= 3:	
			python_command += '3'
			print(f"{output[:-1]} detected, using python3 excecutable\n")
		else: 
			print("No python3 excecutable found, exiting installation\n")
			sys.exit(1)

	print(f"Creating {program_name} executable\n")

	if not os.path.exists(current_dir + '/bin'): os.mkdir(current_dir + '/bin')

	if task == 'install':
		with open(current_dir + '/bin/' + program_name, 'w') as outfile:
			outfile.write('#!/bin/bash\n\n')
			outfile.write(f'{python_command} {current_dir}/src/main.py "$@"')

	else:
		with open(current_dir + '/bin/' + program_name, 'w') as outfile:
			outfile.write('#!/bin/bash\n\n')
			outfile.write(f'{python_command} {current_dir}/src/main_mpi.py "$@"')

		bashCommand = f"chmod +x {current_dir + '/bin/' + speed_test_name}"
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output, error = process.communicate()

	bashCommand = f"chmod +x {current_dir + '/bin/' + program_name}"
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = process.communicate()

	print(f"Copying {program_name} executable to {bin_dir}\n")

	bashCommand = f"cp {current_dir + '/bin/'} {bin_dir}"
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output, error = process.communicate()
	if error:
		print('!' * 30 + ' ERROR ' + '!' * 30 + '\n') 
		print(f"{error}\nUnable to add ImageCol to {bin_dir}\n\nPlease manually create an alias to {current_dir}/{program_name}\n")
		print('!' * 67 + '\n')


if task in ['uninstall', 'uninstall_mpi']:

	bashCommand = f"rm {program_name}"
	try: 
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output, error = process.communicate()
	except: sys.exit(1)
	
	bashCommand = f"rm {bin_dir}/{program_name}"
	try: 
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		output, error = process.communicate()
	except: sys.exit(1)
