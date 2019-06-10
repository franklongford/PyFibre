import sys, os, subprocess, multiprocessing

task = sys.argv[1]
program_name = sys.argv[2]
speed_test_name = "speed_test"
bin_dir = sys.argv[3]
desktop_dir = sys.argv[4]
if '--python' in sys.argv: python_command = sys.argv[sys.argv.index('--python') + 1]
else: python_command = 'python'
python_version = sys.version_info
current_dir = os.getcwd()

if task in ['install', 'install_mpi']:

    print(f"Creating {program_name} executable\n")

    if not os.path.exists(current_dir + '/bin'): os.mkdir(current_dir + '/bin')

    if task == 'install':
        with open(current_dir + '/bin/' + program_name, 'w') as outfile:
            outfile.write('#!/bin/bash\n\n')
            outfile.write(f"{python_command} {current_dir}/pyfibre/main.py \"$@\"")

    else:
        with open(current_dir + '/bin/' + program_name, 'w') as outfile:
            outfile.write('#!/bin/bash\n\n')
            outfile.write(f"{python_command} {current_dir}/pyfibre/main_mpi.py \"$@\"")

        bashCommand = f"chmod +x {current_dir + '/bin/' + speed_test_name}"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

    with open(current_dir + '/bin/' + program_name + '_GUI', 'w') as outfile:
        outfile.write('#!/bin/bash\n\n')
        outfile.write(f"{python_command} {current_dir}/pyfibre/gui.py")

    bashCommand = f"chmod +x {current_dir + '/bin/' + program_name}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    bashCommand = f"chmod +x {current_dir + '/bin/' + program_name}_GUI"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    print(f"Copying {program_name} and {program_name}_GUI executables to {bin_dir}\n")

    bashCommand = f"cp {current_dir}/bin/{program_name} {bin_dir}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print('!' * 30 + ' ERROR ' + '!' * 30 + '\n')
        print \
            (f"{error}\nUnable to add {program_name} to {bin_dir}\n\nPlease manually create an alias to {current_dir}/{program_name}\n")
        print('!' * 67 + '\n')

    bashCommand = f"cp {current_dir}/bin/{program_name}_GUI {bin_dir}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print('!' * 30 + ' ERROR ' + '!' * 30 + '\n')
        print \
            (f"{error}\nUnable to add {program_name}_GUI to {bin_dir}\n\nPlease manually create an alias to {current_dir}/{program_name}\n")
        print('!' * 67 + '\n')


if task in ['uninstall', 'uninstall_mpi']:

    bashCommand = f"rm -r {current_dir}/bin/"
    try:
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
    except: sys.exit(1)

    bashCommand = f"rm {bin_dir}/{program_name}"
    try:
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
    except: sys.exit(1)

    bashCommand = f"rm {bin_dir}/{program_name}_GUI"
    try:
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
    except: sys.exit(1)
