import os


def parse_files(name=None, directory=None, key=None):

    input_files = []

    if key == '':
        key = None

    if name is not None:
        for file_name in name.split(','):
            if file_name.find('/') == -1:
                file_name = os.getcwd() + '/' + file_name
            input_files.append(file_name)

    if directory is not None:
        for folder in directory.split(','):
            for file_name in os.listdir(folder):
                input_files += [folder + '/' + file_name]

    if key is not None:
        removed_files = []
        for key in key.split(','):
            for file_name in input_files:
                if ((file_name.find(key) == -1) and
                        (file_name not in removed_files)):
                    removed_files.append(file_name)

        for file_name in removed_files:
            input_files.remove(file_name)

    return input_files


def parse_file_path(file_path):

    file_name = None
    directory = None

    if os.path.isfile(file_path):
        file_name = file_path
    elif os.path.isdir(file_path):
        directory = file_path

    return file_name, directory
