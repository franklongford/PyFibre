import pandas as pd


def check_string(string, pos, sep, word):
    """Checks index 'pos' of 'string' seperated by 'sep' for substring 'word'
    If present, removes 'word' and returns amended string
    """

    if sep in string:
        temp_string = string.split(sep)
        if temp_string[pos] == word: temp_string.pop(pos)
        string = sep.join(temp_string)

    return string


def check_file_name(file_name, file_type="", extension=""):
    """
    check_file_name(file_name, file_type="", extension="")

    Checks file_name for file_type or extension. If present, returns
    amended file_name without extension or file_type

    """

    file_name = check_string(file_name, -1, '.', extension)
    file_name = check_string(file_name, -1, '_', file_type)

    return file_name


def save_database(database, db_filename, file_type=''):

    db_filename = check_file_name(db_filename, extension='h5')
    db_filename = check_file_name(db_filename, extension='xls')

    database.to_hdf(f"{db_filename}_{file_type}.h5", key='df')
    database.to_excel(f"{db_filename}_{file_type}.xls")


def load_database(db_filename, file_type=''):

    database = pd.read_hdf(f"{db_filename}_{file_type}.h5", key='df')

    return database