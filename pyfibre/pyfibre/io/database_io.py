import pandas as pd

from .utilities import check_file_name


def save_database(database, db_filename, file_type=None):

    db_filename = check_file_name(db_filename, extension='h5')
    db_filename = check_file_name(db_filename, extension='xls')

    if file_type is not None:
        db_filename = '_'.join([db_filename, file_type])

    database.to_hdf(f"{db_filename}.h5", key='df')
    database.to_excel(f"{db_filename}.xls")


def load_database(db_filename, file_type=None):

    db_filename = check_file_name(db_filename, extension='h5')
    db_filename = check_file_name(db_filename, extension='xls')

    if file_type is not None:
        db_filename = '_'.join([db_filename, file_type])

    database = pd.read_hdf(f"{db_filename}.h5", key='df')

    return database
