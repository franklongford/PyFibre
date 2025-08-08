import pandas as pd
from pathlib import Path

from pyfibre.io.utilities import check_file_name


def save_database(
    database: pd.DataFrame, db_filename: Path, file_type: str | None = None
) -> None:
    db_filename = check_file_name(db_filename, extension="h5")
    db_filename = check_file_name(db_filename, extension="xlsx")

    if file_type is not None:
        db_filename = "_".join([db_filename, file_type])

    database.to_hdf(f"{db_filename}.h5", key="df")
    database.to_excel(f"{db_filename}.xlsx")


def load_database(db_filename: Path, file_type: str | None = None) -> pd.DataFrame:
    db_filename = check_file_name(db_filename, extension="h5")
    db_filename = check_file_name(db_filename, extension="xlsx")

    if file_type is not None:
        db_filename = "_".join([db_filename, file_type])

    return pd.read_hdf(f"{db_filename}.h5", key="df")
