
class DatabaseWriter():

    def __init__(self):
        pass

    def check_string(self, string, pos, sep, word):
        """Checks index 'pos' of 'string' seperated by 'sep' for substring 'word'
        If present, removes 'word' and returns amended string
        """

        if sep in string:
            temp_string = string.split(sep)
            if temp_string[pos] == word: temp_string.pop(pos)
            string = sep.join(temp_string)

        return string

    def check_file_name(self, file_name, file_type="", extension=""):
        """
        check_file_name(file_name, file_type="", extension="")

        Checks file_name for file_type or extension. If present, returns
        amended file_name without extension or file_type

        """

        file_name = self.check_string(file_name, -1, '.', extension)
        file_name = self.check_string(file_name, -1, '_', file_type)

        return file_name

    def write_database(self, database, db_filename, file_type=''):

        db_filename = self.check_file_name(db_filename, extension='pkl')
        db_filename = self.check_file_name(db_filename, extension='xls')

        database.to_pickle(db_filename + f"{file_type}.pkl")
        database.to_excel(db_filename + f"{file_type}.xls")
