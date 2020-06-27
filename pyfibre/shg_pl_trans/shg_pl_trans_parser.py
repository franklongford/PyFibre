import copy

from traits.api import Dict, Str

from pyfibre.core.base_file_parser import BaseFileParser, FileSet

from .utils import filter_input_files, get_files_prefixes


class SHGPLTransFileSet(FileSet):
    pass


class SHGPLTransParser(BaseFileParser):
    """Parser class for SHG files"""

    _file_set_cache = Dict(Str, SHGPLTransFileSet)

    _image_types = ('SHG-PL-Trans', 'SHG', 'PL-Trans')

    def _cache_file_sets(self, input_files, image_type):
        """Populate image_dictionary argument using prefixes and filenames
        of input_files list"""

        files, prefixes = get_files_prefixes(
            input_files, image_type)

        for filename, prefix in zip(files, prefixes):
            if prefix in self._file_set_cache:
                file_set = self._file_set_cache[prefix]
            else:
                file_set = SHGPLTransFileSet(prefix=prefix)
                self._file_set_cache[prefix] = file_set
            file_set.registry[image_type] = filename
            input_files.remove(filename)

    def get_file_sets(self, input_files):

        self._file_set_cache = {}
        input_files = filter_input_files(copy.copy(input_files))

        for image_type in self._image_types:
            self._cache_file_sets(input_files, image_type)

        return list(self._file_set_cache.values())
