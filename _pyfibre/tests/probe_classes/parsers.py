from pyfibre.core.base_file_parser import BaseFileParser, FileSet
from pyfibre.tests.fixtures import test_image_path


class ProbeFileSet(FileSet):

    prefix = '/path/to/some/file'

    registry = {'Probe': test_image_path}


class ProbeParser(BaseFileParser):

    def get_file_sets(self, filenames):
        return [
            ProbeFileSet()
            for filename in filenames
            if filename.endswith('.tif')
        ]
