from traits.api import Interface, Str


class IFileSet(Interface):
    """Small container class that represents a collection of related
    image files"""

    prefix = Str


class IFileParser(Interface):

    def get_file_sets(self, filenames):
        """From a given list of file names, returns a dictionary where each entry
        represents the files required to create an instance of a multi image.
        Each key will be passed on as the name of the multi image, used during
        further PyFibre operations. Each value could be passed in as the
        `filenames` argument to the class `create_image_stack` method.

        Returns
        -------
        file_sets: list of IFileSet
            List containing FileSet objects that hold a collection of files to
            be loaded in as a single image

        Examples
        --------
        For a given list of files and file parser:

        >>> file_list = ['/path/to/an/image',
        ...              '/path/to/another/image',
        ...              '/path/to/nothing']
        >>> file_parser = MyFileParser()

        If each "image" file path could be loaded in as a separate MultiImage,
        the return value of `collate_files` could be:

        >>> file_sets = file_parser.get_supported_file_sets(file_list)
        >>> print(file_sets.registry)
        ... {"a file name": ['/path/to/an/image'],
        ...  "another file name": ['/path/to/another/image']}

        Alternatively, if both "image" file paths were required to load
        a single MultiImage, then a return value could be:

        >>> file_sets = file_parser.get_supported_file_sets(file_list)
        >>> print(file_sets.registry)
        ... {"a file name": ['/path/to/an/image',
        ...                  '/path/to/another/image']}

        """
