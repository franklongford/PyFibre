from abc import ABC, abstractmethod


class ABCPyFibreObject(ABC):
    """Abstract base class for an object representing an extracted feature
    in a multi-image. Serialization and de-serialization routines must be
    implemented with a concrete base subclass, along with a method that
    generates a pandas database containing metrics."""

    @classmethod
    @abstractmethod
    def from_json(cls, data):
        """Deserialises JSON data dictionary to return an instance
        of the class"""

    @abstractmethod
    def to_json(self):
        """Serialises instance into a dictionary able to be dumped as a
        JSON file"""

    @classmethod
    @abstractmethod
    def from_array(cls, array, **kwargs):
        """Deserialises numpy array to return an instance
        of the class"""

    @abstractmethod
    def to_array(self, **kwargs):
        """Serialises instance into a numpy array able to be dumped as a
        numpy binary file"""

    @abstractmethod
    def generate_database(self, *args, **kwargs):
        """Generates a Pandas database with all graph and segment
        metrics for assigned image"""
