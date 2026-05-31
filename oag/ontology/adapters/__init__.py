"""Built-in ontology object data adapters."""

from .json_file import JsonFileAdapter
from .sqlite_table import SqliteTableAdapter

__all__ = ["JsonFileAdapter", "SqliteTableAdapter"]
