"""领域目录加载器。

load_domain 读取 ontology.yaml，注册内置 source adapter，导入可选的
functions 模块，并把 YAML 中声明的函数定义绑定到 Python 实现上。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .adapters.json_file import JsonFileAdapter
from .adapters.sqlite_table import SqliteTableAdapter
from .registry import FunctionRegistry
from .schema import Ontology
from .store import Store


def load_domain(domain_dir: str | Path) -> tuple[Ontology, Store, FunctionRegistry]:
    domain_dir = Path(domain_dir).resolve()

    ontology = Ontology.load(domain_dir / "ontology.yaml")
    registry = FunctionRegistry()
    registry.register_adapter("json_file", JsonFileAdapter.factory(domain_dir))
    registry.register_adapter("sqlite_table", SqliteTableAdapter.factory(domain_dir))

    repository = Store(ontology, registry)
    repository.create_tables()

    func_pkg = _import_functions(domain_dir / "functions")

    data_dir = domain_dir / "data"
    data_files = getattr(func_pkg, "DATA_FILES", {})
    field_mappings = getattr(func_pkg, "FIELD_MAPPINGS", {})
    for type_name, filename in data_files.items():
        mapping = field_mappings.get(type_name)
        repository.load_json_file(type_name, data_dir / filename, mapping)

    func_pkg.register(registry, repository, ontology)

    return ontology, repository, registry


def _import_functions(functions_dir: Path):
    pkg_name = f"_domain_{functions_dir.parent.name}_functions"
    init_file = functions_dir / "__init__.py"

    spec = importlib.util.spec_from_file_location(
        pkg_name, init_file,
        submodule_search_locations=[str(functions_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)
    return module
