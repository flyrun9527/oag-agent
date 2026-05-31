"""Ontology object repository and data-source adapters.

The repository is the runtime-facing data boundary for ontology objects. It
routes each ontology object to its declared source adapter or resolver.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import Parameter, signature
from typing import Any, Protocol

from .registry import FunctionRegistry
from .schema import ObjectSourceDef, Ontology


class ObjectAdapter(Protocol):
    def query(self, object_type: str, filters: dict[str, Any] | None = None,
              limit: int | None = None, order_by: str | None = None,
              offset: int | None = None) -> list[dict]: ...
    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int: ...
    def query_by_id(self, object_type: str, id_value: Any) -> dict | None: ...
    def search_text(self, keyword: str, object_types: list[str] | None = None,
                    limit: int = 20) -> list[dict]: ...
    def insert_record(self, object_type: str, data: dict) -> dict: ...
    def update_record(self, object_type: str, id_value: Any, data: dict) -> dict: ...
    def delete_record(self, object_type: str, id_value: Any) -> dict: ...
    def table_count(self, object_type: str) -> int: ...


class ResolverAdapter:
    """Adapter for developer-defined object resolvers."""

    def __init__(self, ontology: Ontology, source: ObjectSourceDef,
                 resolver: Any):
        self.ontology = ontology
        self.source = source
        self.resolver = resolver

    def query(self, object_type: str, filters: dict[str, Any] | None = None,
              limit: int | None = None, order_by: str | None = None,
              offset: int | None = None) -> list[dict]:
        rows = self._call(
            "query",
            object_type=object_type,
            filters=filters,
            limit=limit,
            order_by=order_by,
            offset=offset,
        )
        if rows is None:
            return []
        if isinstance(rows, dict):
            return [rows]
        return [dict(row) for row in rows]

    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int:
        if self._supports("count"):
            return int(self._call("count", object_type=object_type, filters=filters))
        return len(self.query(object_type, filters))

    def query_by_id(self, object_type: str, id_value: Any) -> dict | None:
        if self._supports("query_by_id"):
            row = self._call("query_by_id", object_type=object_type, id_value=id_value)
            return dict(row) if row else None

        id_col = self.source.id_field or self.ontology.get_id_column(object_type)
        if not id_col:
            return None
        rows = self.query(object_type, {id_col: id_value}, limit=1)
        return rows[0] if rows else None

    def search_text(self, keyword: str, object_types: list[str] | None = None,
                    limit: int = 20) -> list[dict]:
        if self._supports("search_text"):
            rows = self._call(
                "search_text",
                keyword=keyword,
                object_types=object_types,
                limit=limit,
            )
            return [dict(row) for row in rows or []]
        return _search_rows(
            ontology=self.ontology,
            keyword=keyword,
            object_types=object_types,
            limit=limit,
            query_fn=self.query,
        )

    def insert_record(self, object_type: str, data: dict) -> dict:
        if self._supports("insert_record"):
            return dict(self._call("insert_record", object_type=object_type, data=data))
        raise ValueError(f"{object_type} 的 resolver 不支持 create")

    def update_record(self, object_type: str, id_value: Any, data: dict) -> dict:
        if self._supports("update_record"):
            return dict(self._call(
                "update_record",
                object_type=object_type,
                id_value=id_value,
                data=data,
            ))
        raise ValueError(f"{object_type} 的 resolver 不支持 update")

    def delete_record(self, object_type: str, id_value: Any) -> dict:
        if self._supports("delete_record"):
            return dict(self._call(
                "delete_record",
                object_type=object_type,
                id_value=id_value,
            ))
        raise ValueError(f"{object_type} 的 resolver 不支持 delete")

    def table_count(self, object_type: str) -> int:
        return self.count(object_type)

    def _supports(self, operation: str) -> bool:
        return hasattr(self.resolver, operation) or (
            operation == "query" and callable(self.resolver)
        )

    def _call(self, operation: str, **kwargs) -> Any:
        if hasattr(self.resolver, operation):
            fn = getattr(self.resolver, operation)
            return _call_resolver(fn, **kwargs)
        if operation == "query" and callable(self.resolver):
            return _call_resolver(self.resolver, **kwargs)
        raise ValueError(f"resolver {self.source.resolver} 不支持 {operation}")


class ObjectRepository:
    """Unified access point for ontology object data."""

    def __init__(self, ontology: Ontology, registry: FunctionRegistry):
        self.ontology = ontology
        self.registry = registry
        self._adapters: dict[str, ObjectAdapter] = {}

    def adapter_for(self, object_type: str) -> ObjectAdapter:
        if object_type in self._adapters:
            return self._adapters[object_type]

        obj_def = self.ontology.objects.get(object_type)
        if not obj_def:
            raise ValueError(f"未知对象类型: {object_type}")

        source = obj_def.source or ObjectSourceDef()
        source_type = source.type or ""
        if source_type == "resolver":
            if not source.resolver:
                raise ValueError(f"{object_type} 的 source.resolver 不能为空")
            resolver = self.registry.get_resolver(source.resolver)
            if resolver is None:
                raise ValueError(f"未注册对象 resolver: {source.resolver}")
            adapter = ResolverAdapter(self.ontology, source, resolver)
        else:
            if not source_type:
                raise ValueError(f"{object_type} 未声明 source.type")
            factory = self.registry.get_adapter_factory(source_type)
            if factory is None:
                raise ValueError(f"{object_type} 不支持的数据源类型: {source_type}")
            adapter = factory(
                ontology=self.ontology,
                registry=self.registry,
                object_type=object_type,
                source=source,
            )

        self._adapters[object_type] = adapter
        return adapter

    def query(self, object_type: str, filters: dict[str, Any] | None = None,
              limit: int | None = None, order_by: str | None = None,
              offset: int | None = None) -> list[dict]:
        return self.adapter_for(object_type).query(
            object_type, filters, limit, order_by, offset,
        )

    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int:
        return self.adapter_for(object_type).count(object_type, filters)

    def query_by_id(self, object_type: str, id_value: Any) -> dict | None:
        return self.adapter_for(object_type).query_by_id(object_type, id_value)

    def query_links(self, source_type: str, source_id: Any,
                    link_name: str) -> list[dict]:
        link = self.ontology.links.get(link_name)
        if not link:
            return []
        if link.source != source_type:
            return []

        source_row = self.query_by_id(source_type, source_id)
        if not source_row:
            return []

        source_key = link.join.get("source_key")
        target_key = link.join.get("target_key")
        if not source_key or not target_key:
            return []

        source_key_value = source_row.get(source_key)
        if source_key_value is None:
            return []
        return self.query(link.target, {target_key: source_key_value})

    def search_text(self, keyword: str, object_types: list[str] | None = None,
                    limit: int = 20) -> list[dict]:
        if not keyword:
            return []

        types_to_search = object_types or list(self.ontology.objects.keys())
        results: list[dict] = []
        for object_type in types_to_search:
            rows = self.adapter_for(object_type).search_text(
                keyword,
                [object_type],
                limit - len(results),
            )
            results.extend(rows)
            if len(results) >= limit:
                break
        return results[:limit]

    def insert_record(self, object_type: str, data: dict) -> dict:
        return self.adapter_for(object_type).insert_record(object_type, data)

    def update_record(self, object_type: str, id_value: Any, data: dict) -> dict:
        return self.adapter_for(object_type).update_record(object_type, id_value, data)

    def delete_record(self, object_type: str, id_value: Any) -> dict:
        return self.adapter_for(object_type).delete_record(object_type, id_value)

    def table_count(self, object_type: str) -> int:
        return self.adapter_for(object_type).table_count(object_type)

    def close(self):
        for adapter in self._adapters.values():
            close = getattr(adapter, "close", None)
            if callable(close):
                close()


def _call_resolver(fn: Callable[..., Any], **kwargs) -> Any:
    sig = signature(fn)
    params = sig.parameters
    if any(p.kind == Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)

    if any(p.kind in (Parameter.VAR_POSITIONAL, Parameter.POSITIONAL_ONLY)
           for p in params.values()):
        return fn(
            kwargs["object_type"],
            kwargs.get("filters"),
            kwargs.get("limit"),
            kwargs.get("order_by"),
            kwargs.get("offset"),
        )

    supported = {
        name: value
        for name, value in kwargs.items()
        if name in params
    }
    return fn(**supported)


def _search_rows(ontology: Ontology, keyword: str,
                 object_types: list[str] | None, limit: int,
                 query_fn: Callable[..., list[dict]]) -> list[dict]:
    if not keyword:
        return []

    types_to_search = object_types or list(ontology.objects.keys())
    results: list[dict] = []
    for type_name in types_to_search:
        obj_def = ontology.objects.get(type_name)
        if not obj_def:
            continue
        text_cols = [p for p, d in obj_def.properties.items() if d.type == "str"]
        if not text_cols:
            continue

        for record in query_fn(type_name):
            matched = [
                col for col in text_cols
                if record.get(col) and keyword in str(record[col])
            ]
            if not matched:
                continue
            enriched = dict(record)
            enriched["_object_type"] = type_name
            enriched["_matched_field"] = ", ".join(matched)
            results.append(enriched)
            if len(results) >= limit:
                return results
    return results
