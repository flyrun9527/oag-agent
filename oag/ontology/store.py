"""Backward-compatible ontology object store.

The current data boundary is ObjectRepository and source adapters. Some legacy
domains still import Store and rely on an in-memory SQLite scratch database.
This module keeps those domains loadable while routing source-declared objects
through ObjectRepository.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .registry import FunctionRegistry
from .repository import ObjectRepository
from .schema import Ontology

TYPE_MAP = {
    "str": "TEXT",
    "int": "INTEGER",
    "integer": "INTEGER",
    "float": "REAL",
    "number": "REAL",
    "bool": "INTEGER",
}


class Store(ObjectRepository):
    """Repository-compatible store with legacy SQLite helpers."""

    def __init__(
        self,
        ontology: Ontology,
        registry: FunctionRegistry | None = None,
        db_path: str = ":memory:",
    ):
        self.registry = registry or FunctionRegistry()
        super().__init__(ontology, self.registry)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def create_tables(self):
        for type_name, obj_def in self.ontology.objects.items():
            if obj_def.source is not None:
                continue
            table = self.ontology.table_name(type_name)
            cols = ['"_id" INTEGER PRIMARY KEY AUTOINCREMENT']
            for prop_name, prop_def in obj_def.properties.items():
                sql_type = TYPE_MAP.get(prop_def.type, "TEXT")
                cols.append(f"{_quote_ident(prop_name)} {sql_type}")
            ddl = f"CREATE TABLE IF NOT EXISTS {_quote_ident(table)} ({', '.join(cols)})"
            self.conn.execute(ddl)
        self.conn.commit()

    def load_data(
        self,
        object_type: str,
        data: list[dict],
        field_mapping: dict[str, str] | None = None,
    ) -> int:
        obj_def = self.ontology.objects.get(object_type)
        if not obj_def or obj_def.source is not None or not data:
            return 0

        table = self.ontology.table_name(object_type)
        count = self.conn.execute(
            f"SELECT COUNT(*) FROM {_quote_ident(table)}"
        ).fetchone()[0]
        if count > 0:
            return 0

        valid_cols = set(obj_def.properties.keys())
        inserted = 0
        for raw_row in data:
            row = {}
            for raw_key, value in raw_row.items():
                mapped = field_mapping.get(raw_key, raw_key) if field_mapping else raw_key
                if mapped in valid_cols:
                    row[mapped] = value
            if not row:
                continue
            cols = list(row.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(_quote_ident(col) for col in cols)
            self.conn.execute(
                f"INSERT INTO {_quote_ident(table)} ({col_names}) VALUES ({placeholders})",
                [row[col] for col in cols],
            )
            inserted += 1
        self.conn.commit()
        return inserted

    def load_json_file(
        self,
        object_type: str,
        file_path: str | Path,
        field_mapping: dict[str, str] | None = None,
    ) -> int:
        path = Path(file_path)
        if not path.exists():
            return 0
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _load_loose_json_objects(text)
        if isinstance(data, dict):
            data = data.get("data", data.get("items", []))
        return self.load_data(object_type, data, field_mapping)

    def query(
        self,
        object_type: str,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
        order_by: str | None = None,
        offset: int | None = None,
    ) -> list[dict]:
        if self._uses_repository_adapter(object_type):
            return super().query(object_type, filters, limit, order_by, offset)

        obj_def = self.ontology.objects.get(object_type)
        if not obj_def:
            return []
        valid_cols = set(obj_def.properties.keys()) | {"_id"}
        table = self.ontology.table_name(object_type)
        sql = f"SELECT * FROM {_quote_ident(table)}"
        where, params = _where_clause(filters, valid_cols)
        if where:
            sql += f" WHERE {where}"
        if order_by:
            col = order_by.lstrip("-")
            if col in valid_cols:
                direction = "DESC" if order_by.startswith("-") else "ASC"
                sql += f" ORDER BY {_quote_ident(col)} {direction}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        if offset is not None:
            sql += " OFFSET ?"
            params.append(offset)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def count(self, object_type: str, filters: dict[str, Any] | None = None) -> int:
        if self._uses_repository_adapter(object_type):
            return super().count(object_type, filters)
        return len(self.query(object_type, filters))

    def query_by_id(self, object_type: str, id_value: Any) -> dict | None:
        if self._uses_repository_adapter(object_type):
            return super().query_by_id(object_type, id_value)
        id_col = self.ontology.get_id_column(object_type)
        if not id_col:
            return None
        results = self.query(object_type, {id_col: id_value}, limit=1)
        return results[0] if results else None

    def execute_sql(self, sql: str, params: list | tuple | None = None) -> list[dict]:
        rows = self.conn.execute(sql, list(params or [])).fetchall()
        return [dict(row) for row in rows]

    def execute_write(self, sql: str, params: list | tuple | None = None) -> int:
        cursor = self.conn.execute(sql, list(params or []))
        self.conn.commit()
        return cursor.rowcount

    def table_count(self, object_type: str) -> int:
        if self._uses_repository_adapter(object_type):
            return super().table_count(object_type)
        table = self.ontology.table_name(object_type)
        return int(self.conn.execute(
            f"SELECT COUNT(*) FROM {_quote_ident(table)}"
        ).fetchone()[0])

    def insert_record(self, object_type: str, data: dict) -> dict:
        if self._uses_repository_adapter(object_type):
            return super().insert_record(object_type, data)

        obj_def = self.ontology.objects.get(object_type)
        if not obj_def:
            raise ValueError(f"未知对象类型: {object_type}")
        valid_cols = set(obj_def.properties.keys())
        row = {key: value for key, value in data.items() if key in valid_cols}
        if not row:
            raise ValueError("没有有效字段可插入")
        table = self.ontology.table_name(object_type)
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(_quote_ident(col) for col in cols)
        cursor = self.conn.execute(
            f"INSERT INTO {_quote_ident(table)} ({col_names}) VALUES ({placeholders})",
            [row[col] for col in cols],
        )
        self.conn.commit()
        return {"_id": cursor.lastrowid, "inserted": 1}

    def update_record(self, object_type: str, id_value: Any, data: dict) -> dict:
        if self._uses_repository_adapter(object_type):
            return super().update_record(object_type, id_value, data)

        obj_def = self.ontology.objects.get(object_type)
        if not obj_def:
            raise ValueError(f"未知对象类型: {object_type}")
        id_col = self.ontology.get_id_column(object_type) or "_id"
        valid_cols = set(obj_def.properties.keys()) - {id_col}
        row = {key: value for key, value in data.items() if key in valid_cols}
        if not row:
            raise ValueError("没有有效字段可更新")
        table = self.ontology.table_name(object_type)
        set_clause = ", ".join(f"{_quote_ident(key)} = ?" for key in row)
        cursor = self.conn.execute(
            (
                f"UPDATE {_quote_ident(table)} SET {set_clause} "
                f"WHERE {_quote_ident(id_col)} = ?"
            ),
            list(row.values()) + [id_value],
        )
        self.conn.commit()
        return {"updated": cursor.rowcount}

    def delete_record(self, object_type: str, id_value: Any) -> dict:
        if self._uses_repository_adapter(object_type):
            return super().delete_record(object_type, id_value)

        obj_def = self.ontology.objects.get(object_type)
        if not obj_def:
            raise ValueError(f"未知对象类型: {object_type}")
        id_col = self.ontology.get_id_column(object_type) or "_id"
        table = self.ontology.table_name(object_type)
        cursor = self.conn.execute(
            f"DELETE FROM {_quote_ident(table)} WHERE {_quote_ident(id_col)} = ?",
            [id_value],
        )
        self.conn.commit()
        return {"deleted": cursor.rowcount}

    def search_text(
        self,
        keyword: str,
        object_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict]:
        if not keyword:
            return []
        types_to_search = object_types or list(self.ontology.objects.keys())
        results: list[dict] = []
        for type_name in types_to_search:
            if self._uses_repository_adapter(type_name):
                rows = super().search_text(keyword, [type_name], limit - len(results))
                results.extend(rows)
            else:
                rows = self._search_sql_text(type_name, keyword, limit - len(results))
                results.extend(rows)
            if len(results) >= limit:
                break
        return results[:limit]

    def close(self):
        super().close()
        self.conn.close()

    def _search_sql_text(self, object_type: str, keyword: str, limit: int) -> list[dict]:
        obj_def = self.ontology.objects.get(object_type)
        if not obj_def or limit <= 0:
            return []
        text_cols = [
            name for name, prop in obj_def.properties.items()
            if prop.type == "str"
        ]
        if not text_cols:
            return []
        table = self.ontology.table_name(object_type)
        where = " OR ".join(f"{_quote_ident(col)} LIKE ?" for col in text_cols)
        rows = self.conn.execute(
            f"SELECT * FROM {_quote_ident(table)} WHERE {where} LIMIT ?",
            [f"%{keyword}%"] * len(text_cols) + [limit],
        ).fetchall()
        results = []
        for row in rows:
            record = dict(row)
            matched = [
                col for col in text_cols
                if record.get(col) and keyword in str(record[col])
            ]
            record["_object_type"] = object_type
            record["_matched_field"] = ", ".join(matched)
            results.append(record)
        return results

    def _uses_repository_adapter(self, object_type: str) -> bool:
        obj_def = self.ontology.objects.get(object_type)
        return bool(obj_def and obj_def.source is not None)


def _where_clause(
    filters: dict[str, Any] | None,
    valid_cols: set[str],
) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    for key, value in (filters or {}).items():
        field, op = key.split("__", 1) if "__" in key else (key, "eq")
        if field not in valid_cols:
            continue
        column = _quote_ident(field)
        if op == "like":
            clauses.append(f"{column} LIKE ?")
            params.append(f"%{value}%")
        elif op == "gt":
            clauses.append(f"{column} > ?")
            params.append(value)
        elif op == "gte":
            clauses.append(f"{column} >= ?")
            params.append(value)
        elif op == "lt":
            clauses.append(f"{column} < ?")
            params.append(value)
        elif op == "lte":
            clauses.append(f"{column} <= ?")
            params.append(value)
        elif op == "ne":
            clauses.append(f"{column} != ?")
            params.append(value)
        else:
            clauses.append(f"{column} = ?")
            params.append(value)
    return " AND ".join(clauses), params


def _load_loose_json_objects(text: str) -> list[dict]:
    cleaned = text.replace("[", "").replace("]", "")
    data = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(cleaned):
        chunk = cleaned[pos:].lstrip(" ,\n\r\t")
        if not chunk:
            break
        pos = len(cleaned) - len(chunk)
        try:
            obj, end = decoder.raw_decode(chunk)
            data.append(obj)
            pos += end
        except json.JSONDecodeError:
            break
    return data


def _quote_ident(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("非法 SQLite 标识符")
    return '"' + value.replace('"', '""') + '"'
