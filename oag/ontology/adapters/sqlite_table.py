"""SQLite table/view-backed ontology object adapter.

This adapter only talks to an existing SQLite database. It does not create
tables and does not import JSON data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..schema import Ontology, ObjectSourceDef


class SqliteTableAdapter:
    """ObjectAdapter backed by an existing SQLite table or view."""

    def __init__(self, ontology: Ontology, object_type: str,
                 source: ObjectSourceDef, domain_dir: Path):
        self.ontology = ontology
        self.object_type = object_type
        self.source = source
        self.domain_dir = domain_dir
        self.table = source.table or source.config.get("table") or source.config.get("view")
        if not self.table:
            self.table = ontology.table_name(object_type)
        self.id_field = source.id_field or ontology.get_id_column(object_type)
        self.conn = sqlite3.connect(self._db_path(), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    @classmethod
    def factory(cls, domain_dir: str | Path):
        base_dir = Path(domain_dir).resolve()

        def build(ontology: Ontology, object_type: str,
                  source: ObjectSourceDef, **kwargs):
            return cls(
                ontology=ontology,
                object_type=object_type,
                source=source,
                domain_dir=base_dir,
            )

        return build

    def query(self, object_type: str, filters: dict[str, Any] | None = None,
              limit: int | None = None, order_by: str | None = None,
              offset: int | None = None) -> list[dict]:
        valid_cols = self._valid_columns()
        sql = f"SELECT * FROM {_quote_ident(self.table)}"
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

    def count(self, object_type: str,
              filters: dict[str, Any] | None = None) -> int:
        valid_cols = self._valid_columns()
        sql = f"SELECT COUNT(*) FROM {_quote_ident(self.table)}"
        where, params = _where_clause(filters, valid_cols)
        if where:
            sql += f" WHERE {where}"
        return int(self.conn.execute(sql, params).fetchone()[0])

    def query_by_id(self, object_type: str, id_value: Any) -> dict | None:
        if not self.id_field:
            return None
        rows = self.query(object_type, {self.id_field: id_value}, limit=1)
        return rows[0] if rows else None

    def search_text(self, keyword: str, object_types: list[str] | None = None,
                    limit: int = 20) -> list[dict]:
        if not keyword:
            return []
        text_cols = [
            name for name, prop in self.ontology.objects[self.object_type].properties.items()
            if prop.type == "str"
        ]
        if not text_cols:
            return []
        valid_cols = self._valid_columns()
        text_cols = [col for col in text_cols if col in valid_cols]
        if not text_cols:
            return []

        where = " OR ".join(f"{_quote_ident(col)} LIKE ?" for col in text_cols)
        rows = self.conn.execute(
            f"SELECT * FROM {_quote_ident(self.table)} WHERE {where} LIMIT ?",
            [f"%{keyword}%"] * len(text_cols) + [limit],
        ).fetchall()
        results = []
        for row in rows:
            record = dict(row)
            matched = [
                col for col in text_cols
                if record.get(col) and keyword in str(record[col])
            ]
            record["_object_type"] = self.object_type
            record["_matched_field"] = ", ".join(matched)
            results.append(record)
        return results

    def insert_record(self, object_type: str, data: dict) -> dict:
        valid_cols = self._writable_columns()
        row = {key: value for key, value in data.items() if key in valid_cols}
        if not row:
            raise ValueError("没有有效字段可插入")
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(_quote_ident(col) for col in cols)
        cursor = self.conn.execute(
            f"INSERT INTO {_quote_ident(self.table)} ({col_names}) VALUES ({placeholders})",
            [row[col] for col in cols],
        )
        self.conn.commit()
        return {"_id": cursor.lastrowid, "inserted": 1}

    def update_record(self, object_type: str, id_value: Any, data: dict) -> dict:
        if not self.id_field:
            raise ValueError(f"{object_type} 没有声明 id 字段，不能 update")
        valid_cols = self._writable_columns() - {self.id_field}
        row = {key: value for key, value in data.items() if key in valid_cols}
        if not row:
            raise ValueError("没有有效字段可更新")
        set_clause = ", ".join(f"{_quote_ident(key)} = ?" for key in row)
        cursor = self.conn.execute(
            (
                f"UPDATE {_quote_ident(self.table)} SET {set_clause} "
                f"WHERE {_quote_ident(self.id_field)} = ?"
            ),
            list(row.values()) + [id_value],
        )
        self.conn.commit()
        return {"updated": cursor.rowcount}

    def delete_record(self, object_type: str, id_value: Any) -> dict:
        if not self.id_field:
            raise ValueError(f"{object_type} 没有声明 id 字段，不能 delete")
        cursor = self.conn.execute(
            f"DELETE FROM {_quote_ident(self.table)} WHERE {_quote_ident(self.id_field)} = ?",
            [id_value],
        )
        self.conn.commit()
        return {"deleted": cursor.rowcount}

    def table_count(self, object_type: str) -> int:
        return self.count(object_type)

    def close(self):
        self.conn.close()

    def _db_path(self) -> str:
        raw = (
            self.source.config.get("db_path")
            or self.source.config.get("database")
            or self.source.config.get("path")
        )
        if not raw:
            raise ValueError(f"{self.object_type} 的 sqlite_table source 需要 config.db_path")
        path = Path(raw)
        if path.is_absolute():
            return str(path)
        return str(self.domain_dir / path)

    def _valid_columns(self) -> set[str]:
        obj_def = self.ontology.objects.get(self.object_type)
        return set(obj_def.properties.keys()) | {"_id"} if obj_def else {"_id"}

    def _writable_columns(self) -> set[str]:
        rows = self.conn.execute(f"PRAGMA table_info({_quote_ident(self.table)})").fetchall()
        db_cols = {row["name"] for row in rows}
        valid_cols = self._valid_columns()
        return db_cols & valid_cols


def _where_clause(filters: dict[str, Any] | None,
                  valid_cols: set[str]) -> tuple[str, list[Any]]:
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


def _quote_ident(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("非法 SQLite 标识符")
    return '"' + value.replace('"', '""') + '"'
