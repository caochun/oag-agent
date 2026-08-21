"""Generic read-mostly SQLite property-graph source adapter.

The adapter maps an existing node/edge database to the logical Object and
Relation contracts. Physical names stay in ``DataSourceDef.config`` and
``DataBindingDef.mapping``; the agent only sees logical records.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from ..schema import DataBindingDef, DataSourceDef, Ontology


class SqlitePropertyGraphSource:
    def __init__(self, ontology: Ontology, source_name: str,
                 source: DataSourceDef, domain_dir: Path):
        self.source = source
        raw = source.config.get("database") or source.config.get("db_path")
        if not raw:
            raise ValueError(f"数据源 {source_name} 缺少 config.database")
        path = Path(str(raw))
        self.database_path = path.resolve() if path.is_absolute() else (domain_dir / path).resolve()
        self.location = self.database_path

    @classmethod
    def factory(cls, domain_dir: str | Path):
        base_dir = Path(domain_dir).resolve()

        def build(ontology: Ontology, source_name: str,
                  source: DataSourceDef, **_: Any):
            return cls(ontology, source_name, source, base_dir)

        return build

    def query_records(self, kind: str, type_name: str, binding: DataBindingDef,
                      filters=None, limit=None, order_by=None, offset=None):
        table, mapping = self._table_mapping(kind, binding)
        where, params = self._where(mapping, binding, filters)
        sql = f"SELECT * FROM {self._ident(table)}{where}"
        if order_by:
            field = order_by.lstrip("-")
            expression = self._field_expression(mapping, field)
            direction = "DESC" if order_by.startswith("-") else "ASC"
            sql += f" ORDER BY {expression} {direction}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        if offset is not None:
            if limit is None:
                sql += " LIMIT -1"
            sql += " OFFSET ?"
            params.append(max(0, int(offset)))
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._logical_record(kind, dict(row), mapping) for row in rows]

    def count_records(self, kind: str, type_name: str, binding: DataBindingDef,
                      filters=None):
        table, mapping = self._table_mapping(kind, binding)
        where, params = self._where(mapping, binding, filters)
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM {self._ident(table)}{where}", params,
            ).fetchone()
        return int(row[0])

    def query_record_by_id(self, kind: str, type_name: str,
                           binding: DataBindingDef, id_value: Any):
        rows = self.query_records(kind, type_name, binding, {"id": id_value}, limit=1)
        return rows[0] if rows else None

    def query_relations(self, relation_type: str, binding: DataBindingDef, *,
                        filters=None, from_id=None, to_id=None,
                        direction="out", limit=None, order_by=None, offset=None):
        effective = dict(filters or {})
        if direction == "both" and from_id is not None and to_id is None:
            outgoing = self.query_records(
                "relation", relation_type, binding,
                {**effective, "from": from_id}, order_by=order_by,
            )
            incoming = self.query_records(
                "relation", relation_type, binding,
                {**effective, "to": from_id}, order_by=order_by,
            )
            rows = list({str(row.get("id")): row for row in [*outgoing, *incoming]}.values())
            start = max(0, int(offset or 0))
            end = None if limit is None else start + max(0, int(limit))
            return rows[start:end]
        if from_id is not None:
            effective["to" if direction == "in" else "from"] = from_id
        if to_id is not None:
            effective["to"] = to_id
        return self.query_records(
            "relation", relation_type, binding, effective,
            limit, order_by, offset,
        )

    def search_records(self, kind: str, type_name: str, binding: DataBindingDef,
                       keyword: str, limit=20):
        table, mapping = self._table_mapping(kind, binding)
        needle = f"%{str(keyword).lower()}%"
        text_fields = list(dict.fromkeys(
            field for key in ("name", "properties")
            if (field := mapping.get(key))
        ))
        if not text_fields:
            return []
        conditions = " OR ".join(
            f"lower(CAST({self._ident(field)} AS TEXT)) LIKE ?"
            for field in text_fields
        )
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM {self._ident(table)} WHERE {conditions} LIMIT ?",
                (*([needle] * len(text_fields)), max(1, int(limit))),
            ).fetchall()
        return [self._logical_record(kind, dict(row), mapping) for row in rows]

    def query_by_ids(self, kind: str, type_name: str, binding: DataBindingDef,
                     ids, *, include_retired: bool = False):
        return [
            record
            for record_id in ids
            if (record := self.query_record_by_id(kind, type_name, binding, record_id))
            is not None
        ]

    def query_adjacent(self, relation_type: str, binding: DataBindingDef,
                       object_ids, *, include_retired: bool = False):
        rows: dict[str, dict[str, Any]] = {}
        for object_id in object_ids:
            for record in self.query_relations(
                relation_type, binding, from_id=object_id, direction="both",
            ):
                rows[str(record.get("id"))] = record
        return list(rows.values())

    def type_counts(self, kind: str, type_name: str, binding: DataBindingDef):
        counts: dict[str, int] = {}
        for record in self.query_records(kind, type_name, binding):
            discriminator = str(record.get("type") or "unknown")
            counts[discriminator] = counts.get(discriminator, 0) + 1
        return counts

    def record_exists(self, kind: str, type_name: str, binding: DataBindingDef,
                      record_id: str):
        return self.query_record_by_id(kind, type_name, binding, record_id) is not None

    def get_record_version(self, kind: str, type_name: str,
                           binding: DataBindingDef, record_id: str):
        return None

    def close(self):
        return None

    def _table_mapping(self, kind: str, binding: DataBindingDef):
        config = self.source.config
        mapping = {
            "id": "id",
            "type": "type",
            "name": "name",
            "properties": "properties",
            "from": "source_id",
            "to": "target_id",
        }
        table = config.get("objects_table", "nodes") if kind == "object" else config.get("relations_table", "edges")
        mapping.update(binding.mapping or {})
        selector_table = binding.selector.get("table")
        if selector_table:
            table = selector_table
        return str(table), mapping

    def _where(self, mapping, binding, filters):
        clauses = []
        params = []
        selector = binding.selector
        if selector.get("type"):
            clauses.append(f"{self._ident(mapping['type'])} = ?")
            params.append(selector["type"])
        for raw_logical, expected in (filters or {}).items():
            logical, op = (
                raw_logical.rsplit("__", 1)
                if "__" in raw_logical
                else (raw_logical, "eq")
            )
            if op not in {"eq", "ne", "like", "gt", "gte", "lt", "lte", "in"}:
                raise ValueError(f"不支持的查询操作符: {op}")
            expression = self._field_expression(mapping, logical)
            if op == "like":
                clauses.append(f"lower(CAST({expression} AS TEXT)) LIKE ?")
                params.append(f"%{str(expected).lower()}%")
                continue
            if op == "in":
                values = (
                    list(expected)
                    if isinstance(expected, (list, tuple, set))
                    else [expected]
                )
                if not values:
                    clauses.append("0")
                    continue
                clauses.append(
                    f"{expression} IN ({', '.join('?' for _ in values)})"
                )
                params.extend(values)
            elif op == "eq" and expected is None:
                clauses.append(f"{expression} IS NULL")
            elif op == "ne" and expected is None:
                clauses.append(f"{expression} IS NOT NULL")
            else:
                comparator = {
                    "eq": "=", "ne": "<>", "gt": ">", "gte": ">=",
                    "lt": "<", "lte": "<=",
                }[op]
                clauses.append(f"{expression} {comparator} ?")
                params.append(expected)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def _field_expression(self, mapping, logical):
        if not isinstance(logical, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.-]*", logical,
        ):
            raise ValueError(f"无效的查询字段: {logical!r}")
        if logical in mapping:
            return self._ident(mapping[logical])
        properties_field = mapping.get("properties")
        if not properties_field:
            raise ValueError(f"查询字段没有物理映射: {logical}")
        return (
            f"json_extract({self._ident(properties_field)}, '$.{logical}')"
        )

    @staticmethod
    def _logical_record(kind, row, mapping):
        properties_value = row.get(mapping.get("properties", "properties"), {})
        if isinstance(properties_value, str):
            try:
                properties_value = json.loads(properties_value)
            except json.JSONDecodeError:
                properties_value = {"value": properties_value}
        result = {
            "id": row.get(mapping.get("id", "id")),
            "type": row.get(mapping.get("type", "type")),
            "properties": properties_value or {},
        }
        if kind == "object":
            result["name"] = row.get(mapping.get("name", "name")) or result["id"]
        else:
            result["from"] = row.get(mapping.get("from", "source_id"))
            result["to"] = row.get(mapping.get("to", "target_id"))
        return result

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _ident(value: str) -> str:
        if not value or "\x00" in value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for char in value):
            raise ValueError(f"非法 SQLite 标识符: {value}")
        return '"' + value + '"'
