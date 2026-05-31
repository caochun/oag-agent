"""本体数据工具和领域函数执行器。

DataExecutor 把工具名映射到对象查询、统计分析、全文搜索、mutate 操作或
领域 Python 函数。它返回 JSON 字符串，因为 ToolDef handler 会直接作为
模型可见的工具结果。
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .registry import FunctionRegistry
from .store import Store


class DataExecutor:

    def __init__(self, store: Store, registry: FunctionRegistry):
        self.store = store
        self.registry = registry

    def execute(self, name: str, args: dict) -> str:
        try:
            # 内置数据工具在这里处理；未匹配时再尝试调用领域函数注册表。
            if name == "query":
                rows = self.store.query(
                    args["object_type"], args.get("filters"),
                    args.get("limit"), args.get("order_by"), args.get("offset"),
                )
                if not rows:
                    total = self.store.count(args["object_type"])
                    if total == 0:
                        return json.dumps({"results": [], "note": f"{args['object_type']} 当前没有数据。"}, ensure_ascii=False)
                    return json.dumps({"results": [], "note": f"未找到匹配记录（共 {total} 条）。"}, ensure_ascii=False)
                return json.dumps(rows, ensure_ascii=False, default=str)

            if name == "count":
                n = self.store.count(args["object_type"], args.get("filters"))
                return json.dumps({"count": n}, ensure_ascii=False)

            if name == "query_links":
                rows = self.store.query_links(
                    args["source_type"], args["source_id"], args["link_name"],
                )
                return json.dumps(rows, ensure_ascii=False, default=str)

            if name == "describe":
                result = self._describe(args["object_type"], args.get("column"))
                return json.dumps(result, ensure_ascii=False, default=str)

            if name == "pivot":
                result = self._pivot(
                    args["object_type"],
                    args["index"], args["columns"], args["values"],
                    args.get("aggfunc", "mean"),
                )
                return json.dumps(result, ensure_ascii=False, default=str)

            if name == "distribution":
                result = self._distribution(
                    args["object_type"],
                    args["column"], args.get("bins", 10),
                )
                return json.dumps(result, ensure_ascii=False, default=str)

            if name == "mutate":
                return self._mutate(args)

            if name == "search":
                return self._search(args)

            if self.registry.has(name):
                return self.registry.call_as_tool(name, args)

            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"工具执行错误: {e}"}, ensure_ascii=False)

    def _mutate(self, args: dict) -> str:
        operation = args["operation"]
        object_type = args["object_type"]

        if operation == "create":
            result = self.store.insert_record(object_type, args.get("data", {}))
        elif operation == "update":
            result = self.store.update_record(object_type, args["object_id"], args.get("data", {}))
        else:
            result = self.store.delete_record(object_type, args["object_id"])

        return json.dumps(result, ensure_ascii=False, default=str)

    def _search(self, args: dict) -> str:
        keyword = args.get("keyword", "")
        object_types = args.get("object_types")
        limit = args.get("limit", 20)
        results = self.store.search_text(keyword, object_types, limit)
        return json.dumps(results, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def _describe(self, object_type: str, column: str | None = None) -> dict:
        rows = self.store.query(object_type)
        if not rows:
            return {"error": f"{object_type} has no data"}
        # Pandas 只用于对查询结果做小规模内存分析，不是权威数据源。
        df = pd.DataFrame(rows).drop(columns=["_id"], errors="ignore")

        if column:
            if column not in df.columns:
                return {"error": f"Unknown column: {column}", "columns": list(df.columns)}
            series = df[column]
            if pd.api.types.is_numeric_dtype(series):
                desc = series.describe().to_dict()
                return {k: _clean(v) for k, v in desc.items()}
            else:
                return {
                    "count": int(series.count()),
                    "unique": int(series.nunique()),
                    "top": _clean(series.mode().iloc[0]) if not series.mode().empty else None,
                    "freq": int(series.value_counts().iloc[0]) if not series.empty else 0,
                    "sample": [_clean(x) for x in series.head(5).tolist()],
                }
        else:
            summary = {
                "rows": len(df),
                "columns": list(df.columns),
                "numeric": [],
                "text": [],
            }
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    summary["numeric"].append(col)
                else:
                    summary["text"].append(col)
            return summary

    def _pivot(self, object_type: str, index: str, columns: str,
               values: str, aggfunc: str = "mean") -> dict:
        rows = self.store.query(object_type)
        if not rows:
            return {"error": f"{object_type} has no data"}
        df = pd.DataFrame(rows).drop(columns=["_id"], errors="ignore")

        for col in [index, columns, values]:
            if col not in df.columns:
                return {"error": f"Unknown column: {col}", "columns": list(df.columns)}

        agg_map = {"mean": "mean", "sum": "sum", "count": "count", "min": "min", "max": "max"}
        func = agg_map.get(aggfunc, "mean")

        try:
            pt = pd.pivot_table(df, values=values, index=index, columns=columns,
                                aggfunc=func, fill_value=0)
            return {
                "index": [_clean(x) for x in pt.index.tolist()],
                "columns": [_clean(x) for x in pt.columns.tolist()],
                "data": [[_clean(cell) for cell in row] for row in pt.values.tolist()],
            }
        except Exception as e:
            return {"error": str(e)}

    def _distribution(self, object_type: str, column: str,
                      bins: int = 10) -> dict:
        rows = self.store.query(object_type)
        if not rows:
            return {"error": f"{object_type} has no data"}
        df = pd.DataFrame(rows).drop(columns=["_id"], errors="ignore")

        if column not in df.columns:
            return {"error": f"Unknown column: {column}", "columns": list(df.columns)}

        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return {"error": f"Column {column} has no numeric values"}

        counts, edges = pd.cut(series, bins=bins, retbins=True)
        hist = counts.value_counts(sort=False)

        return {
            "column": column,
            "count": int(series.count()),
            "min": _clean(series.min()),
            "max": _clean(series.max()),
            "mean": _clean(series.mean()),
            "std": _clean(series.std()),
            "bins": [
                {
                    "range": f"{_clean(edges[i])}-{_clean(edges[i+1])}",
                    "count": int(hist.iloc[i]),
                }
                for i in range(len(hist))
            ],
        }


def _clean(v: Any) -> Any:
    if isinstance(v, float):
        if pd.isna(v):
            return None
        return round(v, 4) if v != int(v) else int(v)
    return v
