# Databricks notebook source
from typing import Dict, Any

def _flatten_nested_params(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, str]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_nested_params(v, new_key, sep=sep))
        else:
          items[new_key] = v
    return items

def tag_delta_table(table_fqn, config):
    flat_config = _flatten_nested_params(config)
    sqls = []
    for key, item in flat_config.items():
        
        sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("{key.replace("/", "__")}" = "{item}")
        """)
    sqls.append(f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("table_source" = "rag_poc_pdf")
        """)
    for sql in sqls:
        # print(sql)
        spark.sql(sql)
