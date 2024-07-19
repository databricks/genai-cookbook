# Databricks notebook source
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk import WorkspaceClient
import os

w = WorkspaceClient()

def validate_catalog_and_schema_exist(UC_CATALOG, UC_SCHEMA):
  # Create UC Catalog if it does not exist, otherwise, raise an exception
  try:
      _ = w.catalogs.get(UC_CATALOG)
      print(f"PASS: UC catalog `{UC_CATALOG}` exists")
  except NotFound as e:
      print(f"`{UC_CATALOG}` does not exist, trying to create...")
      try:
          _ = w.catalogs.create(name=UC_CATALOG)
      except PermissionDenied as e:
          print(f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create.  Please provide an existing UC Catalog.")
          raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")
          
  # Create UC Schema if it does not exist, otherwise, raise an exception
  try:
      _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
      print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
  except NotFound as e:
      print(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
      try:
          _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
          print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` created")
      except PermissionDenied as e:
          print(f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create.  Please provide an existing UC Schema.")
          raise ValueError("Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist.")

# COMMAND ----------

def validate_vector_search_endpoint_exists(VECTOR_SEARCH_ENDPOINT):
  # Create the Vector Search endpoint if it does not exist
  from databricks.sdk import WorkspaceClient
  from databricks.sdk.service.vectorsearch import EndpointType
  w = WorkspaceClient()
  vector_search_endpoints = w.vector_search_endpoints.list_endpoints()
  if sum([VECTOR_SEARCH_ENDPOINT == ve.name for ve in vector_search_endpoints]) == 0:
      print(f"Please wait, creating Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}`.  This can take up to 20 minutes...")
      w.vector_search_endpoints.create_endpoint_and_wait(VECTOR_SEARCH_ENDPOINT, endpoint_type=EndpointType.STANDARD)

  # Make sure vector search endpoint is online and ready.
  w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(VECTOR_SEARCH_ENDPOINT)

  print(f"PASS: Vector Search endpoint `{VECTOR_SEARCH_ENDPOINT}` exists")