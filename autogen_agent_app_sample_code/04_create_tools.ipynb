# Databricks notebook source
# MAGIC %md
# MAGIC ## 👉 START HERE: How to use this notebook
# MAGIC
# MAGIC # Step 2: Create tools for your Agent
# MAGIC
# MAGIC <todo>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Important note:** Throughout this notebook, we indicate which cell's code you:
# MAGIC - ✅✏️ should customize - these cells contain code & config with business logic that you should edit to meet your requirements & tune quality.
# MAGIC - 🚫✏️ should not customize - these cells contain boilerplate code required to load/save/execute your Agent
# MAGIC
# MAGIC *Cells that don't require customization still need to be run!  You CAN change these cells, but if this is the first time using this notebook, we suggest not doing so.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Install Python libraries
# MAGIC
# MAGIC You do not need to modify this cell unless you need additional Python packages in your Agent.

# COMMAND ----------

# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC # Restart to load the packages into the Python environment
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Connect to Databricks
# MAGIC
# MAGIC If running locally in an IDE using Databricks Connect, connect the Spark client & configure MLflow to use Databricks Managed MLflow.  If this running in a Databricks Notebook, these values are already set.

# COMMAND ----------

from mlflow.utils import databricks_utils as du
import os

if not du.is_in_databricks_notebook():
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.getOrCreate()
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Load the Agent's UC storage locations; set up MLflow experiment
# MAGIC
# MAGIC This notebook uses the UC model, MLflow Experiment, and Evaluation Set that you specified in the [Agent setup](02_agent_setup.ipynb) notebook.

# COMMAND ----------

from cookbook.config.shared.agent_storage_location import AgentStorageConfig
from cookbook.databricks_utils import get_mlflow_experiment_url
from cookbook.config import load_serializable_config_from_yaml_file
import mlflow 

# Load the Agent's storage locations
agent_storage_config: AgentStorageConfig= load_serializable_config_from_yaml_file("./configs/agent_storage_config.yaml")

# Show the Agent's storage locations
agent_storage_config.pretty_print()

# set the MLflow experiment
experiment_info = mlflow.set_experiment(agent_storage_config.mlflow_experiment_name)
# If running in a local IDE, set the MLflow experiment name as an environment variable
os.environ["MLFLOW_EXPERIMENT_NAME"] = agent_storage_config.mlflow_experiment_name

print(f"View the MLflow Experiment `{agent_storage_config.mlflow_experiment_name}` at {get_mlflow_experiment_url(experiment_info.experiment_id)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # create tools
# MAGIC
# MAGIC - we will store all tools in the `user_tools` folder
# MAGIC - first, create a local function & test it with pytest
# MAGIC - then, deploy it as a UC tool & test it with pytest
# MAGIC - then, add the tool to the Agent 

# COMMAND ----------

# MAGIC %md
# MAGIC always reload the tool's code

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 3

# COMMAND ----------

# MAGIC %md
# MAGIC ## lets do an example of a simple, but fake tool that translates old to new SKUs.

# COMMAND ----------

# MAGIC %md
# MAGIC 1, create the python function that will become your UC function.  you need to annotate the function with docstrings & type hints - these are used to create the tool's metadata in UC.

# COMMAND ----------

# MAGIC %%writefile tools/sample_tool.py
# MAGIC
# MAGIC def sku_sample_translator(old_sku: str) -> str:
# MAGIC     """
# MAGIC     Translates a pre-2024 SKU formatted as "OLD-XXX-YYYY" to the new SKU format "NEW-YYYY-XXX".
# MAGIC
# MAGIC     Args:
# MAGIC         old_sku (str): The old SKU in the format "OLD-XXX-YYYY".
# MAGIC
# MAGIC     Returns:
# MAGIC         str: The new SKU in the format "NEW-YYYY-XXX".
# MAGIC
# MAGIC     Raises:
# MAGIC         ValueError: If the SKU format is invalid, providing specific error details.
# MAGIC     """
# MAGIC     import re
# MAGIC
# MAGIC     if not isinstance(old_sku, str):
# MAGIC         raise ValueError("SKU must be a string")
# MAGIC
# MAGIC     # Normalize input by removing extra whitespace and converting to uppercase
# MAGIC     old_sku = old_sku.strip().upper()
# MAGIC
# MAGIC     # Define the regex pattern for the old SKU format
# MAGIC     pattern = r"^OLD-([A-Z]{3})-(\d{4})$"
# MAGIC
# MAGIC     # Match the old SKU against the pattern
# MAGIC     match = re.match(pattern, old_sku)
# MAGIC     if not match:
# MAGIC         if not old_sku.startswith("OLD-"):
# MAGIC             raise ValueError("SKU must start with 'OLD-'")
# MAGIC         if not re.match(r"^OLD-[A-Z]{3}-\d{4}$", old_sku):
# MAGIC             raise ValueError(
# MAGIC                 "SKU format must be 'OLD-XXX-YYYY' where X is a letter and Y is a digit"
# MAGIC             )
# MAGIC         raise ValueError("Invalid SKU format")
# MAGIC
# MAGIC     # Extract the letter code and numeric part
# MAGIC     letter_code, numeric_part = match.groups()
# MAGIC
# MAGIC     # Additional validation for numeric part
# MAGIC     if not (1 <= int(numeric_part) <= 9999):
# MAGIC         raise ValueError("Numeric part must be between 0001 and 9999")
# MAGIC
# MAGIC     # Construct the new SKU
# MAGIC     new_sku = f"NEW-{numeric_part}-{letter_code}"
# MAGIC     return new_sku
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's import the tool and test it locally

# COMMAND ----------

from tools.sample_tool import sku_sample_translator

sku_sample_translator("OLD-XXX-1234")

# COMMAND ----------

# MAGIC %md
# MAGIC now, lets write some pyTest unit tests for the tool - these are just samples, you will need to write your own

# COMMAND ----------

# MAGIC %%writefile tools/test_sample_tool.py
# MAGIC import pytest
# MAGIC from tools.sample_tool import sku_sample_translator
# MAGIC
# MAGIC
# MAGIC
# MAGIC def test_valid_sku_translation():
# MAGIC     """Test successful SKU translation with valid input."""
# MAGIC     assert sku_sample_translator("OLD-ABC-1234") == "NEW-1234-ABC"
# MAGIC     assert sku_sample_translator("OLD-XYZ-0001") == "NEW-0001-XYZ"
# MAGIC     assert sku_sample_translator("old-def-5678") == "NEW-5678-DEF"  # Test case insensitivity
# MAGIC
# MAGIC
# MAGIC def test_whitespace_handling():
# MAGIC     """Test that the function handles extra whitespace correctly."""
# MAGIC     assert sku_sample_translator("  OLD-ABC-1234  ") == "NEW-1234-ABC"
# MAGIC     assert sku_sample_translator("\tOLD-ABC-1234\n") == "NEW-1234-ABC"
# MAGIC
# MAGIC
# MAGIC def test_invalid_input_type():
# MAGIC     """Test that non-string inputs raise ValueError."""
# MAGIC     with pytest.raises(ValueError, match="SKU must be a string"):
# MAGIC         sku_sample_translator(123)
# MAGIC     with pytest.raises(ValueError, match="SKU must be a string"):
# MAGIC         sku_sample_translator(None)
# MAGIC
# MAGIC
# MAGIC def test_invalid_prefix():
# MAGIC     """Test that SKUs not starting with 'OLD-' raise ValueError."""
# MAGIC     with pytest.raises(ValueError, match="SKU must start with 'OLD-'"):
# MAGIC         sku_sample_translator("NEW-ABC-1234")
# MAGIC     with pytest.raises(ValueError, match="SKU must start with 'OLD-'"):
# MAGIC         sku_sample_translator("XXX-ABC-1234")
# MAGIC
# MAGIC
# MAGIC def test_invalid_format():
# MAGIC     """Test various invalid SKU formats."""
# MAGIC     invalid_skus = [
# MAGIC         "OLD-AB-1234",  # Too few letters
# MAGIC         "OLD-ABCD-1234",  # Too many letters
# MAGIC         "OLD-123-1234",  # Numbers instead of letters
# MAGIC         "OLD-ABC-123",  # Too few digits
# MAGIC         "OLD-ABC-12345",  # Too many digits
# MAGIC         "OLD-ABC-XXXX",  # Letters instead of numbers
# MAGIC         "OLD-A1C-1234",  # Mixed letters and numbers in middle
# MAGIC     ]
# MAGIC
# MAGIC     for sku in invalid_skus:
# MAGIC         with pytest.raises(
# MAGIC             ValueError,
# MAGIC             match="SKU format must be 'OLD-XXX-YYYY' where X is a letter and Y is a digit",
# MAGIC         ):
# MAGIC             sku_sample_translator(sku)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC now, lets run the tests

# COMMAND ----------

import pytest

# Run tests from test_sku_translator.py
pytest.main(["-v", "tools/test_sample_tool.py"])


# COMMAND ----------

# MAGIC %md
# MAGIC Now, lets deploy the tool to Unity catalog.

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from tools.sample_tool import sku_sample_translator

client = DatabricksFunctionClient()
CATALOG = "casaman_ssa"  # Change me!
SCHEMA = "demos"  # Change me if you want

# this will deploy the tool to UC, automatically setting the metadata in UC based on the tool's docstring & typing hints
tool_uc_info = client.create_python_function(func=sku_sample_translator, catalog=CATALOG, schema=SCHEMA, replace=True)

# the tool will deploy to a function in UC called `{catalog}.{schema}.{func}` where {func} is the name of the function
# Print the deployed Unity Catalog function name
print(f"Deployed Unity Catalog function name: {tool_uc_info.full_name}")


# COMMAND ----------

# MAGIC %md
# MAGIC Now, wrap it into a UCTool that will be used by our Agent.  UC tool is just a Pydnatic base model that is serializable to YAML that will load the tool's metadata from UC and wrap it in a callable object.

# COMMAND ----------

from cookbook.tools.uc_tool import UCTool

# wrap the tool into a UCTool which can be passed to our Agent
translate_sku_tool = UCTool(uc_function_name=tool_uc_info.full_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's test the UC tool - the UCTool is a directly callable wrapper around the UC function, so it can be used just like a local function, but the output will be put into a dictionary with either the output in a 'value' key or an 'error' key if an error is raised.
# MAGIC
# MAGIC when an error happens, the UC tool will also return an instruction prompt to show the agent how to think about handling the error.  this can be changed via the `error_prompt` parameter in the UCTool..
# MAGIC

# COMMAND ----------

# successful call
translate_sku_tool(old_sku="OLD-XXX-1234")

# COMMAND ----------

# unsuccessful call
translate_sku_tool(old_sku="OxxLD-XXX-1234")

# COMMAND ----------

# MAGIC %md
# MAGIC now, let's convert our pytests to work with the UC tool.  this requires a bit of transformation to the test code to account for the fact that the output is in a dictionary & exceptions are not raised directly.

# COMMAND ----------

# MAGIC %%writefile tools/test_sample_tool_uc.py
# MAGIC import pytest
# MAGIC from cookbook.tools.uc_tool import UCTool
# MAGIC
# MAGIC # Load the function from the UCTool versus locally
# MAGIC @pytest.fixture
# MAGIC def uc_tool():
# MAGIC     """Fixture to translate a UC tool into a local function."""
# MAGIC     UC_FUNCTION_NAME = "ep.cookbook_local_test.sku_sample_translator"
# MAGIC     loaded_tool = UCTool(uc_function_name=UC_FUNCTION_NAME)
# MAGIC     return loaded_tool
# MAGIC
# MAGIC
# MAGIC # Note: The value will be post processed into the `value` key, so we must check the returned value there.
# MAGIC def test_valid_sku_translation(uc_tool):
# MAGIC     """Test successful SKU translation with valid input."""
# MAGIC     assert uc_tool(old_sku="OLD-ABC-1234")["value"] == "NEW-1234-ABC"
# MAGIC     assert uc_tool(old_sku="OLD-XYZ-0001")["value"] == "NEW-0001-XYZ"
# MAGIC     assert (
# MAGIC         uc_tool(old_sku="old-def-5678")["value"] == "NEW-5678-DEF"
# MAGIC     )  # Test case insensitivity
# MAGIC
# MAGIC
# MAGIC # Note: The value will be post processed into the `value` key, so we must check the returned value there.
# MAGIC def test_whitespace_handling(uc_tool):
# MAGIC     """Test that the function handles extra whitespace correctly."""
# MAGIC     assert uc_tool(old_sku="  OLD-ABC-1234  ")["value"] == "NEW-1234-ABC"
# MAGIC     assert uc_tool(old_sku="\tOLD-ABC-1234\n")["value"] == "NEW-1234-ABC"
# MAGIC
# MAGIC
# MAGIC # Note: the input validation happens BEFORE the function is called by Spark, so we will never get these exceptions from the function.
# MAGIC # Instead, we will get invalid parameters errors from Spark.
# MAGIC def test_invalid_input_type(uc_tool):
# MAGIC     """Test that non-string inputs raise ValueError."""
# MAGIC     assert (
# MAGIC         uc_tool(old_sku=123)["error"]["error_message"]
# MAGIC         == """Invalid parameters provided: {'old_sku': "Parameter old_sku should be of type STRING (corresponding python type <class 'str'>), but got <class 'int'>"}."""
# MAGIC     )
# MAGIC     assert (
# MAGIC         uc_tool(old_sku=None)["error"]["error_message"]
# MAGIC         == """Invalid parameters provided: {'old_sku': "Parameter old_sku should be of type STRING (corresponding python type <class 'str'>), but got <class 'NoneType'>"}."""
# MAGIC     )
# MAGIC
# MAGIC
# MAGIC # Note: The errors will be post processed into the `error_message` key inside the `error` top level key, so we must check for exceptions there.
# MAGIC def test_invalid_prefix(uc_tool):
# MAGIC     """Test that SKUs not starting with 'OLD-' raise ValueError."""
# MAGIC     assert (
# MAGIC         uc_tool(old_sku="NEW-ABC-1234")["error"]["error_message"]
# MAGIC         == "ValueError: SKU must start with 'OLD-'"
# MAGIC     )
# MAGIC     assert (
# MAGIC         uc_tool(old_sku="XXX-ABC-1234")["error"]["error_message"]
# MAGIC         == "ValueError: SKU must start with 'OLD-'"
# MAGIC     )
# MAGIC
# MAGIC
# MAGIC # Note: The errors will be post processed into the `error_message` key inside the `error` top level key, so we must check for exceptions there.
# MAGIC def test_invalid_format(uc_tool):
# MAGIC     """Test various invalid SKU formats."""
# MAGIC     invalid_skus = [
# MAGIC         "OLD-AB-1234",  # Too few letters
# MAGIC         "OLD-ABCD-1234",  # Too many letters
# MAGIC         "OLD-123-1234",  # Numbers instead of letters
# MAGIC         "OLD-ABC-123",  # Too few digits
# MAGIC         "OLD-ABC-12345",  # Too many digits
# MAGIC         "OLD-ABC-XXXX",  # Letters instead of numbers
# MAGIC         "OLD-A1C-1234",  # Mixed letters and numbers in middle
# MAGIC     ]
# MAGIC
# MAGIC     expected_error = "ValueError: SKU format must be 'OLD-XXX-YYYY' where X is a letter and Y is a digit"
# MAGIC     for sku in invalid_skus:
# MAGIC         assert uc_tool(old_sku=sku)["error"]["error_message"] == expected_error
# MAGIC

# COMMAND ----------

import pytest

# Run tests from test_sku_translator.py
pytest.main(["-v", "tools/test_sample_tool_uc.py"])


# COMMAND ----------

# MAGIC %md
# MAGIC # Now, here's another example of a tool that executes python code.

# COMMAND ----------

# MAGIC %%writefile tools/code_exec.py
# MAGIC def python_exec(code: str) -> str:
# MAGIC     """
# MAGIC     Executes Python code in the sandboxed environment and returns its stdout. The runtime is stateless and you can not read output of the previous tool executions. i.e. No such variables "rows", "observation" defined. Calling another tool inside a Python code is NOT allowed.
# MAGIC     Use only standard python libraries and these python libraries: bleach, chardet, charset-normalizer, defusedxml, googleapis-common-protos, grpcio, grpcio-status, jmespath, joblib, numpy, packaging, pandas, patsy, protobuf, pyarrow, pyparsing, python-dateutil, pytz, scikit-learn, scipy, setuptools, six, threadpoolctl, webencodings, user-agents, cryptography.
# MAGIC
# MAGIC     Args:
# MAGIC       code (str): Python code to execute. Remember to print the final result to stdout.
# MAGIC
# MAGIC     Returns:
# MAGIC       str: The output of the executed code.
# MAGIC     """
# MAGIC     import sys
# MAGIC     from io import StringIO
# MAGIC
# MAGIC     sys_stdout = sys.stdout
# MAGIC     redirected_output = StringIO()
# MAGIC     sys.stdout = redirected_output
# MAGIC     exec(code)
# MAGIC     sys.stdout = sys_stdout
# MAGIC     return redirected_output.getvalue()
# MAGIC

# COMMAND ----------

from tools.code_exec import python_exec

python_exec("print('hello')")

# COMMAND ----------

# MAGIC %md
# MAGIC Test it locally

# COMMAND ----------

# MAGIC %%writefile tools/test_code_exec.py
# MAGIC
# MAGIC import pytest
# MAGIC from .code_exec import python_exec
# MAGIC
# MAGIC
# MAGIC def test_basic_arithmetic():
# MAGIC     code = """result = 2 + 2\nprint(result)"""
# MAGIC     assert python_exec(code).strip() == "4"
# MAGIC
# MAGIC
# MAGIC def test_multiple_lines():
# MAGIC     code = "x = 5\n" "y = 3\n" "result = x * y\n" "print(result)"
# MAGIC     assert python_exec(code).strip() == "15"
# MAGIC
# MAGIC
# MAGIC def test_multiple_prints():
# MAGIC     code = """print('first')\nprint('second')\nprint('third')\n"""
# MAGIC     expected = "first\nsecond\nthird\n"
# MAGIC     assert python_exec(code) == expected
# MAGIC
# MAGIC
# MAGIC def test_using_pandas():
# MAGIC     code = (
# MAGIC         "import pandas as pd\n"
# MAGIC         "data = {'col1': [1, 2], 'col2': [3, 4]}\n"
# MAGIC         "df = pd.DataFrame(data)\n"
# MAGIC         "print(df.shape)"
# MAGIC     )
# MAGIC     assert python_exec(code).strip() == "(2, 2)"
# MAGIC
# MAGIC
# MAGIC def test_using_numpy():
# MAGIC     code = "import numpy as np\n" "arr = np.array([1, 2, 3])\n" "print(arr.mean())"
# MAGIC     assert python_exec(code).strip() == "2.0"
# MAGIC
# MAGIC
# MAGIC def test_syntax_error():
# MAGIC     code = "if True\n" "    print('invalid syntax')"
# MAGIC     with pytest.raises(SyntaxError):
# MAGIC         python_exec(code)
# MAGIC
# MAGIC
# MAGIC def test_runtime_error():
# MAGIC     code = "x = 1 / 0\n" "print(x)"
# MAGIC     with pytest.raises(ZeroDivisionError):
# MAGIC         python_exec(code)
# MAGIC
# MAGIC
# MAGIC def test_undefined_variable():
# MAGIC     code = "print(undefined_variable)"
# MAGIC     with pytest.raises(NameError):
# MAGIC         python_exec(code)
# MAGIC
# MAGIC
# MAGIC def test_multiline_string_manipulation():
# MAGIC     code = "text = '''\n" "Hello\n" "World\n" "'''\n" "print(text.strip())"
# MAGIC     expected = "Hello\nWorld"
# MAGIC     assert python_exec(code).strip() == expected
# MAGIC
# MAGIC # Will not fail locally, but will fail in UC.
# MAGIC # def test_unauthorized_flask():
# MAGIC #     code = "from flask import Flask\n" "app = Flask(__name__)\n" "print(app)"
# MAGIC #     with pytest.raises(ImportError):
# MAGIC #         python_exec(code)
# MAGIC
# MAGIC
# MAGIC def test_no_print_statement():
# MAGIC     code = "x = 42\n" "y = x * 2"
# MAGIC     assert python_exec(code) == ""
# MAGIC
# MAGIC
# MAGIC def test_calculation_without_print():
# MAGIC     code = "result = sum([1, 2, 3, 4, 5])\n" "squared = [x**2 for x in range(5)]"
# MAGIC     assert python_exec(code) == ""
# MAGIC
# MAGIC
# MAGIC def test_function_definition_without_call():
# MAGIC     code = "def add(a, b):\n" "    return a + b\n" "result = add(3, 4)"
# MAGIC     assert python_exec(code) == ""
# MAGIC
# MAGIC
# MAGIC def test_class_definition_without_instantiation():
# MAGIC     code = (
# MAGIC         "class Calculator:\n"
# MAGIC         "    def add(self, a, b):\n"
# MAGIC         "        return a + b\n"
# MAGIC         "calc = Calculator()"
# MAGIC     )
# MAGIC     assert python_exec(code) == ""
# MAGIC

# COMMAND ----------

import pytest

# Run tests from test_sku_translator.py
pytest.main(["-v", "tools/test_code_exec.py"])



# COMMAND ----------

# MAGIC %md
# MAGIC Deploy to UC

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from tools.code_exec import python_exec
from cookbook.tools.uc_tool import UCTool

client = DatabricksFunctionClient()
CATALOG = "casaman_ssa"  # Change me!
SCHEMA = "demos"  # Change me if you want

# this will deploy the tool to UC, automatically setting the metadata in UC based on the tool's docstring & typing hints
python_exec_tool_uc_info = client.create_python_function(func=python_exec, catalog=CATALOG, schema=SCHEMA, replace=True)

# the tool will deploy to a function in UC called `{catalog}.{schema}.{func}` where {func} is the name of the function
# Print the deployed Unity Catalog function name
print(f"Deployed Unity Catalog function name: {python_exec_tool_uc_info.full_name}")



# COMMAND ----------

# MAGIC %md
# MAGIC Test as UC Tool for the Agent

# COMMAND ----------

from cookbook.tools.uc_tool import UCTool


# wrap the tool into a UCTool which can be passed to our Agent
python_exec_tool = UCTool(uc_function_name=python_exec_tool_uc_info.full_name)

python_exec_tool(code="print('hello')")


# COMMAND ----------

# MAGIC %md
# MAGIC New tests

# COMMAND ----------

# MAGIC %%writefile tools/test_code_exec_as_uc_tool.py
# MAGIC
# MAGIC import pytest
# MAGIC from cookbook.tools.uc_tool import UCTool
# MAGIC
# MAGIC CATALOG = "ep"
# MAGIC SCHEMA = "cookbook_local_test"
# MAGIC
# MAGIC
# MAGIC @pytest.fixture
# MAGIC def python_exec():
# MAGIC     """Fixture to provide the python_exec function from UCTool."""
# MAGIC     python_exec_tool = UCTool(uc_function_name=f"{CATALOG}.{SCHEMA}.python_exec")
# MAGIC     return python_exec_tool
# MAGIC
# MAGIC
# MAGIC def test_basic_arithmetic(python_exec):
# MAGIC     code = """result = 2 + 2\nprint(result)"""
# MAGIC     assert python_exec(code=code)["value"].strip() == "4"
# MAGIC
# MAGIC
# MAGIC def test_multiple_lines(python_exec):
# MAGIC     code = "x = 5\n" "y = 3\n" "result = x * y\n" "print(result)"
# MAGIC     assert python_exec(code=code)["value"].strip() == "15"
# MAGIC
# MAGIC
# MAGIC def test_multiple_prints(python_exec):
# MAGIC     code = """print('first')\nprint('second')\nprint('third')\n"""
# MAGIC     expected = "first\nsecond\nthird\n"
# MAGIC     assert python_exec(code=code)["value"] == expected
# MAGIC
# MAGIC
# MAGIC def test_using_pandas(python_exec):
# MAGIC     code = (
# MAGIC         "import pandas as pd\n"
# MAGIC         "data = {'col1': [1, 2], 'col2': [3, 4]}\n"
# MAGIC         "df = pd.DataFrame(data)\n"
# MAGIC         "print(df.shape)"
# MAGIC     )
# MAGIC     assert python_exec(code=code)["value"].strip() == "(2, 2)"
# MAGIC
# MAGIC
# MAGIC def test_using_numpy(python_exec):
# MAGIC     code = "import numpy as np\n" "arr = np.array([1, 2, 3])\n" "print(arr.mean())"
# MAGIC     assert python_exec(code=code)["value"].strip() == "2.0"
# MAGIC
# MAGIC
# MAGIC def test_syntax_error(python_exec):
# MAGIC     code = "if True\n" "    print('invalid syntax')"
# MAGIC     result = python_exec(code=code)
# MAGIC     assert "Syntax error at or near 'invalid'." in result["error"]["error_message"]
# MAGIC
# MAGIC
# MAGIC def test_runtime_error(python_exec):
# MAGIC     code = "x = 1 / 0\n" "print(x)"
# MAGIC     result = python_exec(code=code)
# MAGIC     assert "ZeroDivisionError" in result["error"]["error_message"]
# MAGIC
# MAGIC
# MAGIC def test_undefined_variable(python_exec):
# MAGIC     code = "print(undefined_variable)"
# MAGIC     result = python_exec(code=code)
# MAGIC     assert "NameError" in result["error"]["error_message"]
# MAGIC
# MAGIC
# MAGIC def test_multiline_string_manipulation(python_exec):
# MAGIC     code = "text = '''\n" "Hello\n" "World\n" "'''\n" "print(text.strip())"
# MAGIC     expected = "Hello\nWorld"
# MAGIC     assert python_exec(code=code)["value"].strip() == expected
# MAGIC
# MAGIC
# MAGIC def test_unauthorized_flask(python_exec):
# MAGIC     code = "from flask import Flask\n" "app = Flask(__name__)\n" "print(app)"
# MAGIC     result = python_exec(code=code)
# MAGIC     assert (
# MAGIC         "ModuleNotFoundError: No module named 'flask'"
# MAGIC         in result["error"]["error_message"]
# MAGIC     )
# MAGIC
# MAGIC
# MAGIC def test_no_print_statement(python_exec):
# MAGIC     code = "x = 42\n" "y = x * 2"
# MAGIC     assert python_exec(code=code)["value"] == ""
# MAGIC
# MAGIC
# MAGIC def test_calculation_without_print(python_exec):
# MAGIC     code = "result = sum([1, 2, 3, 4, 5])\n" "squared = [x**2 for x in range(5)]"
# MAGIC     assert python_exec(code=code)["value"] == ""
# MAGIC
# MAGIC
# MAGIC def test_function_definition_without_call(python_exec):
# MAGIC     code = "def add(a, b):\n" "    return a + b\n" "result = add(3, 4)"
# MAGIC     assert python_exec(code=code)["value"] == ""
# MAGIC
# MAGIC
# MAGIC def test_class_definition_without_instantiation(python_exec):
# MAGIC     code = (
# MAGIC         "class Calculator:\n"
# MAGIC         "    def add(self, a, b):\n"
# MAGIC         "        return a + b\n"
# MAGIC         "calc = Calculator()"
# MAGIC     )
# MAGIC     assert python_exec(code=code)["value"] == ""
# MAGIC

# COMMAND ----------

import pytest

# Run tests from test_sku_translator.py
pytest.main(["-v", "tools/test_code_exec_as_uc_tool.py"])

