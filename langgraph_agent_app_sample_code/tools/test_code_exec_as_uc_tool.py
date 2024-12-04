import pytest
from cookbook.tools.uc_tool import UCTool
import os

CATALOG = "ep"
SCHEMA = "cookbook_local_test"


if os.getenv("GITHUB_ACTIONS") == "true":
    pytest.skip("Skipping all tests in this module in CI, as "
                "Databricks auth is not available there", allow_module_level=True)

@pytest.fixture
def python_exec():
    """Fixture to provide the python_exec function from UCTool."""
    python_exec_tool = UCTool(uc_function_name=f"{CATALOG}.{SCHEMA}.python_exec")
    return python_exec_tool


def test_basic_arithmetic(python_exec):
    code = """result = 2 + 2\nprint(result)"""
    assert python_exec(code=code)["value"].strip() == "4"


def test_multiple_lines(python_exec):
    code = "x = 5\n" "y = 3\n" "result = x * y\n" "print(result)"
    assert python_exec(code=code)["value"].strip() == "15"


def test_multiple_prints(python_exec):
    code = """print('first')\nprint('second')\nprint('third')\n"""
    expected = "first\nsecond\nthird\n"
    assert python_exec(code=code)["value"] == expected


def test_using_pandas(python_exec):
    code = (
        "import pandas as pd\n"
        "data = {'col1': [1, 2], 'col2': [3, 4]}\n"
        "df = pd.DataFrame(data)\n"
        "print(df.shape)"
    )
    assert python_exec(code=code)["value"].strip() == "(2, 2)"


def test_using_numpy(python_exec):
    code = "import numpy as np\n" "arr = np.array([1, 2, 3])\n" "print(arr.mean())"
    assert python_exec(code=code)["value"].strip() == "2.0"


def test_syntax_error(python_exec):
    code = "if True\n" "    print('invalid syntax')"
    result = python_exec(code=code)
    assert "Syntax error at or near 'invalid'." in result["error"]["error_message"]


def test_runtime_error(python_exec):
    code = "x = 1 / 0\n" "print(x)"
    result = python_exec(code=code)
    assert "ZeroDivisionError" in result["error"]["error_message"]


def test_undefined_variable(python_exec):
    code = "print(undefined_variable)"
    result = python_exec(code=code)
    assert "NameError" in result["error"]["error_message"]


def test_multiline_string_manipulation(python_exec):
    code = "text = '''\n" "Hello\n" "World\n" "'''\n" "print(text.strip())"
    expected = "Hello\nWorld"
    assert python_exec(code=code)["value"].strip() == expected


def test_unauthorized_flask(python_exec):
    code = "from flask import Flask\n" "app = Flask(__name__)\n" "print(app)"
    result = python_exec(code=code)
    assert (
        "ModuleNotFoundError: No module named 'flask'"
        in result["error"]["error_message"]
    )


def test_no_print_statement(python_exec):
    code = "x = 42\n" "y = x * 2"
    assert python_exec(code=code)["value"] == ""


def test_calculation_without_print(python_exec):
    code = "result = sum([1, 2, 3, 4, 5])\n" "squared = [x**2 for x in range(5)]"
    assert python_exec(code=code)["value"] == ""


def test_function_definition_without_call(python_exec):
    code = "def add(a, b):\n" "    return a + b\n" "result = add(3, 4)"
    assert python_exec(code=code)["value"] == ""


def test_class_definition_without_instantiation(python_exec):
    code = (
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "calc = Calculator()"
    )
    assert python_exec(code=code)["value"] == ""
