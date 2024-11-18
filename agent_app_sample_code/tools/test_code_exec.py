import pytest
from .code_exec import python_exec


def test_basic_arithmetic():
    code = """result = 2 + 2\nprint(result)"""
    assert python_exec(code).strip() == "4"


def test_multiple_lines():
    code = "x = 5\n" "y = 3\n" "result = x * y\n" "print(result)"
    assert python_exec(code).strip() == "15"


def test_multiple_prints():
    code = """print('first')\nprint('second')\nprint('third')\n"""
    expected = "first\nsecond\nthird\n"
    assert python_exec(code) == expected


def test_using_pandas():
    code = (
        "import pandas as pd\n"
        "data = {'col1': [1, 2], 'col2': [3, 4]}\n"
        "df = pd.DataFrame(data)\n"
        "print(df.shape)"
    )
    assert python_exec(code).strip() == "(2, 2)"


def test_using_numpy():
    code = "import numpy as np\n" "arr = np.array([1, 2, 3])\n" "print(arr.mean())"
    assert python_exec(code).strip() == "2.0"


def test_syntax_error():
    code = "if True\n" "    print('invalid syntax')"
    with pytest.raises(SyntaxError):
        python_exec(code)


def test_runtime_error():
    code = "x = 1 / 0\n" "print(x)"
    with pytest.raises(ZeroDivisionError):
        python_exec(code)


def test_undefined_variable():
    code = "print(undefined_variable)"
    with pytest.raises(NameError):
        python_exec(code)


def test_multiline_string_manipulation():
    code = "text = '''\n" "Hello\n" "World\n" "'''\n" "print(text.strip())"
    expected = "Hello\nWorld"
    assert python_exec(code).strip() == expected


# Will not fail locally, but will fail in UC.
# def test_unauthorized_flaskxx():
#     code = "from flask import Flask\n" "app = Flask(__name__)\n" "print(app)"
#     with pytest.raises(ImportError):
#         python_exec(code)


def test_no_print_statement():
    code = "x = 42\n" "y = x * 2"
    assert python_exec(code) == ""


def test_calculation_without_print():
    code = "result = sum([1, 2, 3, 4, 5])\n" "squared = [x**2 for x in range(5)]"
    assert python_exec(code) == ""


def test_function_definition_without_call():
    code = "def add(a, b):\n" "    return a + b\n" "result = add(3, 4)"
    assert python_exec(code) == ""


def test_class_definition_without_instantiation():
    code = (
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "calc = Calculator()"
    )
    assert python_exec(code) == ""
