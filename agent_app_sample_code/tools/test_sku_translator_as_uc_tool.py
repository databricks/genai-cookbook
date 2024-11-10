import pytest
from cookbook.tools.uc_tool import UCTool


# Load the function from the UCTool versus locally
@pytest.fixture
def translate_sku():
    """Fixture to provide the translate_sku function from UCTool."""
    translate_sku_tool = UCTool(uc_function_name="ep.cookbook_local_test.translate_sku")
    return translate_sku_tool


# Note: The value will be post processed into the `value` key, so we must check the returned value there.
def test_valid_sku_translation(translate_sku):
    """Test successful SKU translation with valid input."""
    assert translate_sku(old_sku="OLD-ABC-1234")["value"] == "NEW-1234-ABC"
    assert translate_sku(old_sku="OLD-XYZ-0001")["value"] == "NEW-0001-XYZ"
    assert (
        translate_sku(old_sku="old-def-5678")["value"] == "NEW-5678-DEF"
    )  # Test case insensitivity


# Note: The value will be post processed into the `value` key, so we must check the returned value there.
def test_whitespace_handling(translate_sku):
    """Test that the function handles extra whitespace correctly."""
    assert translate_sku(old_sku="  OLD-ABC-1234  ")["value"] == "NEW-1234-ABC"
    assert translate_sku(old_sku="\tOLD-ABC-1234\n")["value"] == "NEW-1234-ABC"


# Note: the input validation happens BEFORE the function is called by Spark, so we will never get these exceptions from the function.
# Instead, we will get invalid parameters errors from Spark.
def test_invalid_input_type(translate_sku):
    """Test that non-string inputs raise ValueError."""
    assert (
        translate_sku(old_sku=123)["error"]["error_message"]
        == """Invalid parameters provided: {'old_sku': "Parameter old_sku should be of type STRING (corresponding python type <class 'str'>), but got <class 'int'>"}."""
    )
    assert (
        translate_sku(old_sku=None)["error"]["error_message"]
        == """Invalid parameters provided: {'old_sku': "Parameter old_sku should be of type STRING (corresponding python type <class 'str'>), but got <class 'NoneType'>"}."""
    )


# Note: The errors will be post processed into the `error_message` key inside the `error` top level key, so we must check for exceptions there.
def test_invalid_prefix(translate_sku):
    """Test that SKUs not starting with 'OLD-' raise ValueError."""
    assert (
        translate_sku(old_sku="NEW-ABC-1234")["error"]["error_message"]
        == "ValueError: SKU must start with 'OLD-'"
    )
    assert (
        translate_sku(old_sku="XXX-ABC-1234")["error"]["error_message"]
        == "ValueError: SKU must start with 'OLD-'"
    )


# Note: The errors will be post processed into the `error_message` key inside the `error` top level key, so we must check for exceptions there.
def test_invalid_format(translate_sku):
    """Test various invalid SKU formats."""
    invalid_skus = [
        "OLD-AB-1234",  # Too few letters
        "OLD-ABCD-1234",  # Too many letters
        "OLD-123-1234",  # Numbers instead of letters
        "OLD-ABC-123",  # Too few digits
        "OLD-ABC-12345",  # Too many digits
        "OLD-ABC-XXXX",  # Letters instead of numbers
        "OLD-A1C-1234",  # Mixed letters and numbers in middle
    ]

    expected_error = "ValueError: SKU format must be 'OLD-XXX-YYYY' where X is a letter and Y is a digit"
    for sku in invalid_skus:
        assert translate_sku(old_sku=sku)["error"]["error_message"] == expected_error
