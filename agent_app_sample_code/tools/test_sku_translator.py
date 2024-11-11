import pytest
from tools.sku_translator import translate_sku


def test_valid_sku_translation():
    """Test successful SKU translation with valid input."""
    assert translate_sku("OLD-ABC-1234") == "NEW-1234-ABC"
    assert translate_sku("OLD-XYZ-0001") == "NEW-0001-XYZ"
    assert translate_sku("old-def-5678") == "NEW-5678-DEF"  # Test case insensitivity


def test_whitespace_handling():
    """Test that the function handles extra whitespace correctly."""
    assert translate_sku("  OLD-ABC-1234  ") == "NEW-1234-ABC"
    assert translate_sku("\tOLD-ABC-1234\n") == "NEW-1234-ABC"


def test_invalid_input_type():
    """Test that non-string inputs raise ValueError."""
    with pytest.raises(ValueError, match="SKU must be a string"):
        translate_sku(123)
    with pytest.raises(ValueError, match="SKU must be a string"):
        translate_sku(None)


def test_invalid_prefix():
    """Test that SKUs not starting with 'OLD-' raise ValueError."""
    with pytest.raises(ValueError, match="SKU must start with 'OLD-'"):
        translate_sku("NEW-ABC-1234")
    with pytest.raises(ValueError, match="SKU must start with 'OLD-'"):
        translate_sku("XXX-ABC-1234")


def test_invalid_format():
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

    for sku in invalid_skus:
        with pytest.raises(
            ValueError,
            match="SKU format must be 'OLD-XXX-YYYY' where X is a letter and Y is a digit",
        ):
            translate_sku(sku)
