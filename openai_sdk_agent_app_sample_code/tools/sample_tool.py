
def sku_sample_translator(old_sku: str) -> str:
    """
    Translates a pre-2024 SKU formatted as "OLD-XXX-YYYY" to the new SKU format "NEW-YYYY-XXX".

    Args:
        old_sku (str): The old SKU in the format "OLD-XXX-YYYY".

    Returns:
        str: The new SKU in the format "NEW-YYYY-XXX".

    Raises:
        ValueError: If the SKU format is invalid, providing specific error details.
    """
    import re

    if not isinstance(old_sku, str):
        raise ValueError("SKU must be a string")

    # Normalize input by removing extra whitespace and converting to uppercase
    old_sku = old_sku.strip().upper()

    # Define the regex pattern for the old SKU format
    pattern = r"^OLD-([A-Z]{3})-(\d{4})$"

    # Match the old SKU against the pattern
    match = re.match(pattern, old_sku)
    if not match:
        if not old_sku.startswith("OLD-"):
            raise ValueError("SKU must start with 'OLD-'")
        if not re.match(r"^OLD-[A-Z]{3}-\d{4}$", old_sku):
            raise ValueError(
                "SKU format must be 'OLD-XXX-YYYY' where X is a letter and Y is a digit"
            )
        raise ValueError("Invalid SKU format")

    # Extract the letter code and numeric part
    letter_code, numeric_part = match.groups()

    # Additional validation for numeric part
    if not (1 <= int(numeric_part) <= 9999):
        raise ValueError("Numeric part must be between 0001 and 9999")

    # Construct the new SKU
    new_sku = f"NEW-{numeric_part}-{letter_code}"
    return new_sku
