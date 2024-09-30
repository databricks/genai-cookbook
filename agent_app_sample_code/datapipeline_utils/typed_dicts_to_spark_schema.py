from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    BooleanType,
    ArrayType,
    TimestampType,
    DateType,
)
from typing import TypedDict, get_type_hints, List
from datetime import datetime, date, time


def typed_dict_to_spark_fields(typed_dict: type[TypedDict]) -> StructType:
    """
    Converts a TypedDict into a list of Spark StructField objects.

    This function maps Python types defined in a TypedDict to their corresponding
    Spark SQL data types, facilitating the creation of a Spark DataFrame schema
    from Python type annotations.

    Parameters:
    - typed_dict (type[TypedDict]): The TypedDict class to be converted.

    Returns:
    - StructType: A list of StructField objects representing the Spark schema.

    Raises:
    - ValueError: If an unsupported type is encountered or if dictionary types are used.
    """

    # Mapping of type names to Spark type objects
    type_mapping = {
        str: StringType(),
        int: IntegerType(),
        float: DoubleType(),
        bool: BooleanType(),
        list: ArrayType(StringType()),  # Default to StringType for arrays
        datetime: TimestampType(),
        date: DateType(),
    }

    def get_spark_type(value_type):
        """
        Helper function to map a Python type to a Spark SQL data type.

        This function supports basic Python types, lists of a single type, and raises
        an error for unsupported types or dictionaries.

        Parameters:
        - value_type: The Python type to be converted.

        Returns:
        - DataType: The corresponding Spark SQL data type.

        Raises:
        - ValueError: If the type is unsupported or if dictionary types are used.
        """
        if value_type in type_mapping:
            return type_mapping[value_type]
        elif hasattr(value_type, "__origin__") and value_type.__origin__ == list:
            # Handle List[type] types
            return ArrayType(get_spark_type(value_type.__args__[0]))
        elif hasattr(value_type, "__origin__") and value_type.__origin__ == dict:
            # Handle Dict[type, type] types (not fully supported)
            raise ValueError("Dict types are not fully supported")
        else:
            raise ValueError(f"Unsupported type: {value_type}")

    # Get the type hints for the TypedDict
    type_hints = get_type_hints(typed_dict)

    # Convert the type hints into a list of StructField objects
    fields = [
        StructField(key, get_spark_type(value), True)
        for key, value in type_hints.items()
    ]

    # Create and return the StructType object
    return fields
  
def typed_dicts_to_spark_schema(*typed_dicts: type[TypedDict]) -> StructType:
    """
    Converts multiple TypedDicts into a Spark schema.

    This function allows for the combination of multiple TypedDicts into a single
    Spark DataFrame schema, enabling the creation of complex data structures.

    Parameters:
    - *typed_dicts: Variable number of TypedDict classes to be converted.

    Returns:
    - StructType: A Spark schema represented as a StructType object, which is a collection
      of StructField objects derived from the provided TypedDicts.
    """
    fields = []
    for typed_dict in typed_dicts:
        fields.extend(typed_dict_to_spark_fields(typed_dict))

    return StructType(fields)